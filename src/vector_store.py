import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lancedb
import pyarrow as pa

from src.config import DEFAULT_TABLE_NAME, LANCEDB_PATH

logger = logging.getLogger(__name__)


class LanceDBManager:
    """
    LanceDB Table Manager supporting:
    - Table initialization & schema creation
    - Vector insertion
    - Vector ANN search
    - Full-Text BM25 keyword search
    - Hybrid ANN Vector + BM25 search with Reciprocal Rank Fusion (RRF)
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or LANCEDB_PATH).resolve()
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))

    def get_table(self, table_name: str = DEFAULT_TABLE_NAME):
        """Get an existing table from LanceDB."""
        if table_name in self.db.table_names():
            return self.db.open_table(table_name)
        return None

    def init_table(
        self,
        table_name: str = DEFAULT_TABLE_NAME,
        sample_chunk: Optional[Dict[str, Any]] = None,
        sample_vector: Optional[List[float]] = None,
        mode: str = "create",
    ):
        """
        Initialize or reset a table in LanceDB.
        If mode is 'overwrite', drops the table if it already exists.
        """
        existing_tables = self.db.table_names()
        if mode == "overwrite" and table_name in existing_tables:
            self.db.drop_table(table_name)
            logger.info(f"Dropped existing table '{table_name}'.")

        if table_name in self.db.table_names():
            logger.info(f"Table '{table_name}' already exists.")
            return self.db.open_table(table_name)

        # Build sample schema if sample vector provided
        if sample_vector is not None and sample_chunk is not None:
            dim = len(sample_vector)
            schema = pa.schema([
                ("id", pa.string()),
                ("doc_id", pa.string()),
                ("source", pa.string()),
                ("file_type", pa.string()),
                ("page_number", pa.int64()),
                ("chunk_index", pa.int64()),
                ("content", pa.string()),
                ("image_path", pa.string()),
                ("created_at", pa.string()),
                ("metadata_json", pa.string()),
                ("vector", pa.list_(pa.float32(), dim)),
            ])
            data = [{
                **sample_chunk,
                "vector": sample_vector,
            }]
            table = self.db.create_table(table_name, schema=schema, mode="overwrite")
            logger.info(f"Created table '{table_name}' with vector dimension {dim}.")
            return table
        
        logger.info(f"Table '{table_name}' initialized lazily.")
        return None

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        vectors: List[List[float]],
        table_name: str = DEFAULT_TABLE_NAME,
    ):
        """Insert metadata-rich chunks and their embeddings into the vector store."""
        if not chunks or not vectors:
            logger.warning("Empty chunks or vectors provided to add_chunks.")
            return

        if len(chunks) != len(vectors):
            raise ValueError(f"Mismatch between number of chunks ({len(chunks)}) and vectors ({len(vectors)}).")

        records = []
        for chunk, vector in zip(chunks, vectors):
            record = {
                "id": chunk.get("id", ""),
                "doc_id": chunk.get("doc_id", ""),
                "source": chunk.get("source", ""),
                "file_type": chunk.get("file_type", "text"),
                "page_number": chunk.get("page_number") if chunk.get("page_number") is not None else -1,
                "chunk_index": chunk.get("chunk_index", 0),
                "content": chunk.get("content", ""),
                "image_path": chunk.get("image_path") or "",
                "created_at": chunk.get("created_at", ""),
                "metadata_json": chunk.get("metadata_json", json.dumps(chunk)),
                "vector": [float(x) for x in vector],
            }
            records.append(record)

        if table_name not in self.db.table_names():
            # Initialize with first record
            self.init_table(table_name, sample_chunk=chunks[0], sample_vector=vectors[0], mode="overwrite")
            table = self.db.open_table(table_name)
            if len(records) > 1:
                table.add(records[1:])
        else:
            table = self.db.open_table(table_name)
            table.add(records)

        logger.info(f"Successfully added {len(records)} chunks to LanceDB table '{table_name}'.")

    def vector_search(
        self,
        query_vector: List[float],
        table_name: str = DEFAULT_TABLE_NAME,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Perform Approximate Nearest Neighbor (ANN) vector search."""
        table = self.get_table(table_name)
        if table is None:
            logger.warning(f"Table '{table_name}' does not exist.")
            return []

        try:
            results = table.search(query_vector).limit(limit).to_list()
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def fts_search(
        self,
        query_text: str,
        table_name: str = DEFAULT_TABLE_NAME,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Perform Full-Text BM25 Keyword Search."""
        table = self.get_table(table_name)
        if table is None:
            logger.warning(f"Table '{table_name}' does not exist.")
            return []

        # Try LanceDB FTS index search
        try:
            # Create FTS index if not created
            try:
                table.create_fts_index("content", replace=False)
            except Exception:
                pass
            
            results = table.search(query_text).limit(limit).to_list()
            return results
        except Exception as e:
            logger.warning(f"LanceDB native FTS search error: {e}. Executing Python keyword search fallback.")
            return self._python_bm25_fallback(query_text, table, limit)

    def _python_bm25_fallback(
        self,
        query_text: str,
        table: Any,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Fallback lightweight Python term frequency keyword search over content."""
        try:
            all_records = table.to_arrow().to_pylist()
        except Exception as e:
            logger.error(f"Error fetching records for keyword fallback: {e}")
            return []

        query_terms = set(query_text.lower().split())
        scored_records = []

        for record in all_records:
            content = record.get("content", "").lower()
            score = 0.0
            for term in query_terms:
                if term in content:
                    count = content.count(term)
                    score += 1.0 + math.log(1 + count)
            if score > 0:
                scored_records.append((score, record))

        scored_records.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_records[:limit]]

    def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        table_name: str = DEFAULT_TABLE_NAME,
        limit: int = 5,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid Search combining Vector ANN search and BM25 Full-Text search using Reciprocal Rank Fusion (RRF).
        RRF Score Formula: score(d) = sum(1.0 / (k + rank_i(d)))
        """
        table = self.get_table(table_name)
        if table is None:
            logger.warning(f"Table '{table_name}' does not exist.")
            return []

        # Retrieve top 2*limit candidates from both search modalities
        candidate_limit = max(limit * 2, 20)
        vec_results = self.vector_search(query_vector, table_name=table_name, limit=candidate_limit)
        fts_results = self.fts_search(query_text, table_name=table_name, limit=candidate_limit)

        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        vec_ranks: Dict[str, int] = {}
        fts_ranks: Dict[str, int] = {}

        # Process Vector search results
        for rank, doc in enumerate(vec_results, start=1):
            doc_id = doc.get("id") or str(hash(doc.get("content", "")))
            doc_map[doc_id] = doc
            vec_ranks[doc_id] = rank
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        # Process FTS search results
        for rank, doc in enumerate(fts_results, start=1):
            doc_id = doc.get("id") or str(hash(doc.get("content", "")))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            fts_ranks[doc_id] = rank
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        # Sort combined results by RRF score descending
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d_id: rrf_scores[d_id], reverse=True)

        final_results = []
        for doc_id in sorted_doc_ids[:limit]:
            doc = doc_map[doc_id].copy()
            doc["_rrf_score"] = rrf_scores[doc_id]
            doc["_vector_rank"] = vec_ranks.get(doc_id, None)
            doc["_fts_rank"] = fts_ranks.get(doc_id, None)
            final_results.append(doc)

        logger.info(
            f"Hybrid search returned {len(final_results)} results (Vector count: {len(vec_results)}, FTS count: {len(fts_results)})."
        )
        return final_results
