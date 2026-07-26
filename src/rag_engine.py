"""
4-Stage Advanced Multimodal RAG Engine
Stage 1: Hybrid Retrieval (LanceDB Dense Vector + BM25 Sparse Lexical)
Stage 2: Nemotron Cross-Encoder Re-Ranking
Stage 3: Multimodal Nemotron Synthesis (Ultra-550B / Nano-Omni-30B)
Stage 4: Automatic Failover (Ling-3.0-Flash) & Structured Execution Trace
"""

import os
import time
import json
import logging
import uuid
import re
import math
import httpx
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Vector Store & Search Imports
try:
    import lancedb
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from src.reranker import NemotronReranker

logger = logging.getLogger("multimodal_rag.engine")
logging.basicConfig(level=logging.INFO)


class FallbackEmbedder:
    """
    Lightweight deterministic feature embedder fallback when sentence-transformers is unavailable.
    Outputs 384-dimensional normalized TF-IDF / term-hashing vector.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, sentences: List[str] | str, **kwargs) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]

        vectors = []
        for text in sentences:
            vec = np.zeros(self.dim, dtype=np.float32)
            words = re.findall(r"\w+", text.lower())
            if words:
                for word in words:
                    # Simple Murmur-style hash to feature index
                    idx = abs(hash(word)) % self.dim
                    vec[idx] += 1.0
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)

        arr = np.array(vectors, dtype=np.float32)
        return arr[0] if len(sentences) == 1 and not isinstance(sentences, list) else arr


class MultimodalRAGEngine:
    """
    Production-grade 4-Stage Advanced RAG Orchestrator with Nemotron Models.
    """

    def __init__(
        self,
        db_path: str = "./data/lancedb_store",
        openrouter_api_key: Optional[str] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        os.makedirs("./data", exist_ok=True)

        self.api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

        # Model definitions
        self.MODEL_MULTIMODAL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
        self.MODEL_TEXT_REASONING = "nvidia/nemotron-3-ultra-550b-a55b:free"
        self.MODEL_FAILOVER = "inclusionai/ling-3.0-flash:free"

        # Initialize Embedder (Fast load with FallbackEmbedder support)
        self.embedder = None
        use_fast = os.environ.get("FAST_EMBED", "1") == "1"
        if not use_fast and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading embedding model: {embedding_model_name}")
                self.embedder = SentenceTransformer(embedding_model_name)
                self.embed_dim = self.embedder.get_sentence_embedding_dimension()
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer ({e}). Using FallbackEmbedder.")
                self.embedder = FallbackEmbedder(384)
                self.embed_dim = 384
        else:
            logger.info("Using FallbackEmbedder for instant high-throughput indexing.")
            self.embedder = FallbackEmbedder(384)
            self.embed_dim = 384

        # Initialize LanceDB
        self.db = None
        self.table = None
        self._init_lancedb()

        # BM25 In-Memory Store
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunks_store: List[Dict[str, Any]] = []
        self._rebuild_bm25()

        # Stage 2 Re-ranker
        self.reranker = NemotronReranker(api_key=self.api_key)

    def _init_lancedb(self):
        """Initialize LanceDB database connection."""
        if LANCEDB_AVAILABLE:
            try:
                self.db = lancedb.connect(self.db_path)
                if "rag_chunks" in self.db.table_names():
                    self.table = self.db.open_table("rag_chunks")
                    logger.info("Opened existing LanceDB table 'rag_chunks'")
            except Exception as e:
                logger.error(f"Error initializing LanceDB: {e}")
                self.db = None

    def _rebuild_bm25(self):
        """Rebuild BM25 sparse search index from chunks_store."""
        if not self.chunks_store:
            self.bm25_index = None
            return

        corpus = [chunk["text"].lower().split() for chunk in self.chunks_store]
        if BM25_AVAILABLE and corpus:
            self.bm25_index = BM25Okapi(corpus)

    def ingest_document(self, text: str, source_name: str = "upload.txt", metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Splits, embeds, and indexes document into LanceDB and BM25 store.

        Returns:
            Number of chunks indexed.
        """
        if not text or not text.strip():
            return 0

        # Chunk text (500 chars with 50 char overlap)
        chunk_size = 500
        overlap = 50
        text_clean = text.strip()
        
        raw_chunks = []
        start = 0
        while start < len(text_clean):
            end = start + chunk_size
            chunk_str = text_clean[start:end]
            if chunk_str.strip():
                raw_chunks.append(chunk_str.strip())
            start += (chunk_size - overlap)

        if not raw_chunks:
            return 0

        # Embed chunks
        embeddings = self.embedder.encode(raw_chunks)
        if isinstance(embeddings, np.ndarray) and embeddings.ndim == 1:
            embeddings = [embeddings]

        new_entries = []
        for idx, (chunk_text, emb) in enumerate(zip(raw_chunks, embeddings)):
            chunk_id = f"{uuid.uuid4().hex[:8]}_{idx}"
            entry = {
                "id": chunk_id,
                "text": chunk_text,
                "vector": emb.tolist() if hasattr(emb, "tolist") else list(emb),
                "source": source_name,
                "chunk_index": idx,
                "metadata": json.dumps(metadata or {}),
            }
            new_entries.append(entry)
            self.chunks_store.append(entry)

        # Update LanceDB
        if LANCEDB_AVAILABLE and self.db is not None:
            try:
                if "rag_chunks" not in self.db.table_names():
                    self.table = self.db.create_table("rag_chunks", data=new_entries)
                else:
                    self.table.add(new_entries)
            except Exception as e:
                logger.error(f"Failed writing to LanceDB: {e}")

        # Update BM25 index
        self._rebuild_bm25()

        return len(new_entries)

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on LanceDB vector store & pipeline state."""
        lancedb_count = 0
        if self.table is not None:
            try:
                lancedb_count = len(self.table)
            except Exception:
                lancedb_count = len(self.chunks_store)
        else:
            lancedb_count = len(self.chunks_store)

        sources = list(set(c["source"] for c in self.chunks_store)) if self.chunks_store else []

        return {
            "total_chunks": lancedb_count,
            "total_documents": len(sources),
            "vector_dim": self.embed_dim,
            "db_path": os.path.abspath(self.db_path),
            "lancedb_active": LANCEDB_AVAILABLE and self.db is not None,
            "bm25_active": self.bm25_index is not None,
            "embedder": type(self.embedder).__name__,
            "primary_models": {
                "multimodal": self.MODEL_MULTIMODAL,
                "text_reasoning": self.MODEL_TEXT_REASONING,
                "reranker": self.reranker.model,
                "failover": self.MODEL_FAILOVER,
            },
            "sources": sources[:10],
        }

    def _stage1_hybrid_retrieval(self, query: str, top_k: int = 10) -> tuple[List[Dict[str, Any]], float]:
        """
        Stage 1: Dense Vector (LanceDB) + Sparse Lexical (BM25) with RRF / Score Fusion.
        """
        start_time = time.perf_counter()
        if not self.chunks_store:
            return [], round((time.perf_counter() - start_time) * 1000, 2)

        # 1. Dense Vector Retrieval
        dense_results: Dict[str, Dict[str, Any]] = {}
        query_vec = self.embedder.encode(query)
        if hasattr(query_vec, "tolist"):
            query_vec = query_vec.tolist()

        if self.table is not None:
            try:
                res = self.table.search(query_vec).limit(top_k * 2).to_list()
                for rank, r in enumerate(res):
                    cid = r["id"]
                    # Calculate similarity from distance if available
                    dist = r.get("_distance", 1.0)
                    sim_score = max(0.0, 1.0 - (dist / 2.0))
                    dense_results[cid] = {
                        "id": cid,
                        "text": r["text"],
                        "source": r.get("source", "unknown"),
                        "dense_score": round(sim_score, 4),
                        "dense_rank": rank + 1,
                    }
            except Exception as e:
                logger.warning(f"LanceDB search failed: {e}. Falling back to memory cosine search.")
                dense_results = self._memory_vector_search(query_vec, top_k * 2)
        else:
            dense_results = self._memory_vector_search(query_vec, top_k * 2)

        # 2. Sparse BM25 Retrieval
        sparse_results: Dict[str, Dict[str, Any]] = {}
        if self.bm25_index:
            tokenized_query = query.lower().split()
            scores = self.bm25_index.get_scores(tokenized_query)
            top_bm25_indices = np.argsort(scores)[::-1][: top_k * 2]

            max_bm25 = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0
            for rank, idx in enumerate(top_bm25_indices):
                if idx < len(self.chunks_store):
                    chunk = self.chunks_store[idx]
                    cid = chunk["id"]
                    norm_score = round(float(scores[idx] / max_bm25), 4)
                    sparse_results[cid] = {
                        "id": cid,
                        "text": chunk["text"],
                        "source": chunk.get("source", "unknown"),
                        "sparse_score": norm_score,
                        "sparse_rank": rank + 1,
                    }

        # 3. Hybrid Fusion (Reciprocal Rank Fusion k=60 + Linear Score Blend)
        all_cids = set(dense_results.keys()).union(set(sparse_results.keys()))
        hybrid_candidates = []

        k = 60
        for cid in all_cids:
            d_item = dense_results.get(cid, {})
            s_item = sparse_results.get(cid, {})

            d_rank = d_item.get("dense_rank", 999)
            s_rank = s_item.get("sparse_rank", 999)

            rrf_score = (1.0 / (k + d_rank)) + (1.0 / (k + s_rank))
            d_score = d_item.get("dense_score", 0.0)
            s_score = s_item.get("sparse_score", 0.0)

            # Combined hybrid score normalized
            hybrid_score = round((0.6 * d_score) + (0.4 * s_score), 4)

            text = d_item.get("text") or s_item.get("text", "")
            source = d_item.get("source") or s_item.get("source", "unknown")

            hybrid_candidates.append({
                "id": cid,
                "text": text,
                "source": source,
                "dense_score": d_score,
                "sparse_score": s_score,
                "rrf_score": round(rrf_score, 6),
                "score": hybrid_score,  # unified candidate score for stage 2
            })

        # Sort by hybrid score
        hybrid_candidates.sort(key=lambda x: (x["score"], x["rrf_score"]), reverse=True)
        top_candidates = hybrid_candidates[:top_k]

        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        return top_candidates, elapsed

    def _memory_vector_search(self, query_vec: List[float], top_k: int) -> Dict[str, Dict[str, Any]]:
        """In-memory cosine vector search fallback."""
        if not self.chunks_store:
            return {}

        q = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        results = {}
        scores = []
        for idx, chunk in enumerate(self.chunks_store):
            v = np.array(chunk["vector"], dtype=np.float32)
            v_norm = np.linalg.norm(v)
            if v_norm > 0:
                v = v / v_norm
            sim = float(np.dot(q, v))
            scores.append((sim, idx, chunk))

        scores.sort(key=lambda x: x[0], reverse=True)
        for rank, (sim, idx, chunk) in enumerate(scores[:top_k]):
            cid = chunk["id"]
            results[cid] = {
                "id": cid,
                "text": chunk["text"],
                "source": chunk.get("source", "unknown"),
                "dense_score": round(max(0.0, sim), 4),
                "dense_rank": rank + 1,
            }
        return results

    def _stage3_and_4_synthesis(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        multimodal_image: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Stage 3: Multimodal / Text Reasoning Synthesis with Nemotron Models.
        Stage 4: Automatic Failover to Ling-3.0-Flash on rate limits or API errors.
        """
        start_time = time.perf_counter()
        is_multimodal = bool(multimodal_image)

        # Model selection: Nano-Omni-30B for multimodal, Ultra-550B for text reasoning
        primary_model = self.MODEL_MULTIMODAL if is_multimodal else self.MODEL_TEXT_REASONING

        # Context assembly
        context_passages = []
        for idx, chunk in enumerate(retrieved_chunks):
            context_passages.append(f"[{idx+1}] (Source: {chunk.get('source', 'doc')}) {chunk.get('text', '')}")

        context_str = "\n\n".join(context_passages) if context_passages else "No external context retrieved."
        
        system_instruction = (
            "You are an expert AI assistant powered by NVIDIA Nemotron architecture. "
            "Synthesize an insightful, accurate, and structured answer using the provided retrieved context. "
            "If context passages are referenced, cite them clearly using bracketed numbers like [1]. "
            "Be clear, precise, and highly analytical."
        )

        user_content: List[Dict[str, Any]] | str
        if is_multimodal:
            user_content = [
                {"type": "text", "text": f"Context:\n{context_str}\n\nUser Question: {query}"},
            ]
            if multimodal_image.startswith("data:image"):
                user_content.append({"type": "image_url", "image_url": {"url": multimodal_image}})
            elif multimodal_image.startswith("http://") or multimodal_image.startswith("https://"):
                user_content.append({"type": "image_url", "image_url": {"url": multimodal_image}})
            else:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{multimodal_image}"}})
        else:
            user_content = f"Context:\n{context_str}\n\nUser Question: {query}"

        # Function to execute completion call
        def _call_openrouter(model_name: str) -> Tuple[Optional[str], Optional[str]]:
            if not self.api_key or self.api_key == "sk-placeholder":
                return None, "No API key configured"

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/khawar-hasnain/multimodal-rag-nemotron",
                "X-Title": "Multimodal RAG Nemotron Pipeline",
            }
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content if isinstance(user_content, str) else user_content},
            ]
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 1000,
            }

            try:
                with httpx.Client(timeout=4.0) as client:
                    resp = client.post(self.api_url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        ans = data["choices"][0]["message"]["content"]
                        return ans, None
                    else:
                        return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as ex:
                return None, f"Exception: {str(ex)}"

        # Attempt Stage 3: Primary Nemotron Model
        synthesis_res, error_reason = _call_openrouter(primary_model)

        failover_triggered = False
        failover_reason = None
        final_model_used = primary_model

        if synthesis_res is not None:
            answer = synthesis_res
        else:
            # Stage 4: Automatic Failover Triggered!
            failover_triggered = True
            failover_reason = f"Primary model ({primary_model}) failed: {error_reason}"
            logger.warning(f"Stage 4 Triggered! Failover to {self.MODEL_FAILOVER}. Reason: {failover_reason}")

            failover_res, failover_err = _call_openrouter(self.MODEL_FAILOVER)
            if failover_res is not None:
                answer = failover_res
                final_model_used = self.MODEL_FAILOVER
            else:
                # Local heuristic fallback synthesis if remote failover API fails/no key
                final_model_used = "local-synthesis-fallback"
                answer = self._local_synthesis_fallback(query, retrieved_chunks)

        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "answer": answer,
            "model_requested": primary_model,
            "model_used": final_model_used,
            "failover_triggered": failover_triggered,
            "failover_reason": failover_reason,
            "is_multimodal": is_multimodal,
            "latency_ms": elapsed,
        }

    def _local_synthesis_fallback(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Rule-based local synthesis fallback generating readable answer from context."""
        if not chunks:
            return f"I analyzed your request '{query}', but no relevant passages were found in the document index."

        summary_lines = [f"Based on the indexed document context for '{query}':\n"]
        for idx, chunk in enumerate(chunks[:3]):
            src = chunk.get("source", "document")
            txt = chunk.get("text", "").strip()
            summary_lines.append(f"• **Passage [{idx+1}]** ({src}): {txt}")

        summary_lines.append("\n*(Note: Generated via local engine fallback mode as API endpoints were unavailable)*")
        return "\n\n".join(summary_lines)

    def query(
        self,
        query_str: str,
        multimodal_image: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Executes the full 4-Stage Multimodal RAG Pipeline and returns response with detailed RAG trace.
        """
        pipeline_start = time.perf_counter()

        # Stage 1: Hybrid Retrieval
        retrieved_candidates, stage1_latency = self._stage1_hybrid_retrieval(query_str, top_k=top_k * 2)

        # Stage 2: Nemotron Re-Ranking
        rerank_result = self.reranker.rerank(query_str, retrieved_candidates, top_n=top_k)
        top_reranked_chunks = rerank_result["reranked_chunks"]

        # Stage 3 & 4: Synthesis & Failover
        synthesis_result = self._stage3_and_4_synthesis(
            query=query_str,
            retrieved_chunks=top_reranked_chunks,
            multimodal_image=multimodal_image,
        )

        total_latency_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)

        # Structured RAG Trace
        trace = {
            "query": query_str,
            "is_multimodal": bool(multimodal_image),
            "stage1_hybrid_retrieval": {
                "candidate_count": len(retrieved_candidates),
                "candidates": retrieved_candidates,
                "latency_ms": stage1_latency,
            },
            "stage2_nemotron_reranking": {
                "model_used": rerank_result["model_used"],
                "fallback_used": rerank_result["fallback_used"],
                "fallback_reason": rerank_result["fallback_reason"],
                "top_chunks": top_reranked_chunks,
                "latency_ms": rerank_result["latency_ms"],
            },
            "stage3_and_4_synthesis": {
                "model_requested": synthesis_result["model_requested"],
                "model_used": synthesis_result["model_used"],
                "failover_triggered": synthesis_result["failover_triggered"],
                "failover_reason": synthesis_result["failover_reason"],
                "latency_ms": synthesis_result["latency_ms"],
            },
            "total_latency_ms": total_latency_ms,
        }

        return {
            "query": query_str,
            "answer": synthesis_result["answer"],
            "trace": trace,
        }
