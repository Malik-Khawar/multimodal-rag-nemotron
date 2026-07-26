from typing import List, Dict, Any
from src.lancedb_store import LanceDBVectorStore

class HybridSearchEngine:
    """
    Hybrid Search combining Dense Vector Search and BM25 Lexical Search via RRF.
    """
    def __init__(self, vector_store: LanceDBVectorStore, k: int = 60):
        self.vector_store = vector_store
        self.k = k

    def reciprocal_rank_fusion(
        self, 
        dense_results: List[Dict[str, Any]], 
        bm25_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculates RRF score: RRF(d) = sum_{m in M} 1 / (k + r_m(d))
        """
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        # Process dense results
        for doc in dense_results:
            doc_id = doc["doc_id"]
            rank = doc["dense_rank"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.k + rank))
            doc_map[doc_id] = doc

        # Process BM25 results
        for doc in bm25_results:
            doc_id = doc["doc_id"]
            rank = doc["bm25_rank"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.k + rank))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        fused_results = []
        for doc_id, rrf_score in rrf_scores.items():
            item = doc_map[doc_id].copy()
            item["rrf_score"] = float(round(rrf_score, 6))
            fused_results.append(item)

        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused_results

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        dense_res = self.vector_store.dense_search(query, top_k=top_k)
        bm25_res = self.vector_store.bm25_search(query, top_k=top_k)
        return self.reciprocal_rank_fusion(dense_res, bm25_res)
