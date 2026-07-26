import os
import math
import numpy as np
from typing import List, Dict, Any

class LanceDBVectorStore:
    """
    Mock/In-memory vector store simulating LanceDB vector table index & BM25 retrieval.
    """
    def __init__(self, db_path: str = "./.lancedb"):
        self.db_path = db_path
        self.documents: List[Dict[str, Any]] = []
        self._seed_sample_docs()

    def _seed_sample_docs(self):
        """Seed sample enterprise multimodal documentation."""
        sample_data = [
            {
                "id": "doc_1",
                "title": "NVIDIA Nemotron-4 340B Architecture Overview",
                "content": "NVIDIA Nemotron-4 340B is an open model family designed for high-efficiency synthetic data generation and multimodal alignment. It features 340 billion parameters and optimized FP8 throughput.",
                "modality": "text",
                "category": "Architecture"
            },
            {
                "id": "doc_2",
                "title": "LanceDB Zero-Copy Vector Indexing",
                "content": "LanceDB uses Apache Arrow column storage for zero-copy memory mapping. Combined with IVF-PQ (Inverted File Product Quantization), it provides sub-10ms vector search across millions of embedding vectors.",
                "modality": "text",
                "category": "Vector DB"
            },
            {
                "id": "doc_3",
                "title": "Reciprocal Rank Fusion (RRF) Hybrid Retrieval",
                "content": "Combining dense vector embeddings with sparse BM25 keyword search using RRF formula RRF(d) = sum(1 / (k + r(d))) resolves semantic domain shifts and keyword mismatch problems in complex Enterprise RAG pipelines.",
                "modality": "text",
                "category": "Search Systems"
            },
            {
                "id": "doc_4",
                "title": "Multimodal Image Chart Analysis",
                "content": "NVIDIA Nemotron-Vision provides direct visual QA capabilities on architectural flowcharts, financial plots, and tabular PDF schemas, rendering precise structural insights and confidence metrics.",
                "modality": "multimodal",
                "category": "Vision AI"
            },
            {
                "id": "doc_5",
                "title": "Cross-Encoder Re-Ranking & Shannon Entropy Scoring",
                "content": "Re-ranking raw vector candidates via Cross-Encoder transformer heads computes fine-grained query-document attention logits. Shannon entropy H(P) measures context relevance uncertainty.",
                "modality": "text",
                "category": "Re-Ranking"
            }
        ]
        self.documents = sample_data

    def dense_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Simulate dense vector similarity search."""
        query_words = set(query.lower().split())
        results = []
        for i, doc in enumerate(self.documents):
            doc_words = set(doc["content"].lower().split())
            overlap = len(query_words.intersection(doc_words))
            # Compute mock cosine score
            score = 0.5 + 0.45 * (overlap / (len(query_words) + 1e-5))
            results.append({
                "doc_id": doc["id"],
                "title": doc["title"],
                "content": doc["content"],
                "dense_score": float(np.round(score, 4)),
                "rank": i + 1
            })
        results.sort(key=lambda x: x["dense_score"], reverse=True)
        for rank, res in enumerate(results, 1):
            res["dense_rank"] = rank
        return results[:top_k]

    def bm25_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Simulate sparse BM25 keyword search."""
        query_words = query.lower().split()
        results = []
        for i, doc in enumerate(self.documents):
            score = 0.0
            content_lower = doc["content"].lower()
            for word in query_words:
                if word in content_lower:
                    score += 1.5 + content_lower.count(word) * 0.5
            results.append({
                "doc_id": doc["id"],
                "title": doc["title"],
                "content": doc["content"],
                "bm25_score": float(np.round(score, 4)),
            })
        results.sort(key=lambda x: x["bm25_score"], reverse=True)
        for rank, res in enumerate(results, 1):
            res["bm25_rank"] = rank
        return results[:top_k]
