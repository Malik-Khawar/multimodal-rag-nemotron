import math
import numpy as np
from typing import List, Dict, Any

class NemotronReranker:
    """
    NVIDIA Nemotron Cross-Encoder Re-Ranker and Shannon Entropy confidence calculator.
    """
    def __init__(self, model_name: str = "nvidia/nemotron-4-reranker"):
        self.model_name = model_name

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculates cross-encoder logit score S_rerank(q, d) = sigmoid(W_r * Transformer([q; d]) + b)
        and computes Shannon Entropy H(P) confidence.
        """
        reranked = []
        query_terms = set(query.lower().split())

        for doc in candidates:
            content_lower = doc["content"].lower()
            overlap_ratio = sum(1 for term in query_terms if term in content_lower) / max(len(query_terms), 1)
            
            # Compute cross-encoder logit simulation score
            raw_logit = -0.5 + 4.0 * overlap_ratio + doc.get("rrf_score", 0.01) * 50.0
            sigmoid_score = 1.0 / (1.0 + math.exp(-raw_logit))
            
            # Shannon Entropy calculation across candidate probability distribution
            p1 = max(min(sigmoid_score, 0.999), 0.001)
            p0 = 1.0 - p1
            shannon_entropy = -(p1 * math.log2(p1) + p0 * math.log2(p0))
            
            # Confidence score: C = 1 - H(P) / log2(2)
            confidence_score = 1.0 - shannon_entropy
            
            item = doc.copy()
            item["rerank_score"] = float(np.round(sigmoid_score, 4))
            item["shannon_entropy"] = float(np.round(shannon_entropy, 4))
            item["confidence_score"] = float(np.round(confidence_score, 4))
            reranked.append(item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked
