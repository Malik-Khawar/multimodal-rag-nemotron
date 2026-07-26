"""
Nemotron Cross-Encoder Re-Ranker Module
Utilizes OpenRouter 'nvidia/llama-nemotron-rerank-vl-1b-v2:free' for passage re-scoring,
with robust score-based fallback logic for rate limits or missing credentials.
"""

import os
import time
import json
import logging
import math
import httpx
from typing import List, Dict, Any, Optional

logger = logging.getLogger("multimodal_rag.reranker")
logging.basicConfig(level=logging.INFO)


class NemotronReranker:
    """
    Cross-encoder re-ranker leveraging OpenRouter's nvidia/llama-nemotron-rerank-vl-1b-v2:free model
    with automated hybrid score-based fallback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "nvidia/llama-nemotron-rerank-vl-1b-v2:free",
        timeout: float = 8.0,
    ):
        """
        Initialize the Nemotron cross-encoder re-ranker.

        Args:
            api_key: OpenRouter API key. If None, checks OPENROUTER_API_KEY env var.
            model: Model identifier on OpenRouter.
            timeout: HTTP request timeout in seconds.
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        self.model = model
        self.timeout = timeout
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Re-ranks a list of candidate passages based on semantic relevance to the query.

        Args:
            query: User search query.
            candidates: List of candidate dicts containing 'id', 'text', 'score' (retrieval score), 'metadata'.
            top_n: Optional count of top passages to return after reranking.

        Returns:
            Dict containing:
                - reranked_chunks: List of reranked candidates with 'rerank_score' and 'rank'
                - fallback_used: Bool indicating if score-based fallback was used
                - fallback_reason: String explanation if fallback was triggered
                - latency_ms: Reranking duration in milliseconds
                - model_used: Model used for re-ranking
        """
        start_time = time.perf_counter()
        if not candidates:
            return {
                "reranked_chunks": [],
                "fallback_used": False,
                "fallback_reason": None,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "model_used": self.model,
            }

        # Attempt primary API re-ranking if API key is present
        if self.api_key and self.api_key != "sk-placeholder":
            try:
                reranked, reason = self._call_nemotron_api(query, candidates)
                if reranked is not None:
                    elapsed = round((time.perf_counter() - start_time) * 1000, 2)
                    if top_n:
                        reranked = reranked[:top_n]
                    return {
                        "reranked_chunks": reranked,
                        "fallback_used": False,
                        "fallback_reason": None,
                        "latency_ms": elapsed,
                        "model_used": self.model,
                    }
                else:
                    fallback_reason = reason or "API response formatting error"
            except Exception as e:
                logger.warning(f"Nemotron API re-ranking failed: {str(e)}. Triggering score fallback.")
                fallback_reason = f"API Error: {type(e).__name__} - {str(e)}"
        else:
            fallback_reason = "No API key configured (OPENROUTER_API_KEY missing)"

        # Fallback to local score-based reranking
        reranked = self._score_based_fallback(query, candidates)
        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        if top_n:
            reranked = reranked[:top_n]

        return {
            "reranked_chunks": reranked,
            "fallback_used": True,
            "fallback_reason": fallback_reason,
            "latency_ms": elapsed,
            "model_used": "local-score-fallback-v1",
        }

    def _call_nemotron_api(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """
        Calls OpenRouter Nemotron reranker API.
        """
        passages_formatted = []
        for idx, item in enumerate(candidates):
            text_snippet = item.get("text", "")[:400].replace("\n", " ")
            passages_formatted.append(f"Passage [{idx+1}]: {text_snippet}")

        passages_text = "\n".join(passages_formatted)
        system_prompt = (
            "You are a cross-encoder document relevance re-ranker. "
            "Evaluate each candidate passage against the user query and assign a relevance score between 0.0 (completely irrelevant) and 1.0 (perfectly relevant). "
            "Output strictly valid JSON with key 'scores' containing a list of float scores corresponding to each passage index in order."
        )
        user_prompt = f"Query: {query}\n\nCandidate Passages:\n{passages_text}\n\nReturn JSON: {{\"scores\": [score1, score2, ...]}}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/khawar-hasnain/multimodal-rag-nemotron",
            "X-Title": "Multimodal RAG Nemotron Pipeline",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self.api_url, headers=headers, json=payload)
            if resp.status_code != 200:
                return None, f"HTTP {resp.status_code}: {resp.text[:150]}"

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)

            scores = parsed.get("scores", [])
            if not isinstance(scores, list) or len(scores) != len(candidates):
                # Try fallback extraction if schema differs slightly
                if isinstance(scores, dict):
                    scores = list(scores.values())
                else:
                    return None, f"Expected {len(candidates)} scores, got {len(scores)}"

            reranked_chunks = []
            for idx, cand in enumerate(candidates):
                raw_score = float(scores[idx]) if idx < len(scores) else 0.5
                norm_score = max(0.0, min(1.0, raw_score))
                updated = dict(cand)
                updated["rerank_score"] = round(norm_score, 4)
                reranked_chunks.append(updated)

            # Sort descending by rerank_score
            reranked_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            for r_idx, chunk in enumerate(reranked_chunks):
                chunk["rank"] = r_idx + 1

            return reranked_chunks, None

    def _score_based_fallback(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Local cross-scoring fallback mechanism combining retrieval dense/sparse score,
        term overlap ratio, and exact sequence matching density.
        """
        query_terms = set(query.lower().split())
        reranked = []

        for idx, cand in enumerate(candidates):
            retrieval_score = cand.get("score", 0.5)
            text_lower = cand.get("text", "").lower()
            text_words = text_lower.split()

            # Calculate term overlap
            if query_terms and text_words:
                matches = sum(1 for term in query_terms if term in text_lower)
                overlap_ratio = matches / len(query_terms)
            else:
                overlap_ratio = 0.0

            # Calculate exact sequence bonus if query substring appears in text
            phrase_bonus = 0.2 if query.lower() in text_lower else 0.0

            # Combined score formula: 60% retrieval score + 30% term overlap + 10% phrase bonus
            combined = (0.6 * retrieval_score) + (0.3 * overlap_ratio) + (0.1 * phrase_bonus)
            final_rerank_score = round(max(0.0, min(1.0, combined)), 4)

            updated = dict(cand)
            updated["rerank_score"] = final_rerank_score
            reranked.append(updated)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        for r_idx, chunk in enumerate(reranked):
            chunk["rank"] = r_idx + 1

        return reranked
