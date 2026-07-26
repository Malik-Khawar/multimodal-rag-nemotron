import logging
from typing import List, Optional, Union
import requests

from src.config import EMBEDDING_MODEL, OPENROUTER_API_KEY

logger = logging.getLogger(__name__)

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
FALLBACK_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class DualModeEmbedder:
    """
    Dual-mode embedding module:
    Primary: OpenRouter API calling `nvidia/nemotron-3-embed-1b:free`.
    Fallback: Local CPU `sentence-transformers/all-MiniLM-L6-v2` for zero-downtime offline execution.
    """

    def __init__(self, api_key: Optional[str] = None, primary_model: Optional[str] = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.primary_model = primary_model or EMBEDDING_MODEL
        self._local_model = None

    def _get_local_model(self):
        """Lazy load the sentence-transformers local model on demand."""
        if self._local_model is None:
            logger.info(f"Initializing local fallback model: {FALLBACK_MODEL_NAME}")
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer(FALLBACK_MODEL_NAME)
        return self._local_model

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local sentence-transformers model."""
        try:
            model = self._get_local_model()
            embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Local embedding fallback failed: {e}")
            raise RuntimeError(f"Both primary API and local fallback embedding failed: {e}")

    def _embed_openrouter(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Attempt to fetch embeddings from OpenRouter API."""
        if not self.api_key or self.api_key.startswith("your_"):
            logger.warning("OpenRouter API key missing or default, using local fallback.")
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/multimodal-rag-nemotron",
            "X-Title": "Multimodal RAG Nemotron",
        }
        
        payload = {
            "model": self.primary_model,
            "input": texts,
        }

        try:
            response = requests.post(
                OPENROUTER_EMBEDDINGS_URL,
                headers=headers,
                json=payload,
                timeout=15,
            )
            
            if response.status_code != 200:
                logger.warning(
                    f"OpenRouter API embedding error (status {response.status_code}): {response.text}"
                )
                return None

            data = response.json()
            if "data" in data and isinstance(data["data"], list):
                # Sort by index to maintain original text ordering
                items = sorted(data["data"], key=lambda x: x.get("index", 0))
                embeddings = [item["embedding"] for item in items if "embedding" in item]
                if len(embeddings) == len(texts):
                    logger.info(f"Successfully generated {len(embeddings)} embeddings via OpenRouter API.")
                    return embeddings
                    
            logger.warning(f"Unexpected response structure from OpenRouter API: {data}")
            return None

        except Exception as e:
            logger.warning(f"OpenRouter API request failed: {e}. Falling back to local model.")
            return None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of text strings.
        Tries OpenRouter API first; falls back to local sentence-transformers on error.
        """
        if not texts:
            return []

        # Try Primary OpenRouter API
        embeddings = self._embed_openrouter(texts)
        if embeddings is not None:
            return embeddings

        # Fallback to local CPU model
        logger.info("Falling back to local CPU sentence-transformers embedding...")
        return self._embed_local(texts)

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        res = self.embed_texts([query])
        return res[0] if res else []
