import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env if present
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# OpenRouter API Key & Model Configuration
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free")
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "nvidia/llama-nemotron-rerank-vl-1b-v2:free")
REASONING_MODEL: str = os.getenv("REASONING_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
MULTIMODAL_MODEL: str = os.getenv("MULTIMODAL_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")
BACKUP_MODEL: str = os.getenv("BACKUP_MODEL", "inclusionai/ling-3.0-flash:free")

# LanceDB Configuration
LANCEDB_PATH: str = os.getenv("LANCEDB_PATH", str(BASE_DIR / "data" / "lancedb_store"))
DEFAULT_TABLE_NAME: str = "multimodal_documents"

# Chunking Configuration
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))      # Tokens / words per chunk
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))   # Token / word overlap between chunks
