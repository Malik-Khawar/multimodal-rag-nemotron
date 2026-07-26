"""
FastAPI Server for Multimodal RAG Nemotron Pipeline
Exposes endpoints for interactive dashboard, document indexing, multimodal queries, and vector store statistics.
"""

import os
import sys
import logging
import base64
import io
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure src module is discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.rag_engine import MultimodalRAGEngine

# PDF text extraction
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

logger = logging.getLogger("multimodal_rag.app")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Multimodal RAG Nemotron API",
    description="4-Stage Advanced RAG Pipeline powered by NVIDIA Nemotron Architecture",
    version="1.0.0",
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Engine singleton
rag_engine = MultimodalRAGEngine()

# Templates & Static Setup
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Auto-ingest initial sample document if store is empty
def seed_sample_docs():
    stats = rag_engine.get_stats()
    if stats.get("total_chunks", 0) == 0:
        sample_text = (
            "NVIDIA Nemotron-3 Architecture and Hybrid Retrieval Overview\n\n"
            "The NVIDIA Nemotron family represents state-of-the-art multimodal reasoning models. "
            "Nemotron-3 Ultra (550B parameters) provides superior logical reasoning and contextual synthesis "
            "for complex document queries, while Nemotron-3 Nano Omni (30B parameters) offers high-throughput "
            "visual-textual multimodal intelligence.\n\n"
            "Our 4-Stage Advanced RAG Pipeline incorporates:\n"
            "1. Hybrid Retrieval combining dense vector embeddings (LanceDB) and BM25 sparse keyword search via Reciprocal Rank Fusion (RRF).\n"
            "2. Nemotron Cross-Encoder Re-Ranking using nvidia/llama-nemotron-rerank-vl-1b-v2 to evaluate semantic relevance.\n"
            "3. Multimodal Nemotron Synthesis for generating authoritative, cited responses.\n"
            "4. Automatic Rate-Limit Failover to inclusionai/ling-3.0-flash ensuring 99.9% uptime."
        )
        rag_engine.ingest_document(sample_text, source_name="nemotron_overview_sample.txt")
        logger.info("Seeded initial sample document into LanceDB store.")

seed_sample_docs()


# Data Models
class QueryRequest(BaseModel):
    query: str
    multimodal_image: Optional[str] = None
    top_k: Optional[int] = 5


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serves the main sleek glassmorphic dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/stats")
async def get_stats():
    """Returns vector store and pipeline statistics."""
    try:
        stats = rag_engine.get_stats()
        return JSONResponse(content={"status": "success", "stats": stats})
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    """
    Handles file/image/PDF uploads, parses content, and indexes into LanceDB and BM25 store.
    """
    filename = file.filename or "uploaded_file.txt"
    content_bytes = await file.read()
    extracted_text = ""

    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".pdf":
            if PYPDF_AVAILABLE:
                reader = pypdf.PdfReader(io.BytesIO(content_bytes))
                pages_text = []
                for idx, page in enumerate(reader.pages):
                    txt = page.extract_text() or ""
                    if txt.strip():
                        pages_text.append(txt)
                extracted_text = "\n\n".join(pages_text)
            else:
                # Fallback binary text extractor
                extracted_text = content_bytes.decode("utf-8", errors="ignore")
        elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
            # Image file uploaded as document reference
            b64_img = base64.b64encode(content_bytes).decode("utf-8")
            extracted_text = f"Attached Image Document: {filename} (Base64 length: {len(b64_img)})"
        else:
            # Standard text / md / json file
            extracted_text = content_bytes.decode("utf-8", errors="ignore")

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract readable text from uploaded file.")

        chunks_indexed = rag_engine.ingest_document(extracted_text, source_name=filename)
        stats = rag_engine.get_stats()

        return JSONResponse(
            content={
                "status": "success",
                "message": f"Successfully indexed {chunks_indexed} chunks from '{filename}'.",
                "filename": filename,
                "chunks_indexed": chunks_indexed,
                "stats": stats,
            }
        )
    except Exception as e:
        logger.error(f"Upload indexing error: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.post("/api/query")
async def process_query(payload: QueryRequest):
    """
    Processes RAG chat queries across all 4 stages and returns synthesized response with structured JSON trace.
    """
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        result = rag_engine.query(
            query_str=payload.query.strip(),
            multimodal_image=payload.multimodal_image,
            top_k=payload.top_k or 5,
        )
        return JSONResponse(content={"status": "success", **result})
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}")
        raise HTTPException(status_code=500, detail=f"Query execution error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
