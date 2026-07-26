import os
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.config import (
    BACKUP_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    MULTIMODAL_MODEL,
    OPENROUTER_API_KEY,
    REASONING_MODEL,
    RERANK_MODEL,
)
from src.embeddings import DualModeEmbedder
from src.ingest import MultimodalIngestor
from src.vector_store import LanceDBManager


class TestMultimodalRAGInfrastructure(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_variables(self):
        """Verify environment variables and configuration settings."""
        self.assertIsNotNone(OPENROUTER_API_KEY)
        self.assertEqual(EMBEDDING_MODEL, "nvidia/nemotron-3-embed-1b:free")
        self.assertEqual(RERANK_MODEL, "nvidia/llama-nemotron-rerank-vl-1b-v2:free")
        self.assertEqual(REASONING_MODEL, "nvidia/nemotron-3-ultra-550b-a55b:free")
        self.assertEqual(MULTIMODAL_MODEL, "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")
        self.assertEqual(BACKUP_MODEL, "inclusionai/ling-3.0-flash:free")
        self.assertEqual(CHUNK_SIZE, 500)
        self.assertEqual(CHUNK_OVERLAP, 50)

    def test_ingestor_text_and_markdown(self):
        """Test text and markdown parsing into chunks."""
        ingestor = MultimodalIngestor(chunk_size=10, chunk_overlap=2)
        
        # Test text file
        txt_file = self.temp_path / "sample.txt"
        txt_file.write_text("Sentence one test data. Sentence two test data. Sentence three test data.", encoding="utf-8")
        chunks = ingestor.process_file(txt_file)
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0]["file_type"], "text")
        self.assertIn("content", chunks[0])

        # Test markdown file
        md_file = self.temp_path / "sample.md"
        md_file.write_text("# Title\n\nThis is a markdown content for testing.", encoding="utf-8")
        md_chunks = ingestor.process_file(md_file)
        self.assertGreater(len(md_chunks), 0)
        self.assertEqual(md_chunks[0]["file_type"], "markdown")

    def test_ingestor_image(self):
        """Test image parsing using Pillow."""
        ingestor = MultimodalIngestor()
        img_file = self.temp_path / "diagram.png"
        img = Image.new("RGB", (200, 150), color="blue")
        img.save(img_file)

        chunks = ingestor.process_file(img_file)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["file_type"], "image")
        self.assertIn("200x150", chunks[0]["content"])

    def test_embedder_local_fallback(self):
        """Test embedder local fallback generation."""
        embedder = DualModeEmbedder()
        # Test fallback embedding generation
        sample_texts = ["Hello Nemotron", "LanceDB vector store search"]
        embeddings = embedder.embed_texts(sample_texts)
        self.assertEqual(len(embeddings), 2)
        self.assertIsInstance(embeddings[0], list)
        self.assertGreater(len(embeddings[0]), 0)

    def test_vector_store_hybrid_search(self):
        """Test LanceDB vector store initialization, insertion, and hybrid RRF search."""
        db_dir = self.temp_path / "lancedb_test"
        vstore = LanceDBManager(db_path=str(db_dir))
        
        embedder = DualModeEmbedder()
        chunks = [
            {
                "id": "c1",
                "doc_id": "doc1.txt",
                "source": "doc1.txt",
                "file_type": "text",
                "page_number": None,
                "chunk_index": 0,
                "content": "NVIDIA Nemotron embeddings deliver high quality vector search capabilities.",
                "image_path": None,
                "created_at": "2026-07-26T12:00:00Z",
                "metadata_json": "{}",
            },
            {
                "id": "c2",
                "doc_id": "doc2.txt",
                "source": "doc2.txt",
                "file_type": "text",
                "page_number": None,
                "chunk_index": 0,
                "content": "LanceDB provides fast hybrid search combining vector ANN and BM25 keyword matching.",
                "image_path": None,
                "created_at": "2026-07-26T12:00:00Z",
                "metadata_json": "{}",
            },
        ]
        
        texts = [c["content"] for c in chunks]
        vectors = embedder.embed_texts(texts)
        
        table_name = "test_table"
        vstore.add_chunks(chunks, vectors, table_name=table_name)
        
        query_text = "LanceDB hybrid search"
        query_vec = embedder.embed_query(query_text)
        
        hybrid_results = vstore.hybrid_search(query_text, query_vec, table_name=table_name, limit=2)
        self.assertGreater(len(hybrid_results), 0)
        self.assertIn("_rrf_score", hybrid_results[0])


if __name__ == "__main__":
    unittest.main()
