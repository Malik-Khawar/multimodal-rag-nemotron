import os
import sys
import unittest
import shutil
import tempfile
import base64
from pathlib import Path
from io import BytesIO
from PIL import Image
import pypdf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import MultimodalIngestor
from src.embeddings import DualModeEmbedder
from src.vector_store import LanceDBManager
from src.reranker import NemotronReranker
from src.rag_engine import MultimodalRAGEngine
from fastapi.testclient import TestClient
from app import app


class TestMultimodalRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix='rag_test_')
        cls.db_path = os.path.join(cls.temp_dir, 'lancedb_test_store')

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        self.ingestor = MultimodalIngestor(chunk_size=300, chunk_overlap=30)
        self.embedder = DualModeEmbedder()
        self.vector_store = LanceDBManager(db_path=self.db_path)

    def test_01_document_ingestion_formats(self):
        # 1. Plain Text
        text_path = Path(self.temp_dir) / 'spec.txt'
        text_path.write_text('NVIDIA Nemotron-3 is an advanced LLM architecture optimized for enterprise RAG systems.', encoding='utf-8')
        chunks_text = self.ingestor.process_text_file(text_path)
        self.assertGreater(len(chunks_text), 0)
        self.assertEqual(chunks_text[0]['doc_id'], 'spec.txt')
        self.assertEqual(chunks_text[0]['file_type'], 'text')

        # 2. Markdown
        md_path = Path(self.temp_dir) / 'arch.md'
        md_path.write_text('# Architecture\nLanceDB provides zero-copy vector search with IVF-PQ index.', encoding='utf-8')
        chunks_md = self.ingestor.process_markdown_file(md_path)
        self.assertGreater(len(chunks_md), 0)
        self.assertEqual(chunks_md[0]['doc_id'], 'arch.md')

        # 3. PDF
        pdf_path = Path(self.temp_dir) / 'sample.pdf'
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=200, height=200)
        with open(pdf_path, 'wb') as f:
            writer.write(f)
        chunks_pdf = self.ingestor.process_pdf_file(pdf_path)
        self.assertIsInstance(chunks_pdf, list)

        # 4. Image File
        img_path = Path(self.temp_dir) / 'diagram.png'
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(img_path)
        chunks_img = self.ingestor.process_image_file(img_path)
        self.assertEqual(len(chunks_img), 1)
        self.assertEqual(chunks_img[0]['file_type'], 'image')

    def test_02_dual_mode_embedder(self):
        texts = ['Explain RAG pipeline', 'LanceDB vector store']
        embeddings = self.embedder.embed_texts(texts)
        self.assertEqual(len(embeddings), 2)
        # Nemotron 3 embed uses 2048 dims via OpenRouter API, local MiniLM uses 384 dims
        self.assertIn(len(embeddings[0]), [384, 1024, 2048])
        self.assertIn(len(embeddings[1]), [384, 1024, 2048])

    def test_03_vector_store_hybrid_rrf_search(self):
        chunks = [
            {'id': 'c1', 'content': 'NVIDIA Nemotron 3 Ultra offers 550B parameters.', 'source': 'doc1.txt'},
            {'id': 'c2', 'content': 'LanceDB uses Arrow zero-copy memory layout.', 'source': 'doc2.txt'}
        ]
        vectors = [
            [0.1] * 384,
            [0.9] * 384
        ]

        self.vector_store.add_chunks(chunks, vectors, table_name='test_table')
        vec_res = self.vector_store.vector_search([0.1]*384, table_name='test_table', limit=2)
        self.assertGreater(len(vec_res), 0)

        fts_res = self.vector_store.fts_search('Arrow zero-copy', table_name='test_table', limit=2)
        self.assertGreater(len(fts_res), 0)

        hybrid_res = self.vector_store.hybrid_search('Nemotron 550B', [0.1]*384, table_name='test_table', limit=2)
        self.assertGreater(len(hybrid_res), 0)

    def test_04_nemotron_reranker(self):
        reranker = NemotronReranker(api_key='mock_key')
        query = 'What is LanceDB?'
        passages = [
            {'id': 'p1', 'text': 'Unrelated sentence about weather.'},
            {'id': 'p2', 'text': 'LanceDB is a vector database for search.'}
        ]

        rerank_res = reranker.rerank(query, passages, top_n=2)
        self.assertIn('reranked_chunks', rerank_res)

    def test_05_rag_engine_full_pipeline(self):
        engine = MultimodalRAGEngine(db_path=self.db_path)
        engine.ingest_document('LanceDB is high-throughput vector search for Nemotron.', source_name='rag_spec.txt')
        
        result = engine.query('Explain LanceDB and Nemotron', top_k=3)
        self.assertIn('answer', result)
        self.assertIn('trace', result)

    def test_06_fastapi_endpoints(self):
        client = TestClient(app)

        resp_root = client.get('/')
        self.assertEqual(resp_root.status_code, 200)

        resp_stats = client.get('/api/stats')
        self.assertEqual(resp_stats.status_code, 200)

        file_content = b'FastAPI enterprise test document for RAG ingestion.'
        resp_upload = client.post(
            '/api/upload',
            files={'file': ('test_doc.txt', BytesIO(file_content), 'text/plain')}
        )
        self.assertEqual(resp_upload.status_code, 200)

        resp_query = client.post(
            '/api/query',
            json={'query': 'What is in test document?', 'top_k': 3}
        )
        self.assertEqual(resp_query.status_code, 200)

if __name__ == '__main__':
    unittest.main()
