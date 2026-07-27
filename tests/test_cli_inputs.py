# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_engine import MultimodalRAGEngine
from src.ingest import MultimodalIngestor
from test_rag_cli import download_public_samples


class TestCLIInputsVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

        cls.samples = download_public_samples()
        cls.db_path = str(Path(__file__).resolve().parent.parent / 'data' / 'lancedb_cli_verify_store')
        cls.engine = MultimodalRAGEngine(db_path=cls.db_path)
        cls.ingestor = MultimodalIngestor()

        # Ingest all samples into vector store
        for mod, fpath in cls.samples.items():
            chunks = cls.ingestor.process_file(fpath)
            for c in chunks:
                cls.engine.ingest_document(
                    text=c['content'],
                    source_name=fpath.name,
                    metadata={'modality': mod, 'file_type': c.get('file_type')}
                )

    def test_input_01_image_dimensions(self):
        query = "What are the exact pixel dimensions and format of sample_image.png?"
        res = self.engine.query(query, top_k=3)
        ans = res.get('answer', '')
        print(f"\n[Input Test 1] Query: '{query}'")
        print(f"Answer:\n{ans[:250]}...")
        self.assertIn('1025', ans)
        self.assertIn('205', ans)

    def test_input_02_video_specs(self):
        query = "What is the file size and format of sample_video.mp4?"
        res = self.engine.query(query, top_k=3)
        ans = res.get('answer', '')
        print(f"\n[Input Test 2] Query: '{query}'")
        print(f"Answer:\n{ans[:250]}...")
        self.assertTrue('sample_video' in ans.lower() or 'mp4' in ans.lower())

    def test_input_03_audio_format(self):
        query = "What is the file format of sample_audio.mp3?"
        res = self.engine.query(query, top_k=3)
        ans = res.get('answer', '')
        print(f"\n[Input Test 3] Query: '{query}'")
        print(f"Answer:\n{ans[:250]}...")
        self.assertTrue('sample_audio' in ans.lower() or 'mp3' in ans.lower())

    def test_input_04_pdf_name(self):
        query = "What is the name of the PDF document?"
        res = self.engine.query(query, top_k=3)
        ans = res.get('answer', '')
        print(f"\n[Input Test 4] Query: '{query}'")
        print(f"Answer:\n{ans[:250]}...")
        self.assertTrue('sample_pdf' in ans.lower() or 'pdf' in ans.lower())

    def test_input_05_telemetry_trace(self):
        query = "Explain LanceDB vector search"
        res = self.engine.query(query, top_k=3)
        trace = res.get('trace', {})
        print(f"\n[Input Test 5] Telemetry Trace Check: Latency={trace.get('total_latency_ms')} ms")
        self.assertIn('total_latency_ms', trace)
        self.assertIn('stage1_hybrid_retrieval', trace)


if __name__ == '__main__':
    unittest.main()
