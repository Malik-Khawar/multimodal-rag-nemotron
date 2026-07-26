"""
Command-line benchmark & demonstration script for the 4-Stage Multimodal RAG Nemotron Pipeline.
Runs sample document ingestion, hybrid retrieval, cross-encoder re-ranking, and response synthesis.
"""

import os
import sys
import json
import time

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure src package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.rag_engine import MultimodalRAGEngine


def run_benchmark():
    print("=" * 75)
    print(" [RAG PIPELINE] 4-STAGE MULTIMODAL RAG PIPELINE - BENCHMARK DEMO")
    print(" Powered by NVIDIA Nemotron Models & LanceDB Hybrid Search")
    print("=" * 75)

    # 1. Initialize RAG Engine
    print("\n[INFO] Initializing RAG Engine & Vector Store...")
    engine = MultimodalRAGEngine(db_path="./data/lancedb_store")
    stats = engine.get_stats()
    print(f"       Embeddings Dimension : {stats['vector_dim']}")
    print(f"       LanceDB Store Active : {stats['lancedb_active']}")
    print(f"       BM25 Index Active    : {stats['bm25_active']}")
    print(f"       Re-Ranker Model      : {stats['primary_models']['reranker']}")
    print(f"       Multimodal Model     : {stats['primary_models']['multimodal']}")
    print(f"       Text Reasoning Model : {stats['primary_models']['text_reasoning']}")

    # 2. Ingest Sample Benchmark Documents
    sample_docs = [
        (
            "NVIDIA Nemotron-3 Ultra (550B parameters) is engineered for deep reasoning, mathematical problem solving, "
            "and enterprise document analysis. It utilizes an hybrid attention transformer architecture with 128k context window support. "
            "In benchmark evaluations, Nemotron-3 Ultra achieves SOTA performance on RAG context synthesis.",
            "nemotron_ultra_specs.txt"
        ),
        (
            "NVIDIA Nemotron-3 Nano Omni (30B parameters) is a vision-language multimodal model optimized for real-time video, "
            "image, and spatial layout comprehension. It processes high-resolution visual inputs alongside textual context with minimal latency.",
            "nemotron_omni_specs.txt"
        ),
        (
            "Stage 1 Hybrid Retrieval couples dense semantic embeddings generated via LanceDB with sparse lexical inverted index (BM25Okapi). "
            "Scores are combined using Reciprocal Rank Fusion (RRF k=60) and linear score blending: Score_hybrid = 0.6 * Dense + 0.4 * Sparse. "
            "This eliminates vocabulary mismatch errors while maintaining semantic precision.",
            "rag_hybrid_retrieval.txt"
        ),
        (
            "Stage 2 Cross-Encoder Re-Ranking passes top-K candidate passages to nvidia/llama-nemotron-rerank-vl-1b-v2. "
            "The model computes deep query-passage joint attention, outputting normalized relevance scores from 0.0 to 1.0. "
            "If rate limits occur, the engine triggers a score-based cross-term overlap fallback.",
            "rag_reranker_specs.txt"
        ),
        (
            "Stage 4 Automatic Failover guarantees system resilience. If Stage 3 OpenRouter API requests hit HTTP 429 rate limits or network timeouts, "
            "the orchestrator instantly routes requests to inclusionai/ling-3.0-flash:free or local synthesis fallback.",
            "rag_failover_specs.txt"
        )
    ]

    print("\n[1/4] Ingesting Benchmark Sample Documents...")
    total_chunks = 0
    for doc_text, source in sample_docs:
        c_count = engine.ingest_document(doc_text, source_name=source)
        total_chunks += c_count
        print(f"      Indexed {c_count} chunk(s) from '{source}'")
    
    print(f"      --> Total Chunks in LanceDB: {total_chunks}")

    # 3. Execute Sample Benchmark Queries
    benchmark_queries = [
        {
            "title": "Query 1: Text Reasoning & Hybrid Retrieval Analysis",
            "query": "How does Stage 1 Hybrid Retrieval combine LanceDB vector search and BM25 search?",
            "image": None,
        },
        {
            "title": "Query 2: Multimodal Architecture Comparison",
            "query": "What are the primary differences between Nemotron-3 Ultra 550B and Nemotron-3 Nano Omni 30B?",
            "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",  # 1x1 dot base64
        }
    ]

    for idx, q_info in enumerate(benchmark_queries):
        print(f"\n" + "-" * 75)
        print(f" [*] BENCHMARK RUN {idx+1}: {q_info['title']}")
        print(f" Query: \"{q_info['query']}\"")
        if q_info['image']:
            print(" Multimodal Image: Attached (Base64)")
        print("-" * 75)

        res = engine.query(q_info['query'], multimodal_image=q_info['image'], top_k=3)
        trace = res["trace"]

        # Print Stage 1 Trace
        s1 = trace["stage1_hybrid_retrieval"]
        print(f"\n   Stage 1: Hybrid Retrieval ({s1['latency_ms']} ms)")
        print(f"   Retrieved {s1['candidate_count']} candidate passages:")
        for cand in s1["candidates"][:3]:
            print(f"   - [{cand['id']}] Dense: {cand['dense_score']} | Sparse: {cand['sparse_score']} | Hybrid Score: {cand['score']} | Src: {cand['source']}")

        # Print Stage 2 Trace
        s2 = trace["stage2_nemotron_reranking"]
        print(f"\n   Stage 2: Nemotron Re-Ranking ({s2['latency_ms']} ms)")
        print(f"   Model: {s2['model_used']} (Fallback Used: {s2['fallback_used']})")
        if s2['fallback_used']:
            print(f"   Fallback Reason: {s2['fallback_reason']}")
        for r_chunk in s2["top_chunks"][:3]:
            print(f"   - Rank #{r_chunk['rank']} (Score: {r_chunk['rerank_score']}): {r_chunk['text'][:90]}...")

        # Print Stage 3/4 Trace
        s3 = trace["stage3_and_4_synthesis"]
        print(f"\n   Stage 3 & 4: Multimodal Synthesis & Failover ({s3['latency_ms']} ms)")
        print(f"   Requested Model : {s3['model_requested']}")
        print(f"   Executed Model  : {s3['model_used']}")
        print(f"   Failover Active : {s3['failover_triggered']}")
        if s3['failover_triggered']:
            print(f"   Failover Reason : {s3['failover_reason']}")

        # Print Answer Summary
        print("\n [ANSWER] Synthesized Response:")
        print(f" {res['answer']}")
        print(f"\n Total Execution Time: {trace['total_latency_ms']} ms")

    print("\n" + "=" * 75)
    print(" [SUCCESS] BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_benchmark()
