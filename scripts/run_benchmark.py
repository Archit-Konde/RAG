"""
Benchmark script for the RAG pipeline.

Usage:
    # Step 1 — inspect chunk boundaries (no models needed)
    python scripts/run_benchmark.py --inspect

    # Step 2 — run full 4-mode benchmark
    python scripts/run_benchmark.py

    # Optional flags
    python scripts/run_benchmark.py --top-k 5 --rerank-pool 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunker import RecursiveTextChunker

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG pipeline benchmark")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print chunks with indices and exit (no models loaded).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of documents to retrieve per query (default: 5).",
    )
    parser.add_argument(
        "--rerank-pool",
        type=int,
        default=10,
        dest="rerank_pool",
        help="Candidates fetched before reranking (default: 10).",
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

CORPUS_PATH = ROOT / "data" / "benchmark_corpus.txt"
TEST_CASES_PATH = ROOT / "data" / "test_cases.json"
RESULTS_PATH = ROOT / "data" / "benchmark_results.json"

MODES = ["dense", "sparse", "hybrid", "hybrid_rerank"]
MODE_LABELS = {
    "dense": "Dense only",
    "sparse": "Sparse only",
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + Rerank",
}


def load_corpus() -> str:
    if not CORPUS_PATH.exists():
        sys.exit(f"[error] corpus not found: {CORPUS_PATH}")
    return CORPUS_PATH.read_text(encoding="utf-8")


def chunk_corpus(corpus: str) -> list[dict]:
    chunker = RecursiveTextChunker(chunk_size=512, chunk_overlap=64)
    return chunker.split_text(corpus)


# ------------------------------------------------------------------
# Inspect mode
# ------------------------------------------------------------------


def run_inspect(chunks: list[dict]) -> None:
    print(f"\n{'=' * 70}")
    print(f"  CORPUS INSPECTION — {len(chunks)} chunks")
    print(f"{'=' * 70}")
    for chunk in chunks:
        idx = chunk["chunk_index"]
        text = chunk["text"]
        preview = text[:120].replace("\n", " ")
        print(f"\n[{idx:02d}] ({len(text)} chars)")
        print(f"     {preview!r}")
    print(f"\n{'=' * 70}")
    print(f"  Total chunks: {len(chunks)}")
    print("  Use these indices as 'relevant_ids' in data/test_cases.json")
    print(f"{'=' * 70}\n")


# ------------------------------------------------------------------
# Full benchmark
# ------------------------------------------------------------------


def build_index(chunks: list[dict]):
    """Embed corpus and build vectorstore + BM25. Returns (retriever, reranker)."""
    from src.bm25 import BM25
    from src.embeddings import EmbeddingModel
    from src.reranker import CrossEncoderReranker
    from src.retriever import HybridRetriever
    from src.vectorstore import VectorStore

    texts = [c["text"] for c in chunks]
    metadata = [{"chunk_index": c["chunk_index"]} for c in chunks]

    print("\n[1/4] Loading embedding model...")
    embedder = EmbeddingModel()

    print("[2/4] Embedding corpus...")
    embeddings = embedder.embed_texts(texts, show_progress=True)

    print("[3/4] Building VectorStore + BM25...")
    vectorstore = VectorStore()
    vectorstore.add(embeddings=embeddings, documents=texts, metadata=metadata)

    bm25 = BM25()
    bm25.fit(texts)

    print("[4/4] Loading cross-encoder reranker...")
    retriever = HybridRetriever(vectorstore, bm25, embedder, rrf_k=60)
    reranker = CrossEncoderReranker()

    print(f"      Index ready: {len(vectorstore)} chunks\n")
    return retriever, reranker


def make_pipeline_fn(mode: str, retriever, reranker, top_k: int, rerank_pool: int):
    """Return a pipeline_fn(query) → {"retrieved_ids": list[int]}."""

    def pipeline_fn(query: str) -> dict:
        if mode == "dense":
            results = retriever.retrieve(query, top_k=top_k, dense_top_k=top_k, sparse_top_k=0)
        elif mode == "sparse":
            results = retriever.retrieve(query, top_k=top_k, dense_top_k=0, sparse_top_k=top_k)
        elif mode == "hybrid":
            results = retriever.retrieve(query, top_k=top_k)
        elif mode == "hybrid_rerank":
            candidates = retriever.retrieve(query, top_k=rerank_pool)
            results = reranker.rerank(query, candidates, top_k=top_k)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return {"retrieved_ids": [r["index"] for r in results]}

    return pipeline_fn


def run_benchmark(args: argparse.Namespace, chunks: list[dict]) -> None:
    from src.evaluation import run_evaluation_suite

    # Load test cases
    if not TEST_CASES_PATH.exists():
        sys.exit(
            f"[error] test cases not found: {TEST_CASES_PATH}\nRun --inspect first, then create data/test_cases.json"
        )
    test_cases = json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(test_cases)} test cases from {TEST_CASES_PATH.name}")

    # Build index once (shared across all modes)
    retriever, reranker = build_index(chunks)

    # Run each mode
    all_results: dict[str, dict] = {}
    for mode in MODES:
        label = MODE_LABELS[mode]
        print(f"Running: {label}...")
        pipeline_fn = make_pipeline_fn(mode, retriever, reranker, args.top_k, args.rerank_pool)
        results = run_evaluation_suite(test_cases, pipeline_fn)
        all_results[mode] = results
        print(
            f"  Precision@{args.top_k}: {results['mean_precision']:.4f}  "
            f"Recall@{args.top_k}: {results['mean_recall']:.4f}  "
            f"MRR: {results['mean_mrr']:.4f}"
        )

    # Print formatted table
    col = 17
    print(f"\n{'=' * 74}")
    print(f"  BENCHMARK RESULTS  (top_k={args.top_k}, n={len(test_cases)} queries)")
    print(f"{'=' * 74}")
    header = f"{'Metric':<16}" + "".join(f"{MODE_LABELS[m]:>{col}}" for m in MODES)
    print(header)
    print("-" * 74)
    for metric_key, label in [
        ("mean_precision", f"Precision@{args.top_k}"),
        ("mean_recall", f"Recall@{args.top_k}"),
        ("mean_mrr", "MRR"),
        ("mean_f1", f"F1@{args.top_k}"),
    ]:
        row = f"{label:<16}"
        for m in MODES:
            row += f"{all_results[m][metric_key]:>{col}.4f}"
        print(row)
    print(f"{'=' * 74}\n")

    # Save results
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Results saved -> {RESULTS_PATH}")
    print("\nCopy Precision@5 / Recall@5 / MRR rows into README.md benchmarks table.")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    corpus = load_corpus()
    chunks = chunk_corpus(corpus)

    if args.inspect:
        run_inspect(chunks)
    else:
        run_benchmark(args, chunks)


if __name__ == "__main__":
    main()
