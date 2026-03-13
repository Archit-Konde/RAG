# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-13

### Added

- **Recursive text chunker** (`src/chunker.py`) — configurable separators, chunk size, overlap
- **Sentence embeddings** (`src/embeddings.py`) — all-MiniLM-L6-v2 via HF Transformers, mean pooling
- **NumPy vector store** (`src/vectorstore.py`) — cosine similarity search, save/load persistence
- **BM25 sparse retrieval** (`src/bm25.py`) — Okapi BM25 from scratch, Robertson-Walker IDF
- **Hybrid retriever** (`src/retriever.py`) — dense + sparse fusion via Reciprocal Rank Fusion (k=60)
- **Cross-encoder reranker** (`src/reranker.py`) — ms-marco-MiniLM-L-6-v2, raw logit scoring
- **LLM generator** (`src/generator.py`) — raw HTTP to OpenAI-compatible API, source attribution
- **Evaluation suite** (`src/evaluation.py`) — Precision@k, Recall@k, MRR, faithfulness
- **PDF/text ingestion** (`src/ingestion.py`) — PyPDF2 for PDFs, UTF-8 for text files
- **Streamlit app** (`app.py`) — upload, index, query, view chunks + answer + sources
- **Unit test suite** — 87 test cases across 9 test files
- **Learning document** (`docs/LEARNING.md`) — full math derivations for all algorithms
- **Terminal-styled project page** (`docs/index.html`) — interactive landing page
- **HuggingFace Spaces deployment** — Streamlit SDK, dark terminal theme
- **GitHub Actions CI** — linting and testing on push/PR
- **GitHub Pages deployment** — automated via Actions workflow

[1.0.0]: https://github.com/Archit-Konde/RAG/releases/tag/v1.0.0
