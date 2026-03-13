---
title: RAG Pipeline Demo
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.33.0
app_file: app.py
pinned: false
license: mit
---

# RAG Pipeline — From Scratch

> Retrieval-Augmented Generation built entirely by hand. No LangChain. No LlamaIndex. Every algorithm implemented from first principles.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/Archit-Konde/RAG)
[![Project Page](https://img.shields.io/badge/🌐-Project%20Page-C9A84C)](https://archit-konde.github.io/RAG/)

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │           INDEXING PHASE            │
                        └─────────────────────────────────────┘

  PDF / TXT
      │
      ▼
 ┌──────────┐    raw text    ┌───────────┐    chunks     ┌─────────────┐
 │ingestion │ ─────────────► │  chunker  │ ─────────────► │  embeddings │
 └──────────┘                └───────────┘                └─────────────┘
   PyPDF2                  recursive split                 all-MiniLM-L6
                           chunk_size=512                  mean pooling
                           overlap=64                      L2 normalize
                                │                               │
                                │         chunks + embeddings   │
                                ▼                               ▼
                           ┌─────────┐                  ┌────────────┐
                           │  BM25   │                  │VectorStore │
                           └─────────┘                  └────────────┘
                           TF-IDF math                  NumPy arrays
                           fit(corpus)                  cosine search


                        ┌─────────────────────────────────────┐
                        │            QUERY PHASE              │
                        └─────────────────────────────────────┘

  User Query
      │
      ├─────────────────────────────────────────────────┐
      │                                                 │
      ▼                                                 ▼
 ┌──────────┐  query vec   ┌────────────┐         ┌─────────┐
 │embeddings│ ────────────►│VectorStore │         │  BM25   │
 └──────────┘              │  .search() │         │get_top_n│
                           └────────────┘         └─────────┘
                                │                      │
                          dense results          sparse results
                                │                      │
                                └──────────┬───────────┘
                                           │
                                           ▼
                                    ┌────────────┐
                                    │  retriever │
                                    │    RRF     │
                                    └────────────┘
                                    Reciprocal Rank
                                    Fusion (k=60)
                                           │
                                    top candidates
                                           │
                                           ▼
                                    ┌────────────┐
                                    │  reranker  │
                                    │cross-encoder│
                                    └────────────┘
                                    ms-marco-MiniLM
                                    joint attention
                                           │
                                    reranked chunks
                                           │
                                           ▼
                                    ┌────────────┐
                                    │ generator  │
                                    │ raw HTTP   │
                                    └────────────┘
                                    /chat/completions
                                    source attribution
                                           │
                                           ▼
                                    Answer + Sources
```

---

## Components

| File | Role | Key Technology |
|------|------|----------------|
| `src/ingestion.py` | Load PDF and text files | PyPDF2 |
| `src/chunker.py` | Split text into overlapping chunks | Recursive separator algorithm |
| `src/embeddings.py` | Batch sentence embeddings | HF Transformers + mean pooling |
| `src/vectorstore.py` | Exact cosine similarity index | NumPy dot product |
| `src/bm25.py` | Sparse lexical retrieval | Okapi BM25 from scratch |
| `src/retriever.py` | Hybrid dense+sparse fusion | Reciprocal Rank Fusion |
| `src/reranker.py` | Cross-encoder re-ranking | MS-MARCO MiniLM |
| `src/generator.py` | Grounded answer generation | Raw `requests` HTTP call |
| `src/evaluation.py` | Pipeline quality metrics | Precision, Recall, MRR |
| `app.py` | Interactive demo UI | Streamlit |

---

## Setup

```bash
git clone https://github.com/Archit-Konde/RAG.git
cd RAG

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your API key

streamlit run app.py
```

---

## HuggingFace Spaces

1. Fork this repo
2. Create a new Space (SDK: Streamlit)
3. Push or link the repo — Spaces reads the YAML frontmatter above
4. Enter your API key in the sidebar (no `.env` needed on Spaces)

---

## Running Tests

```bash
# All tests (note: embeddings/reranker tests download models on first run)
pytest tests/ -v

# Fast tests only (no model downloads)
pytest tests/ -v --ignore=tests/test_embeddings.py --ignore=tests/test_reranker.py

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Benchmarks

Evaluated on a 50-question QA set over a technical PDF (~80 pages).

| Metric | Dense only | Sparse only | Hybrid (RRF) | Hybrid + Rerank |
|--------|-----------|-------------|--------------|-----------------|
| Precision@5 | — | — | — | — |
| Recall@5 | — | — | — | — |
| MRR | — | — | — | — |

*Benchmarks will be populated after evaluation suite is run on a test corpus.*

---

## Key Implementation Notes

**No framework abstractions** — every algorithm is implemented directly:
- `chunker.py`: Recursive separator-based splitting with a deque-window overlap
- `bm25.py`: Okapi BM25 with Robertson-Walker IDF from the formula up
- `vectorstore.py`: Cosine similarity = dot product after L2 normalization
- `retriever.py`: RRF with `score = Σ 1/(k + rank)` across dense + sparse lists
- `embeddings.py`: HF `AutoModel` + manual mean pooling (not sentence-transformers)
- `reranker.py`: Cross-encoder raw logit scoring (not softmax — ranking only needs order)
- `generator.py`: `requests.post` to `/chat/completions` — works with any OpenAI-compatible API

**Learning document** → [`docs/LEARNING.md`](docs/LEARNING.md) — full math derivations for each algorithm, suitable for a blog post.

---

## Links

- [**Project Page**](https://archit-konde.github.io/RAG/) — terminal-styled landing page
- [**Live Demo**](https://huggingface.co/spaces/Archit-Konde/RAG) — HuggingFace Spaces
- [**Learning Document**](docs/LEARNING.md) — full math derivations (blog post source)

---

## License

[MIT](LICENSE)
