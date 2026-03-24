# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

From-scratch Retrieval-Augmented Generation pipeline. No LangChain. No LlamaIndex. Every algorithm (BM25, vector search, RRF fusion, cross-encoder reranking, chunking) is implemented from first principles using only NumPy, PyTorch, and HuggingFace Transformers. Streamlit UI deployed on HuggingFace Spaces.

**Live demo:** https://huggingface.co/spaces/architechs/RAG
**Project page:** https://archit-konde.github.io/RAG/

## Build & Development

**Run the app:**
```bash
streamlit run app.py
```

**Run tests:**
```bash
# Fast tests (no model downloads)
pytest tests/ -v --ignore=tests/test_embeddings.py --ignore=tests/test_reranker.py

# All tests (downloads ~175 MB of models on first run)
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

**Run benchmarks:**
```bash
python scripts/run_benchmark.py
```

**Lint (ruff):**
```bash
ruff check src/ tests/ app.py scripts/ learn/
ruff format --check src/ tests/ app.py scripts/ learn/
```
Ruff config is in `ruff.toml` (line-length=120, rules: E/W/F/I/N/UP/B/SIM/TCH/RUF).

## Repository Structure

```
app.py                  ← Streamlit UI (document upload, query, chat history)
ruff.toml               ← Ruff linter/formatter configuration
requirements.txt        ← Python dependencies (torch, transformers, numpy, streamlit)
pytest.ini              ← pytest config (pythonpath = ., testpaths = tests)
src/
  ingestion.py          ← PDF/TXT/MD loading via PyPDF2 (standardized dict output)
  chunker.py            ← Recursive separator-based splitting with deque overlap
  embeddings.py         ← all-MiniLM-L6-v2 with manual mean pooling + L2 norm
  vectorstore.py        ← Exact cosine search on NumPy arrays (.npz/.json persistence)
  bm25.py               ← Okapi BM25 with Robertson-Walker IDF from the formula up
  retriever.py          ← Hybrid dense+sparse via Reciprocal Rank Fusion (k=60)
  reranker.py           ← Cross-encoder ms-marco-MiniLM relevance scoring (raw logits)
  generator.py          ← Raw requests.post to /chat/completions (any OpenAI-compatible API)
  evaluation.py         ← Precision@k, Recall@k, F1, MRR, pluggable faithfulness judge
  utils.py              ← Shared helpers: detect_device(), top_k_indices()
tests/
  test_*.py             ← Unit tests for each src/ module (94 tests total)
learn/
  01-08_*.py            ← Standalone educational scripts (one per algorithm)
scripts/
  run_benchmark.py      ← 25-question eval over HTTP/1.1 protocol corpus
docs/
  LEARNING.md           ← Full math derivations for each algorithm
.github/workflows/
  lint-test.yml         ← CI: ruff check + ruff format + pytest
```

## Key Architecture Decisions

- **No frameworks** — every algorithm implemented directly to demonstrate understanding. BM25 scoring, cosine search, RRF fusion, mean pooling, and cross-encoder reranking are all hand-written.
- **Cosine similarity as dot product** — embeddings are L2-normalized at creation time, so `embeddings @ query` gives cosine similarity without runtime normalization.
- **Reciprocal Rank Fusion** — `score(d) = Σ 1/(k + rank)` across dense + sparse lists. k=60 smooths rank differences so appearing in both lists matters more than rank position. Over-fetches 3× top_k from each retriever before fusion.
- **Cross-encoder reranking** — raw logits (no softmax) because ranking only needs order, not calibrated probabilities. Applied only to the small fused candidate set (5–20 docs).
- **BM25 IDF** — Robertson-Walker formula `log((N - df + 0.5) / (df + 0.5) + 1)` with the outer +1 guaranteeing IDF > 0 even for ubiquitous terms.
- **No ANN index** — vectorstore uses exact search (intentionally transparent for portfolio/learning).
- **Generator** — raw `requests` HTTP to `/chat/completions` endpoint, works with any OpenAI-compatible API (OpenAI, Groq, Together, Ollama). System prompt enforces context-only answering with `[Source N]` citations.
- **Lazy model loading** — `@st.cache_resource` defers ~175 MB model downloads until first use so the Streamlit UI appears instantly.

## Important Conventions

- All ingestion functions return a standardized dict: `{"text": str, "filename": str, "num_pages": int, "metadata": {...}}`.
- Chunk metadata propagates through the entire pipeline for source attribution in final answers.
- The `learn/` scripts are intentionally simple educational code — relaxed lint rules apply (see `ruff.toml` per-file-ignores).
- CI skips `test_embeddings.py` and `test_reranker.py` because they download large models.
- API key is entered via the Streamlit sidebar UI — no `.env` file required on HuggingFace Spaces.

## Tech Stack

- **Python 3.11+**
- **PyTorch + HuggingFace Transformers** — embedding and reranking models
- **NumPy** — vector storage, BM25 scoring, top-k selection
- **PyPDF2** — PDF text extraction
- **Streamlit** — web UI
- **Ruff** — linting and formatting
- **pytest** — testing

## gstack — Virtual Engineering Team

gstack is installed globally at `~/.claude/skills/gstack`.

### Available Skills (by workflow stage)

**Planning & Brainstorming:**
- `/office-hours` — YC-style product framing and brainstorming
- `/plan-ceo-review` — CEO/founder scope and strategy review
- `/plan-eng-review` — Engineering architecture review
- `/plan-design-review` — Design quality plan review
- `/design-consultation` — Full design system creation (DESIGN.md)
- `/autoplan` — Automated review pipeline (CEO + design + eng review)

**Development & Review:**
- `/review` — Staff-engineer pre-landing PR review
- `/investigate` — Structured root-cause debugging
- `/design-review` — Visual QA: find and fix design issues
- `/codex` — OpenAI Codex second opinion

**Testing & Security:**
- `/qa` — QA testing with automated bug fixing
- `/qa-only` — QA report only (no fixes)
- `/cso` — Security audit (OWASP + STRIDE)
- `/benchmark` — Performance regression detection

**Shipping & Deployment:**
- `/ship` — Release engineering (tests, PR, changelog)
- `/land-and-deploy` — Merge PR, deploy, verify production
- `/canary` — Post-deploy monitoring
- `/document-release` — Update docs after shipping

**Browser:**
- `/browse` — Headless browser for testing and dogfooding
- `/setup-browser-cookies` — Import cookies for authenticated testing

**Safety:**
- `/careful` — Warn before destructive commands
- `/freeze` — Lock edits to a specific directory
- `/guard` — Combined safety mode (careful + freeze)
- `/unfreeze` — Remove edit lock

**Utilities:**
- `/retro` — Weekly engineering retrospective
- `/gstack-upgrade` — Update gstack to latest version
- `/setup-deploy` — Configure deployment settings
