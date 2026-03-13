# Building RAG From Scratch — A Complete Technical Deep-Dive

> This document explains every algorithm in this RAG pipeline from first principles. It is written to become a blog post. Each section contains the math, the intuition, and the implementation decision behind each component.

---

## Table of Contents

1. [What is RAG and Why?](#1-what-is-rag-and-why)
2. [Recursive Text Chunking](#2-recursive-text-chunking)
3. [Sentence Embeddings — Mean Pooling](#3-sentence-embeddings--mean-pooling)
4. [Vector Store — Cosine Similarity as Dot Product](#4-vector-store--cosine-similarity-as-dot-product)
5. [BM25 — Sparse Retrieval from First Principles](#5-bm25--sparse-retrieval-from-first-principles)
6. [Hybrid Retrieval — Reciprocal Rank Fusion](#6-hybrid-retrieval--reciprocal-rank-fusion)
7. [Cross-Encoder Reranking — Joint Attention](#7-cross-encoder-reranking--joint-attention)
8. [Generation — Prompt Engineering and Source Attribution](#8-generation--prompt-engineering-and-source-attribution)
9. [Evaluation — Measuring What Matters](#9-evaluation--measuring-what-matters)
10. [Implementation Decisions](#10-implementation-decisions)

---

## 1. What is RAG and Why?

Large language models are trained on static snapshots of the internet. They cannot know about documents you wrote last week, the internal policy PDF sitting on your server, or a research paper published after their training cutoff. They also hallucinate — they fill knowledge gaps with plausible-sounding but fabricated text.

**Retrieval-Augmented Generation** (RAG) solves both problems. Instead of asking the model to recall facts from its weights, you:

1. Store your documents in a searchable index.
2. When a question arrives, retrieve the most relevant passages.
3. Stuff those passages into the prompt as context.
4. Ask the model to answer *only from that context*.

The model stops needing to memorize facts. It becomes a *reading comprehension engine* operating over a dynamic knowledge base that you control.

### The pipeline at a glance

```
Documents → Chunk → Embed → Index
                                ↕
Query → Embed → Retrieve → Rerank → Prompt → LLM → Answer
```

The left side (indexing) happens once. The right side (querying) happens on every user request.

---

## 2. Recursive Text Chunking

### Why chunk at all?

Embedding models have a fixed maximum input length. `all-MiniLM-L6-v2` handles at most 256 tokens (≈512 characters of average English prose). You cannot embed a 50-page PDF as a single vector — it won't fit, and even if it did, the resulting vector would be a diffuse average of everything, matching nothing well.

Chunking splits documents into passages that:
- Fit within the embedding model's window
- Are semantically coherent (ideally self-contained)
- Overlap slightly so information at boundaries isn't lost

### Why "recursive" chunking?

A naive approach splits on `\n\n`. But what if a paragraph is still 2,000 characters? You split on `\n`. Still too long? Split on `. `. Still? On ` `. Still? Hard-slice by character.

This hierarchy of separators is the "recursive" part. You try the most semantically meaningful separator first (paragraph break), then fall back to progressively less meaningful ones, never losing content.

### The algorithm

```
def _recursive_split(text, separators):
    separator = separators[0]
    remaining = separators[1:]

    pieces = text.split(separator)          # split on current separator
    good = []
    result = []

    for piece in pieces:
        if len(piece) <= chunk_size:
            good.append(piece)              # fits — accumulate
        else:
            result += merge(good)           # flush accumulated pieces
            good = []
            result += recursive_split(piece, remaining)  # recurse on oversized piece

    result += merge(good)                   # flush remainder
    return result
```

### The overlap window

After splitting, short pieces are *merged* back into full-sized chunks. The merge step maintains a sliding deque of pieces. When the window is full:

1. Emit the current window as one chunk.
2. Pop pieces from the *left* of the deque until only `chunk_overlap` characters remain.
3. This retained tail becomes the *start* of the next chunk.

```
pieces:    ["The cat sat ", "on the mat. ", "The dog slept ", "on the floor."]
chunk_size: 30, overlap: 10

Chunk 1: "The cat sat on the mat. "     (24 chars)
         → flush, retain last 10 chars: "he mat. "
Chunk 2: "he mat. The dog slept "        (starts with overlap)
         ...
```

The overlap prevents a question about a concept that straddles two chunks from missing context entirely.

### start_char / end_char reconstruction

After merging, we need to know where each chunk lives in the original text (for source attribution in answers). We reconstruct positions using a **forward cursor**:

```python
cursor = 0
for chunk in chunks:
    pos = original_text.find(chunk.text, cursor)
    chunk.start_char = pos
    chunk.end_char = pos + len(chunk.text)
    cursor = max(cursor, pos + len(chunk.text) - chunk_overlap)
```

We advance `cursor` to just past the non-overlapping tail of each chunk, so `find()` always starts from roughly the right place and doesn't backtrack unnecessarily.

---

## 3. Sentence Embeddings — Mean Pooling

### What is a sentence embedding?

A sentence embedding maps a variable-length string to a fixed-length vector in ℝᵈ such that **semantically similar sentences are geometrically close**. This is the foundation of semantic search.

`all-MiniLM-L6-v2` is a 6-layer transformer distilled from `MiniLM-L12-v2`, itself distilled from BERT. It outputs 384-dimensional vectors.

### Why not use the [CLS] token?

BERT-family models prepend a special `[CLS]` token whose final hidden state is used for *classification* tasks (e.g. sentiment). For sentence similarity, the `[CLS]` token only aggregates information through its attention to other tokens — it doesn't average them.

Empirically, **mean pooling** (averaging all token embeddings weighted by whether they're real tokens) produces significantly better sentence-level representations for retrieval tasks. This is what sentence-transformers were fine-tuned to use.

### Mean pooling — the math

The model outputs `last_hidden_state` of shape `(B, T, D)`:
- B = batch size
- T = sequence length (including padding)
- D = 384 (hidden dimension)

Not all T tokens are real — padding tokens fill shorter sequences to match the longest in the batch. The `attention_mask` is a `(B, T)` binary tensor where 1 = real token, 0 = padding.

```
mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings)
                # shape: (B, T, D)

sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                # shape: (B, D)  — sum of real token embeddings only

sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                # shape: (B, D)  — count of real tokens per position (all same per row)

pooled = sum_embeddings / sum_mask
                # shape: (B, D)  — mean of real token embeddings
```

The `.unsqueeze(-1)` adds a trailing dimension so the mask broadcasts across the D dimension. This is the most common implementation bug — forgetting `.expand_as()` and getting a shape mismatch or incorrect broadcast.

### L2 normalization

After pooling, we normalize each vector to unit length:

```python
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms = np.clip(norms, 1e-12, None)   # prevent division by zero
normalized = embeddings / norms
```

Why? Because this converts cosine similarity into a simple dot product (see next section). The clip prevents numerical issues for pathological all-zero vectors.

---

## 4. Vector Store — Cosine Similarity as Dot Product

### What is cosine similarity?

For two vectors **a** and **b**:

```
cosine_similarity(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

It measures the angle between them, not their magnitude. Two vectors pointing in the same direction have cosine similarity 1.0; perpendicular vectors have 0.0; opposite vectors have -1.0.

Cosine similarity is the right metric for embeddings because the direction encodes meaning, not the magnitude.

### The optimization: pre-normalize

If all stored vectors are L2-normalized (‖v‖ = 1), then:

```
cosine_similarity(q, v) = (q · v) / (‖q‖ · ‖v‖)
                        = (q · v) / (1 · 1)
                        = q · v
```

**Cosine similarity becomes a plain dot product.** This matters because a dot product against an entire matrix is a single highly-optimized BLAS call:

```python
scores = self._embeddings @ query_embedding   # (N, D) @ (D,) → (N,)
```

This is O(N × D) and runs in microseconds on NumPy for tens of thousands of chunks.

### Top-k without full sort

To find the top 5 results from 50,000 scores, we don't need to sort everything:

```python
# O(N) partial sort — only guarantees top-k, not their order
partition = np.argpartition(scores, -top_k)[-top_k:]
# O(k log k) — sort only the k candidates
top_indices = partition[np.argsort(scores[partition])[::-1]]
```

`np.argpartition` is a quickselect variant — it finds the k largest values in O(N) without sorting everything. We then sort just those k values to put them in order.

### Save format

```
{path}.npz   — compressed NumPy archive (key: "embeddings")
{path}.json  — JSON array: [{"document": str, "metadata": dict}, ...]
```

The two-file format keeps binary data (the float32 matrix) separate from human-readable metadata. `np.savez_compressed` uses zlib internally, reducing a 1M×384 float32 matrix (~1.4GB) to roughly 200–400MB depending on data.

**Gotcha:** `np.savez` (and `np.savez_compressed`) automatically appends `.npz` if not already present. So `np.load("store")` will fail; you must use `np.load("store.npz")`. Handle this consistently in both `save()` and `load()`.

---

## 5. BM25 — Sparse Retrieval from First Principles

### The problem with TF-IDF

Classic TF-IDF scores a document for a query term as:

```
TF-IDF(t, d) = tf(t, d) × log(N / df(t))
```

Where:
- `tf(t, d)` = count of term t in document d
- `N` = total number of documents
- `df(t)` = number of documents containing term t

Problems:
1. **TF unbounded**: A document mentioning "cat" 100 times scores 10× higher than one mentioning it 10 times. But both clearly discuss cats — the additional mentions don't add 10× more relevance.
2. **Length bias**: Longer documents naturally have higher TF even for the same density of the query term.

### BM25 fixes both

**Okapi BM25** (Best Match 25) addresses both with two innovations:

**1. TF saturation (k1 parameter):**

```
TF_BM25(t, d) = tf(t, d) × (k1 + 1)
                ─────────────────────
                tf(t, d) + k1
```

As `tf → ∞`, this approaches `(k1 + 1)`. The score *saturates* — mentioning a term 100 times is only slightly better than 10 times. Standard k1 = 1.5 means a document mentioning the query term once scores at ~0.8 of the maximum, and mentioning it twice scores ~0.9.

**2. Document length normalization (b parameter):**

```
TF_BM25(t, d) = tf(t, d) × (k1 + 1)
                ──────────────────────────────────────────
                tf(t, d) + k1 × (1 - b + b × |d| / avgdl)
```

Where:
- `|d|` = length of document d in tokens
- `avgdl` = average document length across the corpus
- `b` = normalization strength (0 = no normalization, 1 = full)

Standard b = 0.75: a document twice the average length is penalized but not eliminated.

### IDF — Robertson-Walker formula

```
IDF(t) = log( (N - df(t) + 0.5) / (df(t) + 0.5)  +  1 )
```

The outer `+1` is critical. Without it:
- If df(t) = N (term in every document), the inner fraction is `0.5/N ≈ 0`, and `log(0) = -∞`.
- With it: `log(0 + 1) = 0` — universal terms contribute nothing (not negative).

The +0.5 smoothing prevents the IDF from being infinity for rare one-document terms.

### Full BM25 formula

```
score(d, q) = Σ_{t ∈ q}  IDF(t) × tf(t,d) × (k1 + 1)
                          ────────────────────────────────────────────
                          tf(t,d) + k1 × (1 - b + b × |d| / avgdl)
```

### Why whitespace tokenization?

This implementation uses `text.lower().translate(...punctuation...).split()` — no NLTK, no spaCy, no stopword lists. This is intentional:

- **Transparency**: You can trace exactly what tokens are produced.
- **Portability**: No additional downloads or language models.
- **Sufficient for English**: BM25's IDF naturally down-weights common words ("the", "a") since they appear in nearly every document, giving them near-zero IDF.

---

## 6. Hybrid Retrieval — Reciprocal Rank Fusion

### Why hybrid?

Dense retrieval (vector search) excels at semantic matching:
- Query: "furry household companion" → correctly retrieves "The cat sat on the mat"
- Dense embedding captures the *meaning*, not the literal words

Sparse retrieval (BM25) excels at exact keyword matching:
- Query: "PyTorch 2.1 changelog" → must find docs containing exactly "PyTorch 2.1"
- Dense embeddings of version numbers are essentially random

Neither dominates. A hybrid approach gets the best of both.

### Why not just average the scores?

You could compute `score = α × dense_score + (1-α) × bm25_score`. But:
- Dense scores are cosine similarities in [-1, 1], typically 0.2–0.9 for good matches
- BM25 scores are unbounded positive floats depending on corpus size and TF

These distributions are completely different. Combining them directly requires careful calibration (finding the right `α`) that varies by corpus.

### Reciprocal Rank Fusion (RRF)

RRF operates on **rankings** (positions), not scores. It's immune to score calibration problems because ranks are always integers starting at 1.

```
RRF_score(d) = Σ_{list L}   1 / (k + rank_L(d))
```

Where:
- `rank_L(d)` = 1-based position of document d in list L
- `k` = smoothing constant (standard default: 60)
- Documents not in a list contribute 0 for that list

### Why k = 60?

The k constant prevents top-ranked documents from dominating too strongly. Without it (k=0), rank 1 in one list gives a score of 1.0, while rank 2 gives 0.5 — a 2× difference for what might be a trivially small quality gap. With k=60:

```
rank 1:  1/(60+1) = 0.01639
rank 2:  1/(60+2) = 0.01613
rank 10: 1/(60+10) = 0.01429
```

The difference between rank 1 and rank 10 is only ~15%, not 1000%. This makes RRF robust: a document that's rank 1 in dense but rank 15 in sparse will still win over a document that's rank 5 in dense but not in sparse at all.

The value k=60 was empirically determined by Cormack et al. (2009) to perform well across a wide range of retrieval tasks without tuning.

### Over-fetching

Before fusing, we retrieve 3× top_k candidates from each retriever (minimum 20). If we only retrieved top_k=5 from each, RRF would have at most 10 unique documents to merge into 5 results — not enough to benefit from the fusion. Over-fetching gives RRF enough candidates to find the true best results.

---

## 7. Cross-Encoder Reranking — Joint Attention

### Bi-encoder vs cross-encoder

**Bi-encoder** (what `embeddings.py` does):
```
Encode query → q vector
Encode document → d vector
Score = cosine_sim(q, d)
```

Query and document are encoded *independently*. Similarity is computed in the final vector space. This enables precomputing document embeddings offline.

**Cross-encoder**:
```
Input: [CLS] query [SEP] document [SEP]
Output: single relevance score
```

The query and document are concatenated and run through the transformer *together*. Every query token can attend to every document token and vice versa. This captures interactions like "the word 'not' in the query reverting the meaning of a positive word in the document" that a bi-encoder misses entirely.

### The tradeoff

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Accuracy | Moderate | High |
| Precompute docs | Yes | No |
| Query latency | O(1) once indexed | O(candidates) |
| Use case | First-stage retrieval | Re-ranking |

You cannot use a cross-encoder for first-stage retrieval because you'd need to run it against all N documents at query time — too slow for large corpora. But for re-ranking 5–20 candidates (already filtered by the hybrid retriever), it's fast enough and significantly more accurate.

### Raw logits, not softmax

The MS-MARCO model outputs a single logit (the unnormalized log-probability of relevance). We use this directly for sorting:

```python
logits = model(**encoding).logits.squeeze(-1)   # shape (B,)
```

No softmax needed. Softmax would normalize across the batch, making the score of one document dependent on which other documents happen to be in the same batch — wrong for a ranking problem where we want absolute scores.

---

## 8. Generation — Prompt Engineering and Source Attribution

### The RAG contract

The system prompt establishes a binding contract:

```
You are a precise question-answering assistant. Answer questions using ONLY
the provided context. If the context does not contain enough information
to answer, say "I don't have enough information in the provided documents
to answer this question." Do not speculate or use knowledge outside the
provided context. Cite sources using [Source N] notation.
```

This prompt does three things:
1. **Constrains** the model to the retrieved context (reduces hallucination)
2. **Allows graceful refusal** when context is insufficient (better than a wrong answer)
3. **Requires citations** (enables source attribution in the UI)

### Context injection format

```
[Source 1] (my_paper.pdf, chunk 3):
BM25 is an information retrieval function that ranks documents...

[Source 2] (technical_guide.txt, chunk 12):
Reciprocal Rank Fusion combines multiple ranked lists...

Question: What is the best way to combine dense and sparse retrieval?
```

The numbered `[Source N]` headers serve two purposes:
1. They give the model a citation shorthand ("According to [Source 1]...")
2. They enable the UI to map citation numbers back to specific chunks

### Source attribution metadata

Each chunk carries its origin through the entire pipeline:

```
ingestion: metadata["source"] = "/abs/path/to/document.pdf"
chunker:   adds chunk_index to each chunk dict
vectorstore: stores metadata alongside embeddings
generator: _extract_sources() reads metadata["source"] and chunk_index
```

The user sees: "This answer is based on `document.pdf` (chunk 3), (chunk 7)".

### Raw HTTP call

No SDK wrapper. The `requests.post` call directly mirrors the API specification:

```python
requests.post(
    f"{base_url}/chat/completions",
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    json={"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 1024},
    timeout=30,
)
```

This works identically with OpenAI, Together AI, Groq, Ollama (local), or any OpenAI-compatible endpoint. The URL and key change; the code does not.

---

## 9. Evaluation — Measuring What Matters

### Two separate problems

RAG quality has two distinct components with different failure modes:

**Retrieval quality**: Are the right chunks being found?
- Measured without an LLM — just compare retrieved IDs to ground-truth relevant IDs
- Metric failure: high precision but low recall means the system is precise but misses relevant content

**Generation quality**: Does the answer faithfully reflect the retrieved context?
- Cannot be measured without some form of language understanding
- Metric failure: the model ignores context and answers from its parametric knowledge (hallucination)

### Precision@k

```
precision@k = |retrieved@k ∩ relevant| / k
```

Of the k documents we retrieved, what fraction were actually relevant? A precision of 1.0 means every retrieved chunk was relevant. Low precision wastes LLM context with irrelevant text.

### Recall@k

```
recall@k = |retrieved@k ∩ relevant| / |relevant|
```

Of all the relevant documents, what fraction did we retrieve? Low recall means the LLM can't answer because relevant chunks weren't retrieved — a ceiling on generation quality.

### MRR — Mean Reciprocal Rank

```
MRR = (1/|Q|) × Σ_{q ∈ Q}  1 / rank_q(first relevant doc)
```

MRR measures how early the *first* relevant document appears. If the first relevant doc is at rank 1, MRR = 1.0. Rank 3, MRR = 0.33. MRR = 0 if no relevant doc was found.

This matters for RAG because if the most relevant chunk is at rank 5 but we only pass rank 1 to the LLM, the question goes unanswered.

### Faithfulness — LLM-as-Judge

Faithfulness cannot be computed with a simple formula — it requires semantic understanding. The standard approach:

```python
def llm_judge(answer, context_chunks):
    prompt = f"""
    Context: {' '.join(context_chunks)}
    Answer: {answer}

    Rate how faithfully the answer is supported by the context on a scale from 0.0 to 1.0.
    Return only a number.
    """
    return float(llm.generate(prompt))
```

This is called "LLM-as-judge." Its limitations:
- Expensive (requires an LLM call per evaluation)
- The judge LLM may disagree with human assessors ~10–20% of the time
- Bias toward longer, confident-sounding answers

For a portfolio project, a simpler heuristic is measuring what fraction of sentences in the answer contain at least one n-gram also present in the retrieved context. Not perfect, but zero-cost.

---

## 10. Implementation Decisions

### PyPDF2 over pdfminer.six

PyPDF2 is a pure-Python library with zero compiled dependencies. pdfminer.six is more accurate (especially for multi-column layouts and tables) but requires more dependencies and is slower. For a portfolio project demonstrating the RAG pipeline, text extraction quality is secondary to keeping the environment simple.

### Exact cosine search over FAISS

[FAISS](https://github.com/facebookresearch/faiss) provides approximate nearest neighbor (ANN) search that scales to billions of vectors. But it requires a compiled C++ library, is harder to inspect, and ANN introduces errors.

For a corpus of up to ~100,000 chunks (a typical portfolio-scale deployment), exact cosine search with NumPy is:
- Fast enough (< 10ms for 100K × 384 on CPU)
- Perfectly accurate (no ANN errors)
- Trivially inspectable (no index structures to decode)

The crossover point where FAISS wins is roughly 500K–1M vectors.

### Whitespace tokenization for BM25

NLTK stopword lists and Porter stemming are standard in academic IR systems. We deliberately avoid them:
- BM25's IDF naturally devalues stopwords (appear in every doc → near-zero IDF)
- Stemming can conflate words that shouldn't be treated identically
- Keeping the tokenizer simple means the scores are interpretable

If you need better tokenization, swap in `nltk.word_tokenize` in `BM25._tokenize()` — the rest of the code is unchanged.

### sentence-transformers library vs. raw transformers

The `sentence-transformers` library is a convenience wrapper that handles mean pooling, normalization, and batching automatically. Using raw `transformers` instead demonstrates:
1. What the sentence-transformers library is actually doing internally
2. That you understand mean pooling and L2 normalization — not just calling `.encode()`
3. That the approach generalizes to any HuggingFace model, not just sentence-transformers models

For production use, the sentence-transformers library is a perfectly good choice.

### temperature=0.2 for generation

Lower temperatures make the model more deterministic and less likely to "creatively" deviate from the provided context. For RAG, where we explicitly want the model to stick to the context, 0.2 is a good default. Temperature 0.0 (fully deterministic) can cause the model to repeat itself.

---

*This document corresponds directly to the code in `src/`. Each section maps to one file. The implementation is intentionally minimal — no layers of abstraction between you and the algorithm.*
