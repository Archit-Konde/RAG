"""
Cross-encoder re-ranking using a lightweight MS-MARCO cross-encoder model.

Why cross-encoder?
  - Bi-encoders (like embeddings.py) encode query and document independently,
    enabling fast pre-computation but sacrificing some accuracy.
  - Cross-encoders receive [CLS] query [SEP] document [SEP] as a single input,
    allowing full attention between every query and document token.
  - This is 5–10× more accurate at relevance scoring but requires one forward
    pass per candidate — O(candidates) not O(1) — so it's used only on the
    small set of candidates retrieved by the hybrid retriever.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS-MARCO passage ranking (binary relevance)
  - ~22M parameters — fast enough for real-time re-ranking of 5–20 candidates
  - Raw logit output: higher → more relevant (no softmax needed for ranking)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoderReranker:
    """
    Re-ranks a list of retrieved documents using a cross-encoder model.

    Args:
        model_name: HuggingFace model ID for the cross-encoder.
        device:     "cuda", "mps", "cpu", or None for auto-detection.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: Optional[int] = None,
        batch_size: int = 16,
    ) -> List[dict]:
        """
        Re-score and sort a list of retrieved document dicts.

        Adds a "rerank_score" key (raw cross-encoder logit) to each dict.
        All original keys (text, metadata, score, index) are preserved.

        Args:
            query:      Raw query string.
            documents:  List of result dicts from the retriever.
            top_k:      If given, return only the top_k highest-scored docs.
            batch_size: Max (query, doc) pairs per forward pass.

        Returns:
            List of dicts sorted by rerank_score descending.
        """
        if not documents:
            return []

        texts = [doc["text"] for doc in documents]
        scores = self._score_pairs(query, texts, batch_size=batch_size)

        reranked = []
        for doc, score in zip(documents, scores):
            reranked.append({**doc, "rerank_score": float(score)})

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

    # ------------------------------------------------------------------
    # Pair scoring
    # ------------------------------------------------------------------

    def _score_pairs(
        self,
        query: str,
        texts: List[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Score (query, text) pairs with the cross-encoder.

        Input format: the tokenizer automatically produces:
            [CLS] query [SEP] document [SEP]

        Returns:
            np.ndarray of shape (N,), float32 raw logits.
            Higher logit = more relevant (no softmax — ranking only).
        """
        all_scores: List[float] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            queries = [query] * len(batch_texts)

            encoded = self.tokenizer(
                queries,
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                logits = self.model(**encoded).logits

            # logits shape: (B, 1) or (B,) depending on model config
            logits = logits.squeeze(-1).cpu().float().numpy()
            all_scores.extend(logits.tolist())

        return np.array(all_scores, dtype=np.float32)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model={self.model_name!r}, device={self.device!r})"
