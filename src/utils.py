"""
Shared utility functions used across pipeline components.
"""

from __future__ import annotations

import numpy as np


def detect_device() -> str:
    """
    Auto-detect the best available torch device.

    Returns:
        "cuda", "mps", or "cpu".
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of the top-k highest scores, sorted descending.

    Uses argpartition for O(N) selection when k < N, falling back to
    full argsort when k == N.

    Args:
        scores: 1-D array of scores.
        k:      Number of top indices to return (clamped to len(scores)).

    Returns:
        np.ndarray of shape (k,) with indices into *scores*.
    """
    k = min(k, len(scores))
    if k == 0:
        return np.array([], dtype=np.intp)
    if k == len(scores):
        return np.argsort(scores)[::-1]
    partition = np.argpartition(scores, -k)[-k:]
    return partition[np.argsort(scores[partition])[::-1]]
