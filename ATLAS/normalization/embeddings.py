import logging
import re
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def tfidf_embedder(texts: List[str]) -> np.ndarray:
    """L2-normalized bag-of-words TF-IDF vectors. Numpy-only."""
    if not texts:
        return np.zeros((0, 1))

    tokenize = lambda s: re.findall(r"[a-z0-9]+", s.lower())
    corpus = [tokenize(t) for t in texts]

    vocab = {}
    for doc in corpus:
        for tok in doc:
            if tok not in vocab:
                vocab[tok] = len(vocab)

    D = len(vocab) or 1
    N = len(texts)
    mat = np.zeros((N, D), dtype=np.float32)

    for i, doc in enumerate(corpus):
        for tok in doc:
            mat[i, vocab[tok]] += 1

    df = (mat > 0).sum(axis=0)
    idf = np.log((N + 1) / (df + 1)) + 1
    mat = mat * idf

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return mat / norms


def dual_similarity(
    name_vecs: np.ndarray,
    desc_vecs: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """Combined similarity: α·sim_name + (1-α)·sim_desc. Returns (N, N) in [-1, 1]."""
    S_name = name_vecs @ name_vecs.T
    S_desc = desc_vecs @ desc_vecs.T
    return alpha * S_name + (1 - alpha) * S_desc
