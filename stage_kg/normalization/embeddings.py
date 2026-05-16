"""
Embedding utilities for entity normalization (Appendix C.2).

Uses a Gemini embedding model when available. Falls back to a TF-IDF-style
bag-of-words vector if the embedding API is unavailable, unsupported for the
current SDK/API version, or errors at runtime, so the pipeline degrades
gracefully instead of failing normalization outright.
"""

import logging
import math
import os
import re
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ Gemini embeddings

def get_gemini_embedder(api_key: str):
    """Return a callable(texts) -> np.ndarray of shape (N, D)."""
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        model_candidates = [
            os.getenv("GEMINI_EMBED_MODEL", "").strip(),
            "gemini-embedding-001",
            "text-embedding-004",
        ]
        model_candidates = [m for m in model_candidates if m]

        def embed(texts: List[str]) -> np.ndarray:
            if not texts:
                return np.zeros((0, 768))
            last_error = None
            for model_name in model_candidates:
                try:
                    # Batch in groups of 100 (API limit)
                    all_vecs = []
                    for i in range(0, len(texts), 100):
                        batch = texts[i : i + 100]
                        resp = client.models.embed_content(
                            model=model_name,
                            contents=batch,
                        )
                        vecs = [e.values for e in resp.embeddings]
                        all_vecs.extend(vecs)
                    arr = np.array(all_vecs, dtype=np.float32)
                    norms = np.linalg.norm(arr, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1, norms)
                    logger.debug("Using Gemini embedding model %s for normalization", model_name)
                    return arr / norms
                except Exception as e:
                    last_error = e
                    logger.warning(
                        "Gemini embedding model %s failed at runtime (%s); trying fallback model/TF-IDF",
                        model_name,
                        e,
                    )

            logger.warning(
                "All Gemini embedding models failed; falling back to TF-IDF for this normalization batch (%s)",
                last_error,
            )
            return _tfidf_embedder(texts)

        logger.debug("Using Gemini embedder with model fallbacks for normalization")
        return embed

    except Exception as e:
        logger.warning("Gemini embedder unavailable (%s); falling back to TF-IDF", e)
        return _tfidf_embedder


def _tfidf_embedder(texts: List[str]) -> np.ndarray:
    """
    Lightweight bag-of-words fallback (no external deps beyond numpy).
    Returns L2-normalized TF-IDF vectors.
    """
    if not texts:
        return np.zeros((0, 1))

    tokenize = lambda s: re.findall(r"[a-z0-9]+", s.lower())
    corpus = [tokenize(t) for t in texts]

    # Build vocabulary
    vocab = {}
    for doc in corpus:
        for tok in doc:
            if tok not in vocab:
                vocab[tok] = len(vocab)

    D = len(vocab) or 1
    N = len(texts)
    mat = np.zeros((N, D), dtype=np.float32)

    # TF
    for i, doc in enumerate(corpus):
        for tok in doc:
            mat[i, vocab[tok]] += 1

    # IDF
    df = (mat > 0).sum(axis=0)
    idf = np.log((N + 1) / (df + 1)) + 1
    mat = mat * idf

    # L2 normalize
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return mat / norms


# ------------------------------------------------------------------ similarity

def dual_similarity(
    name_vecs: np.ndarray,
    desc_vecs: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Combined similarity matrix: α·sim_name + (1-α)·sim_desc  (Appendix C.2).

    Args:
        name_vecs: (N, D_n) L2-normalized name embeddings.
        desc_vecs: (N, D_d) L2-normalized description embeddings.
        alpha: Weight on name similarity (paper does not fix this; 0.6 is a sensible default).

    Returns:
        (N, N) symmetric similarity matrix, values in [-1, 1].
    """
    S_name = name_vecs @ name_vecs.T
    S_desc = desc_vecs @ desc_vecs.T
    return alpha * S_name + (1 - alpha) * S_desc
