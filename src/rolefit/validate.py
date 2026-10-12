"""Re-identification tests for role vectors.

Does a player's vector built from some matches find the same player's
vector built from others (split-half), or from another team (club <->
country)? Compared with two chance levels: random gallery guessing, and
guessing a random gallery row with the same position group (the thing a
role vector must beat to be worth anything).
"""
from __future__ import annotations

import numpy as np


def cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return A @ B.T


def wilson(k: int, n: int, z: float = 1.645) -> tuple[float, float]:
    """90% Wilson interval."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def reid(query: np.ndarray, gallery: np.ndarray, correct: list[set], exclude: list[set] | None = None,
         groups_q=None, groups_g=None, ks=(1, 5)) -> dict:
    """Rank gallery rows by cosine similarity to each query row.

    correct[i]: gallery indices that are right for query i. exclude[i]:
    gallery indices removed from its candidate list. Queries with no
    surviving right answer are skipped.
    """
    S = cosine(query, gallery)
    hits = {k: 0 for k in ks}
    chance = {k: 0.0 for k in ks}
    pos = {k: 0.0 for k in ks}
    n = 0
    for i in range(len(query)):
        ex = exclude[i] if exclude else set()
        right = correct[i] - ex
        if not right:
            continue
        n += 1
        cand = np.array([j for j in range(len(gallery)) if j not in ex])
        order = cand[np.argsort(-S[i, cand])]
        first = int(np.argmax([j in right for j in order]))
        for k in ks:
            hits[k] += first < k
            chance[k] += min(1.0, k * len(right) / len(cand))
        if groups_q is not None:
            same = [j for j in cand if groups_g[j] == groups_q[i]]
            right_same = sum(1 for j in right if groups_g[j] == groups_q[i])
            for k in ks:
                pos[k] += min(1.0, k * right_same / len(same)) if same else 0.0
    out = {"n": n}
    for k in ks:
        lo, hi = wilson(hits[k], n)
        out[f"top{k}"] = hits[k] / n if n else float("nan")
        out[f"top{k}_ci90"] = [lo, hi]
        out[f"chance_top{k}"] = chance[k] / n if n else float("nan")
        if groups_q is not None:
            out[f"position_baseline_top{k}"] = pos[k] / n if n else float("nan")
    return out
