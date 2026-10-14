"""'Plays like' search on the portability-weighted embedding (same space V3 tested)."""
from __future__ import annotations

import numpy as np
import polars as pl

from rolefit.portability import embed


def merge_weights(weights: dict) -> dict:
    """portability.json holds one weight set per player fold. For a final
    embedding average the weights and take fold 0's version choice."""
    folds = list(weights.values())
    return {t: (folds[0][t][0], float(np.mean([f[t][1] for f in folds]))) for t in folds[0]}


def find_units(units: pl.DataFrame, name: str, min_minutes: float = 900.0) -> pl.DataFrame:
    """Units whose player/nickname contains `name` (case-insensitive), most minutes first."""
    n = name.lower()
    m = ((pl.col("player").fill_null("").str.to_lowercase().str.contains(n, literal=True))
         | (pl.col("nickname").fill_null("").str.to_lowercase().str.contains(n, literal=True)))
    return units.filter(m & (pl.col("minutes") >= min_minutes)).sort("minutes", descending=True)


def plays_like(frame: pl.DataFrame, E: np.ndarray, unit: str, k: int = 8) -> pl.DataFrame:
    """Nearest other players (by Euclidean distance) to `unit`; one row per
    player, their closest unit."""
    ids = frame["unit"].to_list()
    i = ids.index(unit)
    d = np.sqrt(((E - E[i]) ** 2).sum(1))
    order = np.argsort(d)
    pid = frame["player_id"].to_list()
    seen, rows = {pid[i]}, []
    for j in order:
        if pid[j] in seen:
            continue
        seen.add(pid[j])
        rows.append({"unit": ids[j], "distance": round(float(d[j]), 3)})
        if len(rows) == k:
            break
    return pl.DataFrame(rows)


def build_space(adj: pl.DataFrame, weights: dict, min_minutes: float = 900.0):
    f = adj.filter(pl.col("minutes") >= min_minutes)
    return f, embed(f, merge_weights(weights))
