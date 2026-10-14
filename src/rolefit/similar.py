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


def add_absolute(adj: pl.DataFrame, traits) -> pl.DataFrame:
    """{t}__abs: the trait itself minus the mean of everyone in the same
    competition-season (all positions), so league and era effects go but the
    position stays. A defensive midfielder then sits near other defensive
    midfielders instead of near centre-backs who are 'as centre-back-like'."""
    out = adj
    for t in traits:
        raw = pl.col(f"{t}__lpm") + pl.col(f"{t}__pos")
        ok = raw.is_not_null() & raw.is_finite()
        w = pl.when(ok).then(pl.col("minutes")).otherwise(0.0)
        mean = (w * pl.when(ok).then(raw).otherwise(0.0)).sum().over("key") / w.sum().over("key")
        out = out.with_columns((raw - mean).alias(f"{t}__abs"))
    return out


def build_space(adj: pl.DataFrame, weights: dict, min_minutes: float = 900.0, absolute: bool = True):
    w = merge_weights(weights)
    f = adj.filter(pl.col("minutes") >= min_minutes)
    if absolute:
        f = add_absolute(f, list(w))
        w = {t: ("abs", wt) for t, (_, wt) in w.items()}
    return f, embed(f, w)
