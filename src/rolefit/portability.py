"""Portability (V2) and the portability-weighted embedding used by re-ID (V3).

A trait is portable if a player's value at one team predicts their value at
another. Weights are always estimated on one half of the *players* and used
to embed the other half, so the re-ID test never benefits from weights fitted
on the very players it is trying to recognise.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from rolefit.validate import wilson

VERSIONS = ("pos", "sys")


def pair_correlation(values: dict, pairs: pl.DataFrame, rng: np.random.Generator | None = None,
                     n_boot: int = 300) -> dict:
    """Symmetrised Pearson r of a trait across the two units of each pair
    (each pair entered in both orders). 90% CI resamples whole players."""
    rows = [(p, values.get(a), values.get(b)) for p, a, b in
            pairs.select("player_id", "unit_a", "unit_b").iter_rows()]
    rows = [(p, a, b) for p, a, b in rows
            if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    if len(rows) < 20:
        return {"n": len(rows), "r": None, "ci90": None}
    pid = np.array([r[0] for r in rows])
    a = np.array([r[1] for r in rows])
    b = np.array([r[2] for r in rows])

    def corr(idx):
        x, y = np.concatenate([a[idx], b[idx]]), np.concatenate([b[idx], a[idx]])
        if x.std() == 0 or y.std() == 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    r = corr(np.arange(len(a)))
    ci = None
    if rng is not None:
        players = np.unique(pid)
        members = {p: np.flatnonzero(pid == p) for p in players}
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(players, size=len(players), replace=True)
            boots.append(corr(np.concatenate([members[p] for p in pick])))
        boots = np.array(boots)
        boots = boots[np.isfinite(boots)]
        ci = [round(float(np.percentile(boots, 5)), 3), round(float(np.percentile(boots, 95)), 3)]
    return {"n": len(rows), "r": round(r, 3), "ci90": ci}


def portability_table(adj: pl.DataFrame, pairs: pl.DataFrame, traits, groups: dict,
                      rng: np.random.Generator | None = None) -> dict:
    """{trait: {version: {pair_group: {n, r, ci90}}}} for pair groups given
    as {name: [pair types]}."""
    out = {}
    for t in traits:
        out[t] = {}
        for v in VERSIONS:
            vals = dict(zip(adj["unit"].to_list(), adj[f"{t}__{v}"].to_list()))
            out[t][v] = {g: pair_correlation(vals, pairs.filter(pl.col("type").is_in(types)), rng)
                         for g, types in groups.items()}
    return out


def choose_weights(adj: pl.DataFrame, cross_pairs: pl.DataFrame, traits) -> dict:
    """Per trait: the version (pos or sys) whose cross-team r is higher, and
    weight max(0, r). Fitted only on the pairs passed in."""
    choice = {}
    for t in traits:
        best = ("pos", 0.0)
        for v in VERSIONS:
            vals = dict(zip(adj["unit"].to_list(), adj[f"{t}__{v}"].to_list()))
            r = pair_correlation(vals, cross_pairs)["r"]
            if r is not None and r > best[1]:
                best = (v, r)
        choice[t] = best
    return choice


def embed(frame: pl.DataFrame, choice: dict) -> np.ndarray:
    """Z-score each chosen trait column over `frame`, missing -> 0 (the mean),
    scale by sqrt(weight). Traits with weight 0 drop out."""
    cols = []
    for t, (v, w) in choice.items():
        if w <= 0:
            continue
        x = frame[f"{t}__{v}"].cast(pl.Float64).to_numpy().copy()
        x[~np.isfinite(x)] = np.nan
        mu, sd = np.nanmean(x), np.nanstd(x)
        z = (x - mu) / (sd if sd > 0 else 1.0)
        z[np.isnan(z)] = 0.0
        cols.append(z * np.sqrt(w))
    return np.column_stack(cols) if cols else np.zeros((frame.height, 1))


def player_folds(player_ids, seed: int = 0) -> dict:
    ids = sorted(set(player_ids))
    perm = np.random.default_rng(seed).permutation(len(ids))
    return {p: int(perm[i] % 2) for i, p in enumerate(ids)}


def merge_reid(parts: list[dict], ks=(1, 5)) -> dict:
    """Combine re-ID results computed on disjoint query sets."""
    n = sum(p["n"] for p in parts)
    out = {"n": n}
    if n == 0:
        return out
    for k in ks:
        hits = round(sum(p[f"top{k}"] * p["n"] for p in parts if p["n"]))
        out[f"top{k}"] = hits / n
        out[f"top{k}_ci90"] = list(wilson(hits, n))
        for key in (f"chance_top{k}", f"position_baseline_top{k}"):
            if all(key in p for p in parts):
                out[key] = sum(p[key] * p["n"] for p in parts if p["n"]) / n
    return out
