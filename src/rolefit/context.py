"""Team context for a team-season: formation mix, overall style, and what the
rest of the lineup supplies.

formation_shares  share of the team's matches started in each formation family
team_style        minutes-weighted mean of every outfield player's traits
others            for one unit, the minutes-weighted mean traits of every other
                  outfield player of that team-season (what the squad already has)
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

FAMILIES = ("4231", "433", "442", "4141", "352", "343", "532", "451", "other")


def formation_family(code) -> str:
    c = str(code or "").replace("-", "")
    if c in ("4231", "4321"):
        return "4231"
    if c in ("433", "4312", "4411"):
        return "433" if c == "433" else "other"
    if c in ("442", "4222"):
        return "442"
    if c in ("4141",):
        return "4141"
    if c in ("352", "3412", "3142"):
        return "352"
    if c in ("343", "3421"):
        return "343"
    if c in ("532", "541"):
        return "532"
    if c in ("451",):
        return "451"
    return "other"


def starting_formations(events: pl.DataFrame) -> pl.DataFrame:
    """One row per (match, team) with the starting formation family. `events`
    needs match_id, team_id, type, extra, and competition_id/season_id."""
    xi = events.filter(pl.col("type") == "Starting XI").select(
        "match_id", "team_id", "competition_id", "season_id", "extra")
    codes = []
    for e in xi["extra"].to_list():
        try:
            codes.append(formation_family((json.loads(e).get("tactics") or {}).get("formation")))
        except Exception:
            codes.append("other")
    return xi.drop("extra").with_columns(family=pl.Series(codes))


def formation_shares(formations: pl.DataFrame) -> pl.DataFrame:
    """key, team_id, n_matches and f_{family} shares."""
    f = formations.with_columns(key=pl.col("competition_id").cast(pl.Utf8) + "_" + pl.col("season_id").cast(pl.Utf8))
    g = f.group_by("key", "team_id").agg(pl.len().alias("n_matches"),
                                         *[(pl.col("family") == fam).mean().alias(f"f_{fam}") for fam in FAMILIES])
    return g


def team_style(units: pl.DataFrame, traits) -> pl.DataFrame:
    """key, team_id and the minutes-weighted mean of each trait over outfield players."""
    u = units.filter(pl.col("group").fill_null("OTHER") != "GK")
    aggs = []
    for t in traits:
        ok = pl.col(t).is_not_null() & pl.col(t).is_finite()
        aggs.append(((pl.col("minutes") * pl.col(t)).filter(ok).sum() / pl.col("minutes").filter(ok).sum()).alias(f"S_{t}"))
    return u.group_by("key", "team_id").agg(aggs)


def others_demand(adj: pl.DataFrame, traits, min_other_minutes: float = 900.0) -> pl.DataFrame:
    """Per unit: minutes-weighted mean of {t}__pos over the other outfield
    units of the same team-season (null when they have too few minutes)."""
    df = adj
    out = df.select("unit")
    for t in traits:
        v = pl.col(f"{t}__pos")
        ok = v.is_not_null() & v.is_finite()
        w = pl.when(ok).then(pl.col("minutes")).otherwise(0.0)
        num = (w * pl.when(ok).then(v).otherwise(0.0)).sum().over("key", "team_id")
        d = df.with_columns(_w=w, _num=num).with_columns(
            _den=pl.col("_w").sum().over("key", "team_id") - pl.col("_w"),
            _n=num - pl.col("_w") * pl.when(ok).then(v).otherwise(0.0))
        out = out.with_columns(pl.when(d["_den"] >= min_other_minutes).then(d["_n"] / d["_den"])
                               .otherwise(None).alias(f"O_{t}"))
    return out


def style_components(style: pl.DataFrame, traits, n: int = 5):
    """PCA of team style (z-scored traits) to `n` components. Returns
    (frame with key, team_id, c0..c{n-1}, loadings)."""
    cols = [f"S_{t}" for t in traits]
    X = style.select(cols).to_numpy().astype(float)
    X = np.where(np.isfinite(X), X, np.nan)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    Z = np.nan_to_num((X - mu) / np.where(sd > 0, sd, 1.0))
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    k = min(n, len(S))
    comps = U[:, :k] * S[:k]
    out = style.select("key", "team_id").with_columns(
        [pl.Series(f"c{i}", comps[:, i]) for i in range(k)])
    return out, Vt[:k]
