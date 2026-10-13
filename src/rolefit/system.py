"""Separate a trait into league-position level, team-wide system, and player.

    trait = league_position_mean + team_effect + player

league_position_mean   minutes-weighted mean of the trait among units in the
                       same competition-season and position group
team_effect            how the player's *teammates* (all positions) sit
                       relative to their own league-position means: a
                       leave-one-out, shrunk, minutes-weighted mean. A
                       possession side inflates everyone's pass completion;
                       this is what gets removed.

Two versions per trait come out:
  {t}__pos   trait - league_position_mean            (position-relative)
  {t}__sys   trait - league_position_mean - team_effect  (system-adjusted)

Role-specific demands (a team that asks its left-back for more carries)
are not removed here; they are the slot demand profiles of stage 3.
"""
from __future__ import annotations

import polars as pl

SHRINK_MATCHES = 10.0     # pseudo-matches of zero team effect


def system_adjust(units: pl.DataFrame, traits, shrink: float = SHRINK_MATCHES) -> pl.DataFrame:
    """`units` needs unit, key, team_id, group, minutes and the trait columns.
    Returns unit plus, per trait, {t}__lpm, {t}__team, {t}__pos, {t}__sys."""
    df = units.with_columns(w=pl.col("minutes") / 90.0, group=pl.col("group").fill_null("OTHER"))
    out = df.select("unit")
    for t in traits:
        ok = pl.col(t).is_not_null() & pl.col(t).is_finite()
        wt = pl.when(ok).then(pl.col("w")).otherwise(0.0)
        val = pl.when(ok).then(pl.col(t)).otherwise(0.0)
        d = df.with_columns(_wt=wt, _v=val).with_columns(
            lpm=(pl.col("_wt") * pl.col("_v")).sum().over("key", "group")
                / pl.col("_wt").sum().over("key", "group"))
        d = d.with_columns(resid=pl.when(ok).then(pl.col(t) - pl.col("lpm")))
        d = d.with_columns(_wr=pl.col("_wt") * pl.col("resid").fill_null(0.0))
        d = d.with_columns(
            team=(pl.col("_wr").sum().over("key", "team_id") - pl.col("_wr"))
                 / (pl.col("_wt").sum().over("key", "team_id") - pl.col("_wt") + shrink))
        out = out.with_columns(
            d["lpm"].alias(f"{t}__lpm"), d["team"].alias(f"{t}__team"),
            d["resid"].alias(f"{t}__pos"), (d["resid"] - d["team"]).alias(f"{t}__sys"))
    return out


def apply_offsets(frame: pl.DataFrame, offsets: pl.DataFrame, traits) -> pl.DataFrame:
    """Give per-half (or any per-unit) rows the same two versions, using the
    full-season offsets of their unit."""
    cols = [c for t in traits for c in (f"{t}__lpm", f"{t}__team")]
    j = frame.join(offsets.select(["unit"] + cols), on="unit", how="left")
    return j.with_columns(
        [(pl.col(t) - pl.col(f"{t}__lpm")).alias(f"{t}__pos") for t in traits]
        + [(pl.col(t) - pl.col(f"{t}__lpm") - pl.col(f"{t}__team")).alias(f"{t}__sys") for t in traits])
