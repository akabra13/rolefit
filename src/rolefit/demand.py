"""Slot demand: what a team asks of a position, and how much of the system
runs through it.

A slot is (competition-season, team, position group). Its demand profile is
the minutes-weighted mean trait vector of whoever played it. Two views:

  rel_{t}    demand relative to the league's same-position average, in
             standard deviations ("this team's left-back is asked for +1.8 SD
             of progressive carries")
  share_{t}  the slot's share of the team's total of volume trait t (how
             much of the team's progression, creation, defending runs
             through the slot)

The key question (V3b in the README) is whether demand belongs to the
system or to whoever happens to fill the slot: does the same team's slot
look alike across seasons even when its main player changed?
"""
from __future__ import annotations

import numpy as np
import polars as pl

ATTACK_VOLUME = ("touches_p100", "recv_p100", "prog_pass_p100", "prog_carry_p100", "box_entry_p100",
                 "shot_assist_p100", "cross_p100", "through_p100", "carry_p100", "takeon_p100",
                 "shots_p100", "xg_p100", "value_p100")
DEFENCE_VOLUME = ("pressures_p100o", "regains_p100o", "tackles_p100o", "fouls_p100o")
SLOT_KEYS = ["key", "team_id", "group"]


def slot_table(units: pl.DataFrame, traits, min_slot_minutes: float = 900.0) -> pl.DataFrame:
    """One row per slot with minutes, n_units, main player, demand traits and
    shares. `units` needs unit, player_id, team_id, key, group, minutes,
    own_poss, opp_poss and the trait columns (goalkeepers may be included:
    they only count toward team totals)."""
    u = units.with_columns(group=pl.col("group").fill_null("OTHER"))
    shares = [t for t in traits if t in ATTACK_VOLUME + DEFENCE_VOLUME]
    cnt = []
    for t in shares:
        den = "own_poss" if t in ATTACK_VOLUME else "opp_poss"
        cnt.append((pl.col(t).fill_null(0.0) * pl.col(den) / 100.0).alias(f"_c_{t}"))
    u = u.with_columns(cnt).with_columns(
        [pl.col(f"_c_{t}").sum().over("key", "team_id").alias(f"_team_{t}") for t in shares])
    outfield = u.filter(pl.col("group") != "GK")
    w = pl.col("minutes")
    aggs = [pl.col("minutes").sum().alias("minutes"), pl.len().alias("n_units"),
            pl.col("player_id").sort_by("minutes", descending=True).first().alias("main_player")]
    for t in traits:
        ok = pl.col(t).is_not_null() & pl.col(t).is_finite()
        aggs.append(((w * pl.col(t)).filter(ok).sum() / w.filter(ok).sum()).alias(t))
    for t in shares:
        aggs.append((pl.col(f"_c_{t}").sum() / pl.col(f"_team_{t}").first()).alias(f"share_{t}"))
    slots = outfield.group_by(SLOT_KEYS).agg(aggs).filter(pl.col("minutes") >= min_slot_minutes)
    return slots


def relative_demand(slots: pl.DataFrame, traits) -> pl.DataFrame:
    """Add rel_{t}: slot trait minus the league's same-position mean,
    divided by the standard deviation of that gap over all slots of the
    position group (so values are comparable across traits)."""
    out = slots
    for t in traits:
        lpm = ((pl.col("minutes") * pl.col(t)).sum().over("key", "group")
               / pl.col("minutes").sum().over("key", "group"))
        gap = (pl.col(t) - lpm)
        sd = gap.std().over("group")
        out = out.with_columns((gap / pl.when(sd > 0).then(sd).otherwise(1.0)).alias(f"rel_{t}"))
    return out


def slot_card(row: dict, traits, top: int = 5) -> dict:
    """The traits a slot is most unusual on, as (trait, rel) pairs."""
    rel = [(t, row.get(f"rel_{t}")) for t in traits]
    rel = [(t, float(v)) for t, v in rel if v is not None and np.isfinite(v)]
    rel.sort(key=lambda p: -abs(p[1]))
    return {"asks_more": [p for p in rel if p[1] > 0][:top],
            "asks_less": [p for p in rel if p[1] < 0][:top]}


def persistence(slots: pl.DataFrame, traits, rng: np.random.Generator, n_perm: int = 200) -> dict:
    """Does a team's slot look the same in another season?

    Pairs: the same team and position group in two different competition-
    seasons. Per trait, Pearson r of rel_{t} across the pair (symmetrised).
    Reported for all such pairs and for the subset where the main player
    changed. Baseline: the same pairs, but the second slot replaced by a
    random other team's slot of the same group in that season (permuted
    n_perm times; mean r reported).
    """
    s = slots.select(["key", "team_id", "group", "main_player"] + [f"rel_{t}" for t in traits])
    a = s.join(s, on=["team_id", "group"], suffix="_b").filter(pl.col("key") < pl.col("key_b"))
    if a.height < 20:
        return {"n_pairs": a.height, "traits": {}}
    changed = (a["main_player"] != a["main_player_b"]).to_numpy()
    out = {"n_pairs": a.height, "n_player_changed": int(changed.sum()), "traits": {}}

    # baseline partners: for each pair row, another slot from season key_b, same group, other team
    idx = {(r["key"], r["group"]): [] for r in s.iter_rows(named=True)}
    rows = s.to_dicts()
    for i, r in enumerate(rows):
        idx[(r["key"], r["group"])].append(i)
    cand = []
    for r in a.iter_rows(named=True):
        pool = [i for i in idx[(r["key_b"], r["group"])] if rows[i]["team_id"] != r["team_id"]]
        cand.append(pool)
    ok_rows = np.array([len(c) > 0 for c in cand])

    def corr(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 20 or x[m].std() == 0 or y[m].std() == 0:
            return None
        return float(np.corrcoef(x[m], y[m])[0, 1])

    for t in traits:
        x = a[f"rel_{t}"].to_numpy()
        y = a[f"rel_{t}_b"].to_numpy()
        base = []
        col = np.array([r[f"rel_{t}"] for r in rows], dtype=float)
        for _ in range(n_perm):
            yb = np.array([col[rng.choice(c)] if c else np.nan for c in cand])
            r0 = corr(x[ok_rows], yb[ok_rows])
            if r0 is not None:
                base.append(r0)
        out["traits"][t] = {
            "r_same_team": corr(x, y),
            "r_same_team_new_player": corr(x[changed], y[changed]),
            "r_other_team": float(np.mean(base)) if base else None,
        }
    return out
