"""From a team's weaknesses to the type of player it needs.

1. Peer z-scores: each team metric against the other teams in its
   competition-season (or the gender pool when the season is too small).
2. Weakness: the z-score turned so that negative = worse for the team, times
   the metric's importance for results (|standardised coefficient| from the
   V9 model) when V9 passes, otherwise equal weights.
3. Where it comes from: each team metric is linked to the player traits that
   produce it (TRAIT_LINKS, a stated football judgement, not learnt). The slot
   (position group) whose players fall furthest below the league's same-position
   average on those traits is where the need sits.
4. Player type: position group + the linked traits it should be strong in.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from rolefit.team import GOOD_HIGH, METRICS

# team metric -> (player traits that produce it, +1 if more of the trait helps / -1 if less helps)
TRAIT_LINKS = {
    "prog_pp": {"prog_pass_p100": 1, "prog_carry_p100": 1},
    "final_third_pp": {"carry_att_entry_p100": 1, "dst_att_share": 1, "prog_pass_p100": 1},
    "box_entries_pp": {"box_entry_p100": 1, "carry_box_entry_p100": 1, "dst_box_share": 1},
    "shots_pp": {"shots_p100": 1, "touch_box_p100": 1, "xg_p100": 1},
    "own_third_loss_pp": {"loss_per_touch": -1, "up_pass_cmp": 1, "up_retain": 1},
    "pass_cmp": {"pass_cmp": 1, "pass_cmp_oe": 1},
    "possession_share": {"touches_p100": 1, "pass_cmp": 1, "up_retain": 1},
    "setpiece_xg_pp": {"aerials_p100": 1, "header_shot_share": 1},
    "xg_against_pp": {"tackles_p100o": 1, "intercept_p100o": 1, "regains_p100o": 1, "dribbled_past_p100o": -1},
    "box_entries_against_pp": {"tackles_p100o": 1, "intercept_p100o": 1, "clearance_p100o": 1, "dribbled_past_p100o": -1},
    "high_regain_pp": {"pressures_p100o": 1, "press_att_share": 1, "regains_p100o": 1},
    "counter_xg_against_pp": {"recovery_p100o": 1, "regains_p100o": 1, "fouls_p100o": 1},
    "setpiece_xg_against_pp": {"aerials_p100": 1, "clearance_p100o": 1},
    "aerial_win_share": {"aerials_p100": 1, "header_share": 1},
    "xg_for_pp": {"xgchain_p100": 1, "shot_assist_p100": 1, "xg_p100": 1},
}
LABEL = {
    "xg_for_pp": "chance quality created", "shots_pp": "shot volume", "box_entries_pp": "getting into the box",
    "final_third_pp": "reaching the final third", "prog_pp": "progressing the ball",
    "own_third_loss_pp": "losing the ball in its own third", "pass_cmp": "passing accuracy",
    "possession_share": "keeping the ball", "setpiece_xg_pp": "set-piece threat",
    "xg_against_pp": "chances conceded", "box_entries_against_pp": "box entries conceded",
    "high_regain_pp": "winning the ball high", "counter_xg_against_pp": "counter-attacks conceded",
    "setpiece_xg_against_pp": "set-piece chances conceded", "aerial_win_share": "aerial duels",
}


def peer_z(ts: pl.DataFrame, min_teams: int = 8, min_matches: int = 10) -> pl.DataFrame:
    """z_{m} per team-season vs its competition-season peers (teams with
    >= min_matches); seasons with fewer than min_teams such teams use the
    whole pool (column peer_scope says which)."""
    full = ts.with_columns(_n=(pl.col("matches") >= min_matches).sum().over("key"))
    in_league = pl.col("_n") >= min_teams
    out = full.with_columns(peer_scope=pl.when(in_league).then(pl.lit("league")).otherwise(pl.lit("pool")))
    ok = pl.col("matches") >= min_matches
    zs = []
    for m in METRICS:
        v = pl.when(ok).then(pl.col(m))
        lz = (pl.col(m) - v.mean().over("key")) / v.std().over("key")
        gz = (pl.col(m) - v.mean()) / v.std()
        zs.append(pl.when(in_league).then(lz).otherwise(gz).alias(f"z_{m}"))
    return out.with_columns(zs).drop("_n")


def weaknesses(row: dict, importance: dict | None, top: int = 3) -> list:
    """Ranked weaknesses for one team-season row (needs z_{m} columns)."""
    out = []
    for m in METRICS:
        good = GOOD_HIGH.get(m)
        z = row.get(f"z_{m}")
        if good is None or z is None or not np.isfinite(z):
            continue
        zg = z if good else -z
        w = importance.get(m, 0.0) if importance else 1.0
        out.append({"metric": m, "label": LABEL.get(m, m), "z_good": round(float(zg), 2),
                    "weight": round(float(w), 3), "score": round(float(-zg * w), 3)})
    out.sort(key=lambda r: -r["score"])
    return [r for r in out if r["z_good"] < -0.5][:top]


def group_contribution(all_slots: pl.DataFrame, metric: str) -> dict:
    """How much each position group typically produces of a metric's linked
    traits: mean over the league's slots of the slot's share of the team total
    (share_{t}), averaged over linked traits that have shares. Groups without
    share data get the mean of the others."""
    links = TRAIT_LINKS.get(metric, {})
    cols = [f"share_{t}" for t, sgn in links.items() if sgn > 0 and f"share_{t}" in all_slots.columns]
    if not cols or all_slots.is_empty():
        return {}
    g = all_slots.group_by("group").agg([pl.col(c).mean() for c in cols])
    out = {r["group"]: float(np.nanmean([r[c] for c in cols if r[c] is not None])) for r in g.iter_rows(named=True)}
    return out


def slot_for_need(metric: str, team_slots: pl.DataFrame, contribution: dict | None = None) -> dict | None:
    """team_slots: this team-season's rows of slot_demand (group + rel_{t}).
    Returns the group where the shortfall on the metric's linked traits, weighted
    by how much that position usually contributes to them, is largest."""
    links = TRAIT_LINKS.get(metric)
    if not links or team_slots.is_empty():
        return None
    best = None
    for r in team_slots.iter_rows(named=True):
        vals = [s * r[f"rel_{t}"] for t, s in links.items() if f"rel_{t}" in r and r[f"rel_{t}"] is not None
                and np.isfinite(r[f"rel_{t}"])]
        if not vals:
            continue
        w = (contribution or {}).get(r["group"], 1.0)
        sc = float(np.mean(vals)) * (w if contribution else 1.0)
        if best is None or sc < best["shortfall"]:
            best = {"group": r["group"], "shortfall": round(sc, 2),
                    "traits": {t: round(float(r[f"rel_{t}"]), 2) for t in links if f"rel_{t}" in r
                               and r[f"rel_{t}"] is not None and np.isfinite(r[f"rel_{t}"])}}
    return best
