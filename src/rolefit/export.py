"""Build the Library payload: everything the browser page needs, as JSON.

Per exported unit (outfield, >= 900 minutes): identity, position-relative
traits x, leave-one-out slot demand D, and the portability-weighted
embedding e (the V3 space). Plus the V4 translation coefficients per trait
so the page can predict how any player would play in any slot.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from rolefit.fit import _ridge, build_samples, loo_slot_demand
from rolefit.movers import CROSS_TEAM, mover_pairs
from rolefit.similar import merge_weights
from rolefit.portability import embed
from rolefit.traits import TRAITS

MIN_MINUTES = 900.0


def r(v, nd=2):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def translation_coefficients(adj, loo, units, kept) -> dict:
    pairs = mover_pairs(units, 270.0)
    S = build_samples(adj, loo, pairs, kept, CROSS_TEAM)
    out = {}
    for t, name in enumerate(kept):
        F = np.column_stack([S["x_old"][:, t], S["d_new"][:, t], S["d_old"][:, t]])
        coef, b0, _, _ = _ridge(F, S["y"][:, t])
        out[name] = [float(c) for c in coef] + [float(b0)]
    return {"n_pairs": int(len(S["y"])), "n_players": int(len(np.unique(S["player"]))), "coef": out}


def build_payload(units: pl.DataFrame, adj: pl.DataFrame, weights: dict, kept: list,
                  team_names: dict, min_minutes: float = MIN_MINUTES) -> dict:
    loo = loo_slot_demand(adj, kept)
    frame = (adj.filter(pl.col("minutes") >= min_minutes)
                .join(loo, on="unit")
                .join(units.select("unit", "player", "nickname", "competition", "season", "gender",
                                   "position"), on="unit", how="left"))
    E = embed(frame, merge_weights(weights))
    tr = translation_coefficients(adj, loo, units.filter(pl.col("group").fill_null("OTHER") != "GK"), kept)

    X = frame.select([f"{t}__pos" for t in kept]).to_numpy().astype(float)
    D = frame.select([f"D_{t}" for t in kept]).to_numpy().astype(float)
    sd = np.nanstd(X, axis=0)
    players = []
    for i, row in enumerate(frame.select("unit", "player_id", "player", "nickname", "team_id", "competition",
                                         "season", "gender", "group", "position", "minutes").iter_rows(named=True)):
        players.append({
            "u": row["unit"], "id": row["player_id"],
            "n": row["player"] or row["nickname"] or str(row["player_id"]),
            "t": team_names.get(row["team_id"], str(row["team_id"])), "c": row["competition"],
            "s": row["season"], "g": row["gender"], "p": row["group"], "pos": row["position"],
            "m": int(round(row["minutes"])),
            "x": [r(v) for v in X[i]], "d": [r(v) for v in D[i]], "e": [r(v, 2) for v in E[i]],
        })
    return {
        "traits": [{"k": t, "family": TRAITS.get(t, "other"), "sd": r(s, 3)} for t, s in zip(kept, sd)],
        "translation": tr, "players": players, "n_embed_dims": int(E.shape[1]),
    }
