"""Level: how good a player is, on a scale comparable across leagues.

Everything else in RoleFit is relative to the player's own competition-season,
so a dominant player in a weak league looks like a star. This module puts a
common scale under it.

quality index q   per unit: mean of z-scored output/security traits, z-scored
                  within gender and position group pooled across ALL leagues
                  (value added, chance-chain involvement, completion over
                  expected, losses per touch reversed, retention under pressure)
two-way model     q_u = theta_player + lambda_competition_season + noise,
                  minutes-weighted, ridge-shrunk, lambda centred. A positive
                  lambda means players look better there than elsewhere, i.e.
                  an easier competition; league strength = -lambda.
                  International tournaments link players from every league.
level             q_u - lambda_key: the unit's output adjusted for where it
                  was produced; reported as a percentile within gender and
                  position group.
"""
from __future__ import annotations

import numpy as np
import polars as pl

QUALITY = ("value_p100", "xgchain_p100", "pass_cmp_oe", "loss_per_touch", "up_retain")
SIGN = {"loss_per_touch": -1.0}
K_THETA = 2.0       # pseudo-units (of 900 minutes) of a player at the population mean
K_LAMBDA = 5.0      # pseudo-units of a competition-season at the average strength


def quality_index(units: pl.DataFrame, traits=QUALITY) -> pl.DataFrame:
    """unit, q. Each trait z-scored within position group (pooled over all
    competitions), sign-corrected, then averaged over available traits."""
    present = [t for t in traits if t in units.columns]
    u = units.with_columns(group=pl.col("group").fill_null("OTHER"))
    zs = []
    for t in present:
        v = pl.when(pl.col(t).is_finite()).then(pl.col(t))
        z = (v - v.mean().over("group")) / v.std().over("group")
        zs.append((SIGN.get(t, 1.0) * z).alias(f"_z_{t}"))
    u = u.with_columns(zs)
    zc = [f"_z_{t}" for t in present]
    return u.select("unit", q=pl.mean_horizontal(zc))


def fit_two_way(unit_ids, players, keys, q, w, k_theta=K_THETA, k_lambda=K_LAMBDA, iters=60):
    """Alternating ridge fits of player and competition-season effects.
    Returns (theta: dict player -> effect, lam: dict key -> effect)."""
    q, w = np.asarray(q, float), np.asarray(w, float)
    pi = {p: i for i, p in enumerate(sorted(set(players)))}
    ki = {k: i for i, k in enumerate(sorted(set(keys)))}
    P = np.array([pi[p] for p in players])
    Kx = np.array([ki[k] for k in keys])
    th, la = np.zeros(len(pi)), np.zeros(len(ki))
    for _ in range(iters):
        r = q - la[Kx]
        th = np.bincount(P, w * r, len(pi)) / (np.bincount(P, w, len(pi)) + k_theta)
        r = q - th[P]
        la = np.bincount(Kx, w * r, len(ki)) / (np.bincount(Kx, w, len(ki)) + k_lambda)
        la -= np.average(la[Kx], weights=w)
    return {p: th[i] for p, i in pi.items()}, {k: la[i] for k, i in ki.items()}


def linked_players(players, keys) -> dict:
    """Per key: how many of its players also appear in another key (what
    identifies that key's strength)."""
    by_p: dict = {}
    for p, k in zip(players, keys):
        by_p.setdefault(p, set()).add(k)
    out: dict = {}
    for p, ks in by_p.items():
        if len(ks) > 1:
            for k in ks:
                out[k] = out.get(k, 0) + 1
    return out


def crossfit_v7(players, keys, q, w, n_folds=5, seed=0, n_boot=300) -> dict:
    """Held-out test of the league adjustment. For players in a held-out fold
    with units in two different competition-seasons, predict q at B from q at
    A either as q_A (no adjustment) or q_A - lambda_A + lambda_B (lambda fitted
    without the fold). Returns MSEs and the improvement with a 90% CI that
    resamples players."""
    players, keys, q, w = np.asarray(players), np.asarray(keys), np.asarray(q, float), np.asarray(w, float)
    uniq = np.unique(players)
    fold_of = dict(zip(uniq, np.random.default_rng(seed).permutation(len(uniq)) % n_folds))
    fold = np.array([fold_of[p] for p in players])
    rows = []      # (player, sq_err_none, sq_err_adj)
    for f in range(n_folds):
        tr = fold != f
        _, lam = fit_two_way(None, players[tr], keys[tr], q[tr], w[tr])
        te = np.flatnonzero(~tr)
        by_p: dict = {}
        for i in te:
            by_p.setdefault(players[i], []).append(i)
        for p, idx in by_p.items():
            for a in idx:
                for b in idx:
                    if a == b or keys[a] == keys[b]:
                        continue
                    la, lb = lam.get(keys[a], 0.0), lam.get(keys[b], 0.0)
                    rows.append((p, (q[b] - q[a]) ** 2, (q[b] - (q[a] - la + lb)) ** 2))
    if not rows:
        return {"n_pairs": 0}
    pl_ = np.array([r[0] for r in rows])
    e0, e1 = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    up, inv = np.unique(pl_, return_inverse=True)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        wb = np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up)).astype(float)[inv]
        boots.append((wb * e0).sum() / wb.sum() - (wb * e1).sum() / wb.sum())
    return {"n_pairs": int(len(rows)), "n_players": int(len(up)), "mse_no_adjustment": round(float(e0.mean()), 4),
            "mse_league_adjusted": round(float(e1.mean()), 4),
            "improvement": round(float(e0.mean() - e1.mean()), 4),
            "ci90": [round(float(np.percentile(boots, 5)), 4), round(float(np.percentile(boots, 95)), 4)]}
