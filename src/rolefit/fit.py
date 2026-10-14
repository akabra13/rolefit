"""Role translation: predict how a player will play at a new team.

For a player moving from unit `old` to unit `new` (different teams), per
trait t (all values position-relative: trait minus the league's same-
position mean):

    y_t      the player's trait in the new unit
    x_old_t  the player's trait in the old unit
    D_new_t  what the new slot asks for: minutes-weighted mean of the OTHER
             players' trait in the new slot (leave-one-out: the player's own
             new numbers must not leak into their own prediction)
    D_old_t  the same for the old slot

Models, each fitted per trait by ridge and scored out of fold (folds are
players, so nobody is predicted by a model that saw their own moves):

    full         y ~ x_old + D_new + D_old
    player_only  y ~ x_old                 (the player carries their habits)
    system_only  y ~ D_new                 (the slot dictates)
    position     y = 0                     (league position average)

The fitted coefficients say where a trait lives: a on x_old is the part that
travels with the player, b on D_new the part the new system supplies.
"""
from __future__ import annotations

import numpy as np
import polars as pl

LAMBDA = 1.0
N_FOLDS = 5
MODELS = {"full": (0, 1, 2), "player_only": (0,), "system_only": (1,), "position": ()}


def loo_slot_demand(adj: pl.DataFrame, traits, min_other_minutes: float = 270.0) -> pl.DataFrame:
    """Per unit: the other players' demand in its slot, D_{t}. `adj` needs
    unit, key, team_id, group, minutes and {t}__pos. Missing (null) when the
    rest of the slot has fewer than `min_other_minutes`."""
    df = adj.with_columns(group=pl.col("group").fill_null("OTHER"))
    out = df.select("unit")
    for t in traits:
        v = pl.col(f"{t}__pos")
        ok = v.is_not_null() & v.is_finite()
        w = pl.when(ok).then(pl.col("minutes")).otherwise(0.0)
        num = (w * pl.when(ok).then(v).otherwise(0.0)).sum().over("key", "team_id", "group")
        d = df.with_columns(_w=w, _num=num).with_columns(
            _den=pl.col("_w").sum().over("key", "team_id", "group") - pl.col("_w"),
            _n=num - pl.col("_w") * pl.when(ok).then(v).otherwise(0.0))
        out = out.with_columns(
            pl.when(d["_den"] >= min_other_minutes).then(d["_n"] / d["_den"]).otherwise(None).alias(f"D_{t}"))
    return out


def build_samples(adj: pl.DataFrame, loo: pl.DataFrame, pairs: pl.DataFrame, traits, types) -> dict:
    """Both directions of every pair of the given types. Rows with any
    missing input are dropped, identically for all models."""
    j = adj.select(["unit", "player_id"] + [f"{t}__pos" for t in traits]).join(loo, on="unit")
    cols = {u: i for i, u in enumerate(j["unit"].to_list())}
    X = j.select([f"{t}__pos" for t in traits]).to_numpy().astype(float)
    D = j.select([f"D_{t}" for t in traits]).to_numpy().astype(float)
    pid = j["player_id"].to_numpy()
    p = pairs.filter(pl.col("type").is_in(list(types)))
    old, new, pl_ = [], [], []
    for a, b, player in p.select("unit_a", "unit_b", "player_id").iter_rows():
        if a in cols and b in cols:
            old += [cols[a], cols[b]]
            new += [cols[b], cols[a]]
            pl_ += [player, player]
    old, new = np.array(old), np.array(new)
    S = {"x_old": X[old], "d_new": D[new], "d_old": D[old], "y": X[new], "player": np.array(pl_),
         "unit_old": np.array(j["unit"].to_list())[old], "unit_new": np.array(j["unit"].to_list())[new]}
    ok = np.all(np.isfinite(np.stack([S["x_old"], S["d_new"], S["d_old"], S["y"]])), axis=(0, 2))
    return {k: v[ok] for k, v in S.items()}


def _ridge(F: np.ndarray, y: np.ndarray, lam: float = LAMBDA):
    """Ridge with intercept; features standardised so lam is comparable."""
    if F.shape[1] == 0:
        return np.array([]), float(y.mean()), np.array([]), np.array([])
    mu, sd = F.mean(0), F.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (F - mu) / sd
    n, k = Z.shape
    A = Z.T @ Z + lam * n / 100.0 * np.eye(k)
    beta = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return beta / sd, float(y.mean() - (mu / sd) @ beta), mu, sd


def _predict(F, coef, intercept):
    return np.full(len(F), intercept) if F.shape[1] == 0 else F @ coef + intercept


def crossfit(S: dict, n_folds: int = N_FOLDS, seed: int = 0) -> dict:
    """Out-of-fold squared errors per model, shape (n, n_traits)."""
    n, T = S["y"].shape
    players = np.unique(S["player"])
    fold_of = dict(zip(players, np.random.default_rng(seed).permutation(len(players)) % n_folds))
    fold = np.array([fold_of[p] for p in S["player"]])
    feats = [S["x_old"], S["d_new"], S["d_old"]]
    err = {m: np.zeros((n, T)) for m in MODELS}
    for t in range(T):
        y = S["y"][:, t]
        for m, idx in MODELS.items():
            F = np.column_stack([feats[i][:, t] for i in idx]) if idx else np.zeros((n, 0))
            for f in range(n_folds):
                tr, te = fold != f, fold == f
                if not te.any():
                    continue
                if m == "position":
                    pred = np.zeros(te.sum())
                else:
                    coef, b0, _, _ = _ridge(F[tr], y[tr])
                    pred = _predict(F[te], coef, b0)
                err[m][te, t] = (y[te] - pred) ** 2
    return err


def standardised_error(err: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Per trait: MSE / variance of y (about 1.0 = no better than the mean)."""
    w = np.ones(len(y)) if weights is None else weights
    mu = (w[:, None] * y).sum(0) / w.sum()
    var = (w[:, None] * (y - mu) ** 2).sum(0) / w.sum()
    return (w[:, None] * err).sum(0) / w.sum() / np.where(var > 0, var, np.nan)


def player_bootstrap(S: dict, err: dict, n_boot: int = 300, seed: int = 0) -> dict:
    """Median-over-traits standardised error per model, and the improvement of
    `full` over each other model (positive = full is better), with 90% CIs
    from resampling whole players."""
    rng = np.random.default_rng(seed)
    players, inv = np.unique(S["player"], return_inverse=True)
    P = len(players)
    boots = {m: [] for m in err}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, P, P), minlength=P).astype(float)[inv]
        for m in err:
            boots[m].append(np.nanmedian(standardised_error(err[m], S["y"], w)))
    boots = {m: np.array(v) for m, v in boots.items()}
    out = {}
    for m in err:
        pt = float(np.nanmedian(standardised_error(err[m], S["y"])))
        out[m] = {"median_std_error": round(pt, 4),
                  "ci90": [round(float(np.percentile(boots[m], 5)), 4), round(float(np.percentile(boots[m], 95)), 4)]}
    for m in ("player_only", "system_only", "position"):
        d = boots[m] - boots["full"]
        out[f"full_vs_{m}"] = {"improvement": round(out[m]["median_std_error"] - out["full"]["median_std_error"], 4),
                               "ci90": [round(float(np.percentile(d, 5)), 4), round(float(np.percentile(d, 95)), 4)]}
    return out


def fit_coefficients(S: dict, traits) -> dict:
    """Full-model coefficients on all data: where each trait lives. In
    standardised units (features and target z-scored), so a and b are
    comparable across traits."""
    out = {}
    for t, name in enumerate(traits):
        y = S["y"][:, t]
        F = np.column_stack([S["x_old"][:, t], S["d_new"][:, t], S["d_old"][:, t]])
        sdy = y.std() or 1.0
        coef, _, _, _ = _ridge(F, y)
        sd = F.std(0)
        std_coef = coef * np.where(sd > 0, sd, 1.0) / sdy
        out[name] = {"a_player": round(float(std_coef[0]), 3), "b_new_slot": round(float(std_coef[1]), 3),
                     "c_old_slot": round(float(std_coef[2]), 3)}
    return out


def demand_gap(predicted: np.ndarray, d_new: np.ndarray, rel_weight: np.ndarray | None = None) -> np.ndarray:
    """What would change: predicted traits minus what the slot asks for
    (position-relative units), optionally weighted by how much the system
    leans on each trait."""
    gap = predicted - d_new
    return gap if rel_weight is None else gap * rel_weight
