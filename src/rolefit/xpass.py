"""Expected pass completion, cross-fitted by competition-season.

A pass is only ever scored by a model that never saw its competition-season,
so "completion over expected" is not flattered by the model having learnt
the player's own passes.
"""
from __future__ import annotations

import numpy as np
import polars as pl

HEIGHTS = ("Ground Pass", "Low Pass", "High Pass")
BODY = ("Right Foot", "Left Foot", "Head", "Other")
SUBTYPES = ("Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in", "Interception", "Recovery")
PATTERNS = ("Regular Play", "From Counter", "From Free Kick", "From Corner", "From Throw In",
            "From Goal Kick", "From Keeper", "From Kick Off", "Other")
FEATURES = ("x", "y", "end_x", "end_y", "length", "cos", "sin", "height_c", "body_c",
            "under_pressure_i", "sub_c", "pattern_c")


def _code(col: str, cats: tuple) -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).replace_strict(list(cats), list(range(len(cats))), default=len(cats)).cast(pl.Int8)


def pass_features(passes: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """Add model features to a frame of passes (needs x, y, end_x, end_y,
    height, body_part, under_pressure, sub_type, play_pattern, outcome)."""
    dx, dy = pl.col("end_x") - pl.col("x"), pl.col("end_y") - pl.col("y")
    length = (dx ** 2 + dy ** 2).sqrt()
    return passes.lazy().with_columns(
        length=length,
        cos=pl.when(length > 0).then(dx / length).otherwise(0.0),
        sin=pl.when(length > 0).then(dy / length).otherwise(0.0),
        height_c=_code("height", HEIGHTS), body_c=_code("body_part", BODY),
        under_pressure_i=pl.col("under_pressure").cast(pl.Int8),
        sub_c=_code("sub_type", SUBTYPES), pattern_c=_code("play_pattern", PATTERNS),
        complete=pl.col("outcome").is_null().cast(pl.Int8),
    )


def crossfit_xpass(passes: pl.DataFrame, group: str = "key", n_folds: int = 5,
                   seed: int = 0, log=print) -> pl.DataFrame:
    """Out-of-fold P(complete) for every pass. `passes` must already carry
    FEATURES, `complete`, `event_id` and the `group` column. Groups (whole
    competition-seasons) are assigned to folds at random."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss, roc_auc_score

    groups = passes[group].unique().sort().to_list()
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(groups, rng.permutation(len(groups)) % n_folds))
    fold = passes[group].replace_strict(fold_of).to_numpy()
    X = passes.select(FEATURES).to_numpy().astype(np.float32)
    y = passes["complete"].to_numpy()
    cat = [FEATURES.index(c) for c in ("height_c", "body_c", "sub_c", "pattern_c")]
    out = np.zeros(len(y))
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        if not te.any():
            continue
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                           categorical_features=cat, random_state=seed)
        m.fit(X[tr], y[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
    log(f"xpass out-of-fold: n={len(y)} AUC={roc_auc_score(y, out):.3f} "
        f"logloss={log_loss(y, np.clip(out, 1e-6, 1 - 1e-6)):.3f} base rate={y.mean():.3f}")
    return pl.DataFrame({"event_id": passes["event_id"], "xp": out})
