import numpy as np
import polars as pl

from rolefit.roles import BLOCKS, embed, role_vectors
from rolefit.validate import reid


def _events(rng, centre_x, unit, n=300):
    xs = np.clip(rng.normal(centre_x, 8, n), 0, 119.9)
    ys = np.clip(rng.normal(40, 15, n), 0, 79.9)
    return pl.DataFrame({
        "unit": [unit] * n,
        "type": rng.choice(["Pass", "Carry", "Ball Receipt*", "Pressure"], n),
        "x": xs, "y": ys,
        "end_x": np.clip(xs + rng.normal(10, 5, n), 0, 119.9), "end_y": ys,
    })


def test_vector_blocks_are_distributions():
    rng = np.random.default_rng(0)
    units, X, off = role_vectors(_events(rng, 40, "a"))
    assert units == ["a"] and X.shape == (1, sum(BLOCKS.values()))
    for b in ("act", "rec", "dst", "def"):
        assert abs(X[0, off[b]].sum() - 1.0) < 1e-9


def test_vectors_separate_a_deep_from_an_advanced_player():
    rng = np.random.default_rng(1)
    ev = pl.concat([_events(rng, 30, "deep_A"), _events(rng, 30, "deep_B"),
                    _events(rng, 90, "high_A"), _events(rng, 90, "high_B")])
    units, X, off = role_vectors(ev)
    _, P = embed(X, off, 3)
    i = {u: k for k, u in enumerate(units)}
    sim = P @ P.T
    assert sim[i["deep_A"], i["deep_B"]] > sim[i["deep_A"], i["high_B"]]


def test_reid_perfect_and_chance():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(40, 6))
    perfect = reid(base, base + 0.01 * rng.normal(size=base.shape), [{i} for i in range(40)])
    assert perfect["top1"] == 1.0
    noise = reid(rng.normal(size=(40, 6)), rng.normal(size=(40, 6)), [{i} for i in range(40)])
    assert noise["top5"] < 0.5 and abs(noise["chance_top1"] - 1 / 40) < 1e-9


def test_reid_exclusions_and_position_baseline():
    g = ["CB"] * 5 + ["ST"] * 5
    X = np.eye(10)
    r = reid(X, X, [{i} for i in range(10)], exclude=[{(i + 1) % 10} for i in range(10)],
             groups_q=g, groups_g=g, ks=(1,))
    assert r["top1"] == 1.0 and abs(r["position_baseline_top1"] - 1 / 4) < 0.06
