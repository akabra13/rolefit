import numpy as np
import polars as pl

from rolefit.level import crossfit_v7, fit_two_way, linked_players, quality_index


def world(seed=0, n=600):
    """Players with true ability; three leagues with true difficulty; a shared
    tournament links them. Observed q = ability - difficulty + noise."""
    rng = np.random.default_rng(seed)
    diff = {"strong": 0.8, "mid": 0.0, "weak": -0.8, "cup": 0.3}
    players, keys, q = [], [], []
    for p in range(n):
        a = rng.normal()
        home = ["strong", "mid", "weak"][p % 3]
        for k in [home] + (["cup"] if p % 2 == 0 else []) + (["mid"] if p % 7 == 0 and home != "mid" else []):
            players.append(p)
            keys.append(k)
            q.append(a - diff[k] + rng.normal(scale=0.3))
    return players, keys, np.array(q), diff


def test_two_way_recovers_league_difficulty():
    players, keys, q, diff = world()
    _, lam = fit_two_way(None, players, keys, q, np.ones(len(q)))
    assert lam["weak"] > lam["mid"] > lam["strong"]
    assert abs((lam["weak"] - lam["strong"]) - 1.6) < 0.3


def test_v7_league_adjustment_helps_out_of_sample():
    players, keys, q, _ = world()
    r = crossfit_v7(players, keys, q, np.ones(len(q)), n_boot=100)
    assert r["improvement"] > 0 and r["ci90"][0] > 0


def test_linked_players_and_quality_index():
    assert linked_players([1, 1, 2], ["a", "b", "a"]) == {"a": 1, "b": 1}
    u = pl.DataFrame({"unit": ["1", "2", "3", "4"], "group": ["CM"] * 4,
                      "value_p100": [1.0, 2.0, 3.0, 4.0], "loss_per_touch": [0.4, 0.3, 0.2, 0.1]})
    q = quality_index(u).sort("unit")["q"].to_list()
    assert q == sorted(q)      # more value and fewer losses both raise q
