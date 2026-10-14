import numpy as np
import polars as pl

from rolefit.fit import (build_samples, crossfit, fit_coefficients, loo_slot_demand, player_bootstrap,
                         standardised_error)
from rolefit.movers import mover_pairs

TR = ("t_player", "t_system", "t_mixed")


def world(seed=0, n_players=500):
    """t_player travels with the player; t_system is set by the team's slot;
    t_mixed is half and half. Each player has two club stints."""
    rng = np.random.default_rng(seed)
    rows = []
    slot_eff = {}
    for p in range(n_players):
        pe = rng.normal(size=3)
        for stint in range(2):
            team = int(rng.integers(0, 40))
            key = f"k{stint}"
            g = "CM"
            se = slot_eff.setdefault((key, team), rng.normal(size=3))
            for mate in range(3):       # slot-mates so leave-one-out demand exists
                pid = 10_000 + p * 10 + stint * 5 + mate
                m_eff = rng.normal(size=3) * 0.2
                rows.append({"unit": f"{pid}|{team}|{key}", "player_id": pid, "team_id": team, "key": key,
                             "group": g, "minutes": 1500.0,
                             **{f"{t}__pos": float(v) for t, v in zip(TR, (se[0] * 0 + m_eff[0], se[1] + m_eff[1],
                                                                           0.5 * se[2] + m_eff[2]))}})
            noise = rng.normal(scale=0.3, size=3)
            rows.append({"unit": f"{p}|{team}|{key}", "player_id": p, "team_id": team, "key": key, "group": g,
                         "minutes": 1500.0,
                         "t_player__pos": float(pe[0] + noise[0]),
                         "t_system__pos": float(se[1] + noise[1]),
                         "t_mixed__pos": float(0.5 * pe[2] + 0.5 * se[2] + noise[2])})
    return pl.DataFrame(rows)


def prep(seed=0):
    adj = world(seed)
    units = adj.with_columns(competition=pl.lit("La Liga"))
    pairs = mover_pairs(units.filter(pl.col("player_id") < 10_000), 270)
    loo = loo_slot_demand(adj, TR)
    return adj, loo, pairs


def test_loo_demand_excludes_the_unit_itself():
    adj = pl.DataFrame([{"unit": u, "key": "k", "team_id": 1, "group": "CM", "minutes": 1000.0,
                         "t__pos": v} for u, v in (("a", 10.0), ("b", 2.0), ("c", 4.0))])
    d = loo_slot_demand(adj, ("t",)).sort("unit")["D_t"].to_list()
    assert d == [3.0, 7.0, 6.0]
    adj2 = adj.with_columns(minutes=pl.when(pl.col("unit") == "a").then(5000.0).otherwise(100.0))
    assert loo_slot_demand(adj2, ("t",)).sort("unit")["D_t"].to_list()[0] is None   # others < 270 min


def test_translation_recovers_where_each_trait_lives():
    adj, loo, pairs = prep()
    S = build_samples(adj, loo, pairs, TR, ("transfer",))
    assert len(S["y"]) > 400
    c = fit_coefficients(S, TR)
    assert c["t_player"]["a_player"] > 0.6 and abs(c["t_player"]["b_new_slot"]) < 0.2
    assert c["t_system"]["b_new_slot"] > 0.4 and abs(c["t_system"]["a_player"]) < 0.2
    assert c["t_mixed"]["a_player"] > 0.2 and c["t_mixed"]["b_new_slot"] > 0.2


def test_full_model_beats_each_single_source_baseline():
    adj, loo, pairs = prep()
    S = build_samples(adj, loo, pairs, TR, ("transfer",))
    err = crossfit(S)
    per = {m: standardised_error(err[m], S["y"]) for m in err}
    assert per["full"][2] < per["player_only"][2] and per["full"][2] < per["system_only"][2]  # mixed trait
    assert per["player_only"][0] < 0.35 and per["system_only"][1] < 0.7
    assert per["position"][0] > 0.9
    boot = player_bootstrap(S, err, n_boot=60)
    assert boot["full_vs_position"]["ci90"][0] > 0


def test_context_models_help_only_when_context_matters():
    from rolefit.fit import CTX_MODELS, context_bootstrap
    rng = np.random.default_rng(0)
    n, T = 1500, 2
    x_old = rng.normal(size=(n, T))
    O_new = rng.normal(size=(n, T))
    ctx = rng.normal(size=(n, 6))
    player = rng.integers(0, 300, n)

    def sample(y):
        return {"x_old": x_old, "d_new": rng.normal(size=(n, T)), "d_old": rng.normal(size=(n, T)),
                "y": y, "player": player, "O_new": O_new, "ctx": ctx}

    matters = sample(0.5 * x_old + 0.8 * O_new + 0.1 * rng.normal(size=(n, T)))
    ignores = sample(0.8 * x_old + 0.3 * rng.normal(size=(n, T)))
    for S, expect in ((matters, True), (ignores, False)):
        err = crossfit(S)
        c = context_bootstrap(S, err, n_boot=80)
        assert (c["full_ctx"]["ci90"][0] > 0) == expect or (not expect and c["full_ctx"]["improvement_over_full"] < 0.02)
    assert set(CTX_MODELS) <= set(crossfit(matters))


def test_add_context_builds_the_new_teams_context_without_the_mover():
    from rolefit.context import others_demand
    from rolefit.fit import add_context
    rows = []
    for team, vals in ((1, (1.0, 3.0, 5.0)), (2, (10.0, 20.0, 30.0))):
        for i, v in enumerate(vals):
            rows.append({"unit": f"{team}{i}|{team}|c|s", "key": "c_s", "team_id": team, "group": "CM",
                         "minutes": 2000.0, "t__pos": v})
    adj = pl.DataFrame(rows)
    others = others_demand(adj, ("t",))
    shares = pl.DataFrame({"key": ["c_s", "c_s"], "team_id": [1, 2], "f_433": [1.0, 0.0], "f_442": [0.0, 1.0]})
    S = {"y": np.zeros((2, 1)), "x_old": np.zeros((2, 1)), "d_new": np.zeros((2, 1)), "d_old": np.zeros((2, 1)),
         "player": np.array([10, 11]), "unit_new": np.array(["11|1|c|s", "20|2|c|s"])}
    out = add_context(S, others, shares, ("t",), n_pca=1)
    assert np.allclose(out["O_new"].ravel(), [(1.0 + 5.0) / 2, (20.0 + 30.0) / 2])
    assert out["ctx"][0][0] == 1.0 and out["ctx"][1][1] == 1.0
