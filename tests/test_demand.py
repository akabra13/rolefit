import numpy as np
import polars as pl

from rolefit.demand import persistence, relative_demand, slot_card, slot_table

TR = ("prog_carry_p100", "pressures_p100o", "pass_len")


def world(seed=0, n_teams=30, seasons=("a", "b", "c"), team_style_sd=1.0, player_sd=0.3):
    """Each team asks a fixed amount of each trait from each group (system);
    the player on top adds noise; every season the main player changes."""
    rng = np.random.default_rng(seed)
    rows = []
    style = {(t, g): rng.normal(scale=team_style_sd, size=len(TR)) for t in range(n_teams) for g in ("FB", "CM")}
    pid = 0
    for k in seasons:
        for t in range(n_teams):
            for g in ("FB", "CM"):
                for _ in range(2):
                    pid += 1
                    v = style[(t, g)] + rng.normal(scale=player_sd, size=len(TR))
                    rows.append({"unit": f"{pid}|{t}|{k}", "player_id": pid, "team_id": t, "key": k, "group": g,
                                 "minutes": 1500.0, "own_poss": 1000, "opp_poss": 1000,
                                 **dict(zip(TR, v + (5.0 if g == "CM" else 0.0)))})
    return pl.DataFrame(rows)


def test_slot_table_weights_by_minutes_and_computes_shares():
    u = pl.DataFrame([
        {"unit": "1", "player_id": 1, "team_id": 1, "key": "k", "group": "FB", "minutes": 3000.0,
         "own_poss": 100, "opp_poss": 100, "prog_carry_p100": 10.0},
        {"unit": "2", "player_id": 2, "team_id": 1, "key": "k", "group": "FB", "minutes": 1000.0,
         "own_poss": 100, "opp_poss": 100, "prog_carry_p100": 2.0},
        {"unit": "3", "player_id": 3, "team_id": 1, "key": "k", "group": "CM", "minutes": 4000.0,
         "own_poss": 100, "opp_poss": 100, "prog_carry_p100": 8.0},
    ])
    s = slot_table(u, ("prog_carry_p100",), 900.0).filter(pl.col("group") == "FB").row(0, named=True)
    assert abs(s["prog_carry_p100"] - 8.0) < 1e-9 and s["main_player"] == 1 and s["n_units"] == 2
    assert abs(s["share_prog_carry_p100"] - (10 + 2) / (10 + 2 + 8)) < 1e-9


def test_persistence_separates_system_from_noise():
    s = relative_demand(slot_table(world(), TR, 900.0), TR)
    per = persistence(s, TR, np.random.default_rng(1), n_perm=50)
    assert per["n_player_changed"] == per["n_pairs"]
    for t in TR:
        x = per["traits"][t]
        assert x["r_same_team_new_player"] > 0.6 and abs(x["r_other_team"]) < 0.25


def test_no_persistence_when_demand_is_pure_player():
    s = relative_demand(slot_table(world(team_style_sd=0.01, player_sd=1.0), TR, 900.0), TR)
    per = persistence(s, TR, np.random.default_rng(1), n_perm=50)
    assert all(abs(x["r_same_team"]) < 0.3 for x in per["traits"].values())


def test_slot_card_orders_by_magnitude():
    c = slot_card({"rel_a": 2.0, "rel_b": -3.0, "rel_c": 0.5}, ("a", "b", "c"), top=2)
    assert c["asks_more"][0][0] == "a" and c["asks_less"][0][0] == "b"
