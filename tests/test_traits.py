import json

import numpy as np
import polars as pl

from rolefit.traits import (EVENT_COLS, add_derived, aggregate, on_pitch_possessions, spells,
                            split_half_reliability)


def _ev(**kw):
    base = {c: None for c in EVENT_COLS}
    base.update({"idx": 0, "period_seconds": 0.0})
    base.update({"match_id": 1, "player_id": 7, "team_id": 1, "possession_team_id": 1, "period": 1,
                 "minute": 0, "second": 0, "under_pressure": False, "counterpress": False,
                 "competition_id": 1, "season_id": 1, "f_cross": False, "f_switch": False,
                 "f_through": False, "f_assist": False, "f_aerial_won": False, "f_cutback": False,
                 "f_backheel": False, "extra_card": False, "value": None, "xp": None,
                 "one_touch": False, "hold_s": None, "recv_space_e": None, "recv_crowd_e": None})
    base.update(kw)
    return base


def _frame(rows):
    return pl.DataFrame(rows, infer_schema_length=None)


def test_spells_and_on_pitch_possessions_respect_halves():
    lu = pl.DataFrame({"match_id": [1, 1], "team_id": [1, 1], "player_id": [7, 8], "positions": [
        json.dumps([{"from": "00:00", "to": "60:00", "from_period": 1, "to_period": 2}]),
        json.dumps([{"from": "60:00", "to": None, "from_period": 2, "to_period": None}])]})
    sp = spells(lu)
    assert sp.filter(pl.col("player_id") == 7)["minutes"][0] == 60
    assert sp.filter(pl.col("player_id") == 8)["minutes"][0] == 30
    # possessions: 1st half 46' (stoppage), 2nd half 50', 2nd half 70' (opponent)
    ev = _frame([_ev(possession=1, period=1, minute=46),
                 _ev(possession=2, period=2, minute=50),
                 _ev(possession=3, period=2, minute=70, possession_team_id=2)])
    onp = on_pitch_possessions(ev, sp).sort("player_id")
    assert onp["own_poss"].to_list() == [2, 0] and onp["opp_poss"].to_list() == [0, 1]


def test_rates_are_per_on_pitch_possession_and_shares_ignore_volume():
    rows = []
    for i in range(10):     # 10 open passes, 8 completed, 4 progressive (and long)
        rows.append(_ev(type="Pass", x=50.0, y=40.0,
                        end_x=95.0 if i < 2 else (85.0 if i < 4 else 55.0), end_y=40.0,
                        outcome=None if i < 8 else "Incomplete", xp=0.7))
    ev = add_derived(_frame(rows)).with_columns(unit=pl.lit("u"))
    onp = pl.DataFrame({"unit": ["u"], "match_id": [1], "own_poss": [20], "opp_poss": [20], "minutes": [90.0]})
    t = aggregate(ev, onp, ["unit"]).row(0, named=True)
    assert abs(t["pass_cmp"] - 0.8) < 1e-9
    assert abs(t["pass_cmp_oe"] - 0.1) < 1e-9
    assert abs(t["prog_pass_p100"] - 100 * 4 / 20) < 1e-9
    assert abs(t["long_share"] - 0.4) < 1e-9
    assert abs(t["pass_fwd"] - 1.0) < 1e-9


def test_within_group_reliability_ignores_position_differences():
    rng = np.random.default_rng(0)
    rows = []
    for u in range(200):
        g = "CB" if u < 100 else "ST"
        shift = 0.0 if g == "CB" else 10.0      # big between-group gap, no within-group signal
        for h in (0, 1):
            rows.append({"unit": f"u{u}", "group": g, "half": h, "avg_x": shift + rng.normal()})
    rel = split_half_reliability(pl.DataFrame(rows), "unit", "group", traits=("avg_x",))
    assert rel["avg_x"]["r"] > 0.9 and abs(rel["avg_x"]["r_within"]) < 0.2


def test_playstyle_traits_on_hand_traceable_events():
    rows = [
        _ev(type="Ball Receipt*", x=20.0, y=40.0, possession=1),     # own third, central
        _ev(type="Ball Receipt*", x=90.0, y=5.0, possession=1),      # attacking third, wide
        _ev(type="Pass", x=90.0, y=5.0, end_x=108.0, end_y=40.0, possession=1, body_part="Left Foot"),  # into box
        _ev(type="Pass", x=60.0, y=40.0, end_x=60.0, end_y=60.0, possession=2, body_part="Right Foot"),  # sideways
        _ev(type="Carry", x=70.0, y=40.0, end_x=85.0, end_y=40.0, possession=2, duration=3.0),
        _ev(type="Pass", x=5.0, y=40.0, end_x=30.0, end_y=40.0, possession=3, sub_type="Corner"),
        _ev(type="Pressure", x=85.0, y=30.0, possession_team_id=2, team_id=1),
        _ev(type="Dribbled Past", x=30.0, y=30.0, possession_team_id=2, team_id=1),
        _ev(type="Shot", x=105.0, y=40.0, possession=1, xg=0.3, outcome="Goal", body_part="Head"),
    ]
    ev = add_derived(_frame(rows)).with_columns(unit=pl.lit("u"))
    onp = pl.DataFrame({"unit": ["u"], "match_id": [1], "own_poss": [10], "opp_poss": [10], "minutes": [90.0]})
    t = aggregate(ev, onp, ["unit"]).row(0, named=True)
    assert t["recv_def_share"] == 0.5 and t["recv_att_share"] == 0.5 and t["recv_wide_share"] == 0.5
    assert abs(t["dst_box_share"] - 1 / 2) < 1e-9 and t["pass_sideways_share"] == 1.0
    assert t["left_foot_share"] == 0.5
    assert abs(t["carry_att_entry_p100"] - 10.0) < 1e-9 and abs(t["carry_speed"] - 5.0) < 1e-9
    assert abs(t["corner_p100"] - 10.0) < 1e-9
    assert t["press_att_share"] == 1.0 and abs(t["dribbled_past_p100o"] - 10.0) < 1e-9
    assert t["goals_p100"] == 10.0 and abs(t["finishing_oe"] - 0.7) < 1e-9 and t["header_shot_share"] == 1.0
    # possessions 1, 2, 3 are touched; only possession 1 produced xG (0.3), and the player took that shot
    assert abs(t["poss_touch_share"] - 0.3) < 1e-9 and abs(t["xgchain_p100"] - 3.0) < 1e-9
    assert t["xgbuildup_p100"] == 0.0


def test_one_touch_and_hold_time_from_the_event_stream():
    from rolefit.traits import tempo_columns
    rows = [
        dict(event_id="a", idx=1, player_id=7, type="Ball Receipt*", period_seconds=10.0),
        dict(event_id="b", idx=2, player_id=7, type="Pass", period_seconds=10.5),          # first-time
        dict(event_id="o", idx=3, player_id=9, team_id=2, type="Pressure", period_seconds=11.0),  # opponent, ignored
        dict(event_id="c", idx=4, player_id=8, type="Ball Receipt*", period_seconds=12.0),
        dict(event_id="d", idx=5, player_id=8, type="Carry", period_seconds=12.1),
        dict(event_id="e", idx=6, player_id=8, type="Pass", period_seconds=15.0),          # after a carry, held 3s
    ]
    ev = pl.DataFrame([_ev(possession=1, **r) for r in rows], infer_schema_length=None)
    t = tempo_columns(ev).sort("idx")
    oneT = dict(zip(t["event_id"].to_list(), t["one_touch"].to_list()))
    hold = dict(zip(t["event_id"].to_list(), t["hold_s"].to_list()))
    assert oneT["b"] and not oneT["e"]
    assert abs(hold["b"] - 0.5) < 1e-9 and abs(hold["e"] - 3.0) < 1e-9


def test_receipt_space_counts_nearby_opponents():
    from rolefit.traits import receipt_space
    ev = pl.DataFrame([_ev(event_id="r", type="Ball Receipt*", x=50.0, y=40.0)], infer_schema_length=None)
    fr = pl.DataFrame({"event_id": ["r"] * 4, "teammate": [False, False, False, True],
                       "x": [53.0, 50.0, 70.0, 50.5], "y": [40.0, 44.0, 40.0, 40.0]})
    out = receipt_space(ev.drop("recv_space_e", "recv_crowd_e"), fr).row(0, named=True)
    assert out["recv_space_e"] == 3.0 and out["recv_crowd_e"] == 2.0


def test_quick_release_and_short_carries():
    rows = [_ev(type="Pass", x=50.0, y=40.0, end_x=60.0, end_y=40.0, hold_s=1.0),
            _ev(type="Pass", x=50.0, y=40.0, end_x=60.0, end_y=40.0, hold_s=4.0),
            _ev(type="Carry", x=50.0, y=40.0, end_x=51.0, end_y=41.0),
            _ev(type="Carry", x=50.0, y=40.0, end_x=60.0, end_y=40.0)]
    ev = add_derived(_frame(rows)).with_columns(unit=pl.lit("u"))
    onp = pl.DataFrame({"unit": ["u"], "match_id": [1], "own_poss": [10], "opp_poss": [10], "minutes": [90.0]})
    t = aggregate(ev, onp, ["unit"]).row(0, named=True)
    assert t["quick_release_share"] == 0.5 and t["short_carry_share"] == 0.5
