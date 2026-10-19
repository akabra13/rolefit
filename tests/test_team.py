import polars as pl

from rolefit.needs import peer_z, slot_for_need, weaknesses
from rolefit.team import METRICS, team_match_counts, team_season_metrics


def _ev(**kw):
    base = dict(match_id=1, team_id=1, possession=1, possession_team_id=1, period=1, type="Pass", sub_type=None,
                outcome=None, x=50.0, y=40.0, end_x=60.0, end_y=40.0, xg=None, play_pattern="Regular Play", extra="{}")
    base.update(kw)
    return base


def test_team_counts_for_and_against():
    rows = [
        _ev(type="Pass", x=90.0, end_x=105.0, end_y=40.0),                                 # box entry, team 1
        _ev(type="Shot", x=105.0, xg=0.3, end_x=None, end_y=None),
        _ev(type="Pass", x=20.0, outcome="Incomplete", end_x=30.0),                       # lost in own third
        _ev(team_id=2, possession=2, possession_team_id=2, type="Shot", x=110.0, xg=0.5,
            play_pattern="From Counter", end_x=None, end_y=None),
    ]
    ev = pl.DataFrame(rows, infer_schema_length=None)
    m = pl.DataFrame({"match_id": [1], "home_team_id": [1], "away_team_id": [2], "home_score": [1], "away_score": [0]})
    tm = team_match_counts(ev, m).with_columns(key=pl.lit("k"))
    t1 = team_season_metrics(tm).filter(pl.col("team_id") == 1).row(0, named=True)
    assert t1["xg_for_pp"] == 0.3 and t1["xg_against_pp"] == 0.5 and t1["counter_xg_against_pp"] == 0.5
    assert t1["box_entries_pp"] == 1.0 and t1["own_third_loss_pp"] == 1.0 and t1["points_pm"] == 3.0


def test_weaknesses_and_slot():
    rows = []
    for i in range(10):
        r = {"key": "k", "team_id": i, "matches": 20}
        for m in METRICS:
            r[m] = float(i)
        rows.append(r)
    ts = peer_z(pl.DataFrame(rows))
    worst = ts.filter(pl.col("team_id") == 0).row(0, named=True)
    w = weaknesses(worst, None, top=20)
    assert w and all(x["z_good"] < 0 for x in w)
    assert {x["metric"] for x in w}.isdisjoint({"own_third_loss_pp", "xg_against_pp"})   # low is good there
    slots = pl.DataFrame({"group": ["CB", "DM"], "rel_prog_pass_p100": [-1.5, 0.2], "rel_prog_carry_p100": [-0.5, 0.1]})
    s = slot_for_need("prog_pp", slots)
    assert s["group"] == "CB" and s["shortfall"] == -1.0
