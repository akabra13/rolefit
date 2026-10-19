import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "busquets"))

from metrics import COLS, defensive_counts, frame_metrics, touch_sequences  # noqa: E402


def ev(rows):
    base = {c: None for c in COLS}
    out = []
    for i, r in enumerate(rows):
        d = dict(base, match_id=1, idx=i, period=1, possession=1, possession_team_id=1, team_id=1,
                 under_pressure=False, event_id=f"e{i}", period_seconds=float(i))
        d.update(r)
        out.append(d)
    return pl.DataFrame(out, infer_schema_length=None)


def test_turn_bounce_redirect():
    rows = [
        # CB (2) plays forward into DM (5) moving rightward; DM turns and plays forward back to the left
        dict(player_id=2, type="Pass", x=30.0, y=20.0, end_x=45.0, end_y=30.0, recipient_id=5),
        dict(player_id=5, type="Ball Receipt*", x=45.0, y=30.0),
        dict(player_id=5, type="Carry", x=45.0, y=30.0, end_x=46.0, end_y=31.0),
        dict(player_id=5, type="Pass", x=46.0, y=31.0, end_x=60.0, end_y=10.0, recipient_id=7),
        # 7 receives and bounces straight back to 5
        dict(player_id=7, type="Ball Receipt*", x=60.0, y=10.0),
        dict(player_id=7, type="Pass", x=60.0, y=10.0, end_x=50.0, end_y=20.0, recipient_id=5),
    ]
    s = touch_sequences(ev(rows))
    dm = s.filter(pl.col("player_id") == 5).row(0, named=True)
    assert dm["rp"] and dm["rp_fwd"] and dm["rb"] and dm["rb_fwd"] and dm["redirect"] and not dm["bounce"]
    seven = s.filter(pl.col("player_id") == 7).row(0, named=True)
    assert seven["rp"] and seven["bounce"] and not seven["rp_fwd"]


def test_counter_foul_and_midfield_interception():
    rows = [
        dict(player_id=9, team_id=2, possession_team_id=2, possession=2, type="Pass", x=50.0, y=40.0, period_seconds=100.0),
        dict(player_id=5, team_id=1, possession_team_id=2, possession=2, type="Foul Committed", x=55.0, y=40.0, period_seconds=104.0),
        dict(player_id=5, team_id=1, possession_team_id=2, possession=3, type="Interception", x=50.0, y=40.0, period_seconds=200.0),
    ]
    d = defensive_counts(ev(rows)).filter(pl.col("player_id") == 5).row(0, named=True)
    assert d["counter_foul"] == 1 and d["mid_intercept"] == 1


def test_frame_metrics_bypass_and_space():
    rows = [dict(player_id=5, type="Pass", x=40.0, y=40.0, end_x=70.0, end_y=40.0, recipient_id=7, event_id="p"),
            dict(player_id=7, type="Ball Receipt*", x=70.0, y=40.0, event_id="r")]
    e = ev(rows)
    seqs = touch_sequences(e)
    fr = pl.DataFrame({"event_id": ["p", "p", "p", "r", "r"], "teammate": [False, False, False, False, True],
                       "x": [50.0, 60.0, 75.0, 60.0, 72.0], "y": [40.0, 45.0, 40.0, 40.0, 40.0]})
    m = frame_metrics(seqs, e, fr)
    p5 = m.filter(pl.col("player_id") == 5).row(0, named=True)
    assert p5["bypassed_sum"] == 2 and p5["line_breaks"] == 1 and abs(p5["space_sum"] - 5.0) < 1e-9
    r7 = m.filter(pl.col("player_id") == 7).row(0, named=True)
    assert r7["behind_sum"] == 1 and r7["crowd_sum"] == 0
