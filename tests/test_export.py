import numpy as np
import polars as pl

from rolefit.export import build_payload
from tests.test_fit import TR, prep


def test_payload_shapes_and_leave_one_out_demand():
    adj, loo, pairs = prep()
    units = adj.select("unit", "player_id", "team_id", "key", "group", "minutes").with_columns(
        player=pl.col("player_id").cast(pl.Utf8), nickname=pl.lit(None, pl.Utf8), competition=pl.lit("La Liga"),
        season=pl.lit("2015/2016"), gender=pl.lit("male"), position=pl.lit("Center Midfield"))
    weights = {"0": {t: ["pos", 0.5] for t in TR}, "1": {t: ["pos", 0.5] for t in TR}}
    pay = build_payload(units, adj, weights, list(TR), {1: "Team"}, min_minutes=900.0)
    p = pay["players"][0]
    assert len(p["x"]) == len(TR) == len(p["d"]) and len(p["e"]) == pay["n_embed_dims"]
    assert set(pay["translation"]["coef"]) == set(TR) and len(pay["translation"]["coef"]["t_player"]) == 4
    assert pay["translation"]["coef"]["t_player"][0] > 0.5
