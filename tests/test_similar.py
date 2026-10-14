import numpy as np
import polars as pl

from rolefit.similar import find_units, merge_weights, plays_like


def test_merge_weights_averages_and_find_units_matches_names():
    w = merge_weights({"0": {"a": ["pos", 0.2]}, "1": {"a": ["sys", 0.4]}})
    assert w == {"a": ("pos", 0.30000000000000004)} or abs(w["a"][1] - 0.3) < 1e-9
    u = pl.DataFrame({"unit": ["1", "2"], "player": ["Sergio Busquets", "X"], "nickname": [None, "Busi"],
                      "minutes": [2000.0, 3000.0]})
    assert find_units(u, "busquets")["unit"].to_list() == ["1"]


def test_plays_like_skips_same_player_and_orders_by_distance():
    frame = pl.DataFrame({"unit": ["a1", "a2", "b", "c"], "player_id": [1, 1, 2, 3]})
    E = np.array([[0.0], [0.1], [1.0], [5.0]])
    r = plays_like(frame, E, "a1", k=2)
    assert r["unit"].to_list() == ["b", "c"]
