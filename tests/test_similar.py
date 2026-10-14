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


def test_absolute_space_keeps_position_and_relative_space_drops_it():
    from rolefit.similar import add_absolute, build_space
    rows = []
    for u, g, lpm, pos in (("a", "DM", 10.0, 1.0), ("b", "DM", 10.0, -1.0), ("c", "CB", 2.0, 1.0), ("d", "CB", 2.0, -1.0)):
        rows.append({"unit": u, "key": "k", "group": g, "minutes": 2000.0, "t__lpm": lpm, "t__pos": pos, "t__team": 0.0})
    adj = pl.DataFrame(rows)
    w = {"0": {"t": ["pos", 1.0]}}
    f, E = build_space(adj, w, absolute=True)
    ids = f["unit"].to_list()
    d = lambda i, j: abs(E[ids.index(i), 0] - E[ids.index(j), 0])  # noqa: E731
    assert d("a", "b") < d("a", "c")                  # absolute: DMs sit with DMs
    f2, E2 = build_space(adj, w, absolute=False)
    ids2 = f2["unit"].to_list()
    assert abs(E2[ids2.index("a"), 0] - E2[ids2.index("c"), 0]) < 1e-9   # relative: DM 'a' == CB 'c'
