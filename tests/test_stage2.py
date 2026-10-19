import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from rolefit.movers import mover_pairs
from rolefit.portability import choose_weights, merge_reid, pair_correlation
from rolefit.system import apply_offsets, system_adjust

TRAITS = ("t_player", "t_player2", "t_player3", "t_system", "t_noise")


def synthetic_world(seed=0, n_players=240, noise=0.3):
    """Players with a fixed personal effect; teams with a team-wide effect.
    t_player = player effect only; t_system = team effect only; t_noise = noise.
    Every player has 2-3 club seasons, a third also play for a national team."""
    rng = np.random.default_rng(seed)
    leagues = {"1_1": "La Liga", "2_1": "Premier League", "3_1": "FA Women's Super League"}
    team_eff = {}
    units, halves = [], []
    groups = ("CB", "CM", "ST")
    for p in range(n_players):
        pe, pe2, pe3 = rng.normal(size=3)
        g = groups[p % 3]
        gshift = {"CB": -2.0, "CM": 0.0, "ST": 2.0}[g]
        stints = []
        for key in rng.choice(list(leagues), size=int(rng.integers(2, 4)), replace=False):
            stints.append((key, leagues[key], int(rng.integers(0, 8)) + 100 * int(key[0])))
        if p % 3 == 0:
            stints.append(("9_1", "FIFA World Cup", 900 + p % 12))
        for key, comp, team in stints:
            te = team_eff.setdefault((key, team), rng.normal(scale=1.5))
            unit = f"{p}|{team}|{key}"
            mins = 1800.0 if comp != "FIFA World Cup" else 450.0
            base = {"t_player": gshift + pe, "t_player2": pe2, "t_player3": pe3,
                    "t_system": gshift + te, "t_noise": gshift}
            row = {t: v + rng.normal(scale=noise) for t, v in base.items()}
            units.append({"unit": unit, "player_id": p, "team_id": team, "key": key, "competition": comp,
                          "group": g, "position": g, "minutes": mins, **row})
            for h in (0, 1):
                halves.append({"unit": unit, "half": h, "minutes": mins / 2, "group": g, "position": g,
                               **{t: v + rng.normal(scale=noise) for t, v in base.items()}})
    return pl.DataFrame(units), pl.DataFrame(halves)


def test_system_adjust_removes_team_effect_but_keeps_player_effect():
    units, _ = synthetic_world()
    adj = units.select("unit", "player_id").join(system_adjust(units, TRAITS), on="unit")
    pairs = mover_pairs(units, 270)
    cross = pairs.filter(pl.col("type") != "same_team")

    def r(col):
        return pair_correlation(dict(zip(adj["unit"].to_list(), adj[col].to_list())), cross)["r"]

    assert r("t_player__pos") > 0.6 and r("t_player__sys") > 0.6     # a player's own trait travels
    assert abs(r("t_system__sys")) < 0.2                              # a team trait does not
    team_col = system_adjust(units, ("t_system",))["t_system__team"].to_numpy()
    assert np.corrcoef(team_col, (units["t_system"] - units["t_system"].mean()).to_numpy())[0, 1] > 0.3


def test_apply_offsets_matches_full_season_versions():
    units, _ = synthetic_world(n_players=60)
    adj = system_adjust(units, TRAITS)
    again = apply_offsets(units.select("unit", *TRAITS), adj, TRAITS).sort("unit")
    ref = adj.sort("unit")
    for t in TRAITS:
        assert np.allclose(again[f"{t}__sys"].to_numpy(), ref[f"{t}__sys"].to_numpy(), equal_nan=True)


def test_mover_pair_types():
    units = pl.DataFrame({
        "unit": ["a", "b", "c", "d", "e"], "player_id": [1, 1, 1, 1, 2], "team_id": [10, 10, 20, 99, 10],
        "competition": ["La Liga", "La Liga", "Premier League", "FIFA World Cup", "La Liga"],
        "minutes": [900.0, 900.0, 900.0, 300.0, 900.0]})
    pairs = mover_pairs(units, 270)
    types = sorted(pairs["type"].to_list())
    assert types == ["club_country", "club_country", "club_country", "same_team", "transfer", "transfer"]
    assert pairs.filter(pl.col("player_id") == 2).height == 0


def test_choose_weights_prefers_portable_traits():
    units, _ = synthetic_world()
    adj = units.select("unit").join(system_adjust(units, TRAITS), on="unit")
    cross = mover_pairs(units, 270).filter(pl.col("type") != "same_team")
    w = choose_weights(adj, cross, TRAITS)
    assert w["t_player"][1] > 0.6
    assert w["t_noise"][1] < 0.2


def test_merge_reid_pools_counts():
    a = {"n": 10, "top1": 0.5, "top5": 0.8, "chance_top1": 0.01, "chance_top5": 0.05}
    b = {"n": 30, "top1": 0.1, "top5": 0.4, "chance_top1": 0.01, "chance_top5": 0.05}
    m = merge_reid([a, b])
    assert m["n"] == 40 and abs(m["top1"] - 0.2) < 1e-9 and abs(m["top5"] - 0.5) < 1e-9


def test_run_portability_end_to_end(tmp_path):
    units, halves = synthetic_world()
    out = tmp_path / "out"
    out.mkdir()
    units.write_parquet(out / "traits.parquet")
    halves.write_parquet(out / "traits_halves.parquet")
    (out / "reliability.json").write_text(json.dumps({"kept": list(TRAITS)}), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, POLARS_MAX_THREADS="2", PYTHONPATH=os.pathsep.join([str(root / "src"), os.environ.get("PYTHONPATH", "")]))
    res = subprocess.run([sys.executable, str(root / "scripts" / "run_portability.py")], cwd=tmp_path,
                         env=env, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, res.stderr[-3000:]
    reid = json.loads((out / "reid.json").read_text(encoding="utf-8"))
    # three personal traits are a strong signature, so the same player is easy to find
    assert reid["cross_team"]["n"] > 50 and reid["cross_team"]["top5"] > 3 * reid["cross_team"]["position_baseline_top5"]
    assert reid["split_half"]["top5"] > 5 * reid["split_half"]["position_baseline_top5"]
    assert (out / "traits_adjusted.parquet").exists()
    res2 = subprocess.run([sys.executable, str(root / "scripts" / "run_space.py")], cwd=tmp_path,
                          env=env, capture_output=True, text=True, encoding="utf-8")
    assert res2.returncode == 0, res2.stderr[-3000:]
    space = json.loads((out / "space.json").read_text(encoding="utf-8"))
    assert space["chosen"] in space["results"] and len(space["results"]) == 3
