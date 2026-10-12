"""Pre-registered re-identification tests for role vectors (see README).

    python scripts/run_reid.py [--keys 37_4 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from pitchgraph.data.lake import Lake  # noqa: E402

from rolefit.roles import UNIT, embed, minutes_by_unit, primary_group, role_vectors  # noqa: E402
from rolefit.validate import reid  # noqa: E402

N_COMPONENTS = 20
MIN_MIN_SPLIT = 900.0
MIN_MIN_CROSS = 270.0
COLS = ["match_id", "match_date", "player_id", "team_id", "competition_id", "season_id",
        "type", "position", "x", "y", "end_x", "end_y"]


def unit_expr() -> pl.Expr:
    return pl.format("{}|{}|{}|{}", *[pl.col(c) for c in UNIT])


def build(lake: Lake, keys: list[str] | None):
    matches = lake.matches().with_columns(
        key=pl.col("competition_id").cast(pl.Utf8) + "_" + pl.col("season_id").cast(pl.Utf8))
    if keys:
        matches = matches.filter(pl.col("key").is_in(keys))
    lineups = lake.scan("lineups").collect()
    mins = minutes_by_unit(lineups, matches).with_columns(unit=unit_expr())
    mins_of = dict(zip(mins["unit"].to_list(), mins["minutes"].to_list()))
    seasons = matches.group_by("key", "competition_id", "season_id", "competition", "season").agg(
        pl.len().alias("n")).sort("key")

    full, halfA, halfB, group, names = {}, {}, {}, {}, {}
    t0 = time.time()
    for r in seasons.iter_rows(named=True):
        ev = (lake.events()
              .filter((pl.col("competition_id") == r["competition_id"]) & (pl.col("season_id") == r["season_id"])
                      & pl.col("player_id").is_not_null())
              .select(COLS).collect()
              .with_columns(unit=unit_expr()))
        if ev.is_empty():
            continue
        order = (ev.select("unit", "match_id", "match_date").unique()
                   .sort("unit", "match_date", "match_id")
                   .with_columns(half=pl.int_range(pl.len()).over("unit") % 2))
        ev = ev.join(order.select("unit", "match_id", "half"), on=["unit", "match_id"])
        g = primary_group(ev)
        nm = dict(zip(ev["unit"].to_list(), ev["player_id"].to_list()))
        for tag, sub, store in (("full", ev, full),
                                ("A", ev.filter(pl.col("half") == 0).with_columns(unit=pl.col("unit")), halfA),
                                ("B", ev.filter(pl.col("half") == 1), halfB)):
            units, X, offsets = role_vectors(sub)
            for u, row in zip(units, X):
                store[u] = row
        group.update(g)
        names.update(nm)
        print(f"{r['competition']} {r['season']}: {ev['unit'].n_unique()} units [{time.time() - t0:.0f}s]", flush=True)
    return full, halfA, halfB, group, mins_of, offsets


def split_half(full, halfA, halfB, group, mins_of, offsets):
    units = sorted(u for u in full if u in halfA and u in halfB and mins_of.get(u, 0) >= MIN_MIN_SPLIT)
    XA = np.array([halfA[u] for u in units])
    XB = np.array([halfB[u] for u in units])
    # unsupervised standardisation + PCA fit on both halves together
    Z, P = embed(np.vstack([XA, XB]), offsets, N_COMPONENTS)
    n = len(units)
    g = [group[u] for u in units]
    out = {"n_units": n}
    for name, M in (("pca20", P), ("standardised", Z)):
        res = reid(M[:n], M[n:], [{i} for i in range(n)], groups_q=g, groups_g=g)
        out[name] = res
    return out


def cross_team(full, group, mins_of, offsets):
    units = sorted(u for u in full if mins_of.get(u, 0) >= MIN_MIN_CROSS)
    X = np.array([full[u] for u in units])
    _, P = embed(X, offsets, N_COMPONENTS)
    pid = [u.split("|")[0] for u in units]
    tid = [u.split("|")[1] for u in units]
    by_player: dict = {}
    for i, p in enumerate(pid):
        by_player.setdefault(p, []).append(i)
    correct, exclude, qidx = [], [], []
    for i, p in enumerate(pid):
        same_player = by_player[p]
        diff_team = {j for j in same_player if tid[j] != tid[i]}
        if not diff_team:
            continue
        qidx.append(i)
        correct.append(diff_team)
        exclude.append({j for j in same_player if tid[j] == tid[i]})   # self and same-team seasons
    g = [group[u] for u in units]
    res = reid(P[qidx], P, correct, exclude, groups_q=[g[i] for i in qidx], groups_g=g)
    res["n_gallery"] = len(units)
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keys", nargs="*")
    args = ap.parse_args()

    lake = Lake.open()
    full, halfA, halfB, group, mins_of, offsets = build(lake, args.keys)
    print(f"\n{len(full)} units total")

    sh = split_half(full, halfA, halfB, group, mins_of, offsets)
    print(f"\n== split-half re-identification ({sh['n_units']} units with >= {MIN_MIN_SPLIT:.0f} min) ==")
    for name in ("pca20", "standardised"):
        r = sh[name]
        print(f"  [{name}] top1={r['top1']:.3f} (90% CI {r['top1_ci90'][0]:.3f}-{r['top1_ci90'][1]:.3f}) "
              f"top5={r['top5']:.3f} ({r['top5_ci90'][0]:.3f}-{r['top5_ci90'][1]:.3f}) | "
              f"gallery chance top1={r['chance_top1']:.4f} top5={r['chance_top5']:.4f} | "
              f"position-label baseline top1={r['position_baseline_top1']:.4f} top5={r['position_baseline_top5']:.4f}")

    ct = cross_team(full, group, mins_of, offsets)
    print(f"\n== cross-team re-identification ({ct['n']} queries, gallery {ct['n_gallery']}, >= {MIN_MIN_CROSS:.0f} min) ==")
    print(f"  top1={ct['top1']:.3f} ({ct['top1_ci90'][0]:.3f}-{ct['top1_ci90'][1]:.3f}) "
          f"top5={ct['top5']:.3f} ({ct['top5_ci90'][0]:.3f}-{ct['top5_ci90'][1]:.3f}) | "
          f"gallery chance top1={ct['chance_top1']:.4f} top5={ct['chance_top5']:.4f} | "
          f"position-label baseline top1={ct['position_baseline_top1']:.4f} top5={ct['position_baseline_top5']:.4f}")

    Path("reid_results.json").write_text(json.dumps({"split_half": sh, "cross_team": ct}, indent=1), encoding="utf-8")
