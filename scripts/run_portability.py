"""Stage 2: system adjustment, portability (V2) and re-identification (V3).

Reads stage 1's out/traits.parquet, out/traits_halves.parquet and
out/reliability.json (pulled from Drive by the workflow). Writes
out/portability.json, out/reid.json, out/traits_adjusted.parquet.

    python scripts/run_portability.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from rolefit.movers import CROSS_TEAM, mover_pairs  # noqa: E402
from rolefit.portability import (choose_weights, embed, merge_reid, player_folds,  # noqa: E402
                                 portability_table)
from rolefit.system import apply_offsets, system_adjust  # noqa: E402
from rolefit.validate import reid  # noqa: E402

OUT = Path(os.environ.get("ROLEFIT_OUT", "out"))
MIN_MIN_SPLIT = 900.0
MIN_MIN_CROSS = 270.0
METRIC = "euclidean"   # z-scored traits: magnitude matters, cosine would discard it
PAIR_GROUPS = {"same_team": ["same_team"], "club_country": ["club_country"],
               "transfer": ["transfer"], "cross_team": list(CROSS_TEAM)}


def fmt(r: dict) -> str:
    if not r.get("n"):
        return "n=0"
    s = (f"n={r['n']} top1={r['top1']:.3f} top5={r['top5']:.3f} "
         f"(90% CI {r['top5_ci90'][0]:.3f}-{r['top5_ci90'][1]:.3f}) | chance top5={r['chance_top5']:.4f}")
    if "position_baseline_top5" in r:
        s += f" | position-label top5={r['position_baseline_top5']:.4f}"
    return s


def split_half(halves: pl.DataFrame, choices: dict, fold_of: dict) -> dict:
    """Query = half A, gallery = every half B. Each query is embedded with
    the weights fitted on the other fold of players."""
    a = halves.filter(pl.col("half") == 0).sort("unit")
    b = halves.filter(pl.col("half") == 1).sort("unit")
    units = sorted(set(a["unit"].to_list()) & set(b["unit"].to_list()))
    a = a.filter(pl.col("unit").is_in(units)).sort("unit")
    b = b.filter(pl.col("unit").is_in(units)).sort("unit")
    both = pl.concat([a, b], how="diagonal_relaxed")
    groups = a["group"].fill_null("OTHER").to_list()
    pids = a["player_id"].to_list()
    n = len(units)
    parts = []
    for f in (0, 1):
        E = embed(both, choices[1 - f])
        q = [i for i in range(n) if fold_of[pids[i]] == f]
        parts.append(reid(E[:n][q], E[n:], [{i} for i in q],
                          groups_q=[groups[i] for i in q], groups_g=groups,
                          metric=METRIC))
    return merge_reid(parts)


def cross_team(units: pl.DataFrame, pairs: pl.DataFrame, choices: dict, fold_of: dict,
               types: tuple) -> dict:
    """Query = a unit, gallery = every unit. Right = the same player's units at
    a different team of the given pair types. Removed from the candidates:
    the query itself, the player's same-team units, and the player's units
    from other cross-team types (so each type is scored on its own)."""
    ulist = units["unit"].to_list()
    idx = {u: i for i, u in enumerate(ulist)}
    groups = units["group"].fill_null("OTHER").to_list()
    pids = units["player_id"].to_list()
    partners: dict = {}
    for a, b, t in pairs.select("unit_a", "unit_b", "type").iter_rows():
        if a in idx and b in idx:
            partners.setdefault(idx[a], []).append((idx[b], t))
            partners.setdefault(idx[b], []).append((idx[a], t))
    parts = []
    for f in (0, 1):
        E = embed(units, choices[1 - f])
        q, correct, exclude = [], [], []
        for i, plist in partners.items():
            if fold_of[pids[i]] != f:
                continue
            right = {j for j, t in plist if t in types}
            if not right:
                continue
            q.append(i)
            correct.append(right)
            exclude.append({i} | {j for j, t in plist if t not in types})
        parts.append(reid(E[q], E, correct, exclude, groups_q=[groups[i] for i in q], groups_g=groups,
                          metric=METRIC))
    return merge_reid(parts)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    rng = np.random.default_rng(0)
    rel = json.loads((OUT / "reliability.json").read_text(encoding="utf-8"))
    kept = rel["kept"]
    print(f"{len(kept)} traits kept by V1: {', '.join(kept)}")
    if not kept:
        sys.exit("no trait passed V1; nothing to test")

    units = pl.read_parquet(OUT / "traits.parquet").filter(pl.col("group").fill_null("OTHER") != "GK")
    halves = pl.read_parquet(OUT / "traits_halves.parquet")

    adj = system_adjust(units, kept)
    meta = units.select("unit", "player_id", "team_id", "key", "competition", "group", "minutes")
    full = meta.join(adj, on="unit")
    full.write_parquet(OUT / "traits_adjusted.parquet")

    pairs = mover_pairs(units, MIN_MIN_CROSS)
    counts = dict(pairs.group_by("type").len().iter_rows())
    print(f"mover pairs (both units >= {MIN_MIN_CROSS:.0f} min, outfield): {counts} [{time.time() - t0:.0f}s]")

    # V2 portability (descriptive)
    port = portability_table(full, pairs, kept, PAIR_GROUPS, rng)
    print("\n== V2 portability: r across the two units of a player (90% CI by player) ==")
    print(f"  {'trait':22s} {'version':4s} " + "  ".join(f"{g:>20s}" for g in PAIR_GROUPS))
    for t in kept:
        for v in ("pos", "sys"):
            cells = []
            for g in PAIR_GROUPS:
                x = port[t][v][g]
                cells.append(f"{'-' if x['r'] is None else format(x['r'], '.2f')} (n={x['n']})".rjust(20))
            print(f"  {t:22s} {v:4s} " + "  ".join(cells))

    # weights, cross-fitted by player
    fold_of = player_folds(units["player_id"].to_list())
    cross_pairs = pairs.filter(pl.col("type").is_in(list(CROSS_TEAM)))
    choices = {}
    for f in (0, 1):
        fp = cross_pairs.filter(pl.col("player_id").is_in([p for p, k in fold_of.items() if k == f]))
        choices[f] = choose_weights(full, fp, kept)
    for f in (0, 1):
        live = {t: (v, round(w, 3)) for t, (v, w) in choices[f].items() if w > 0}
        print(f"\nfold {f} weights (version, weight), {len(live)} traits with weight > 0: {live}")

    # V3 split-half
    hv = (apply_offsets(halves.select(pl.exclude("group", "position", "player_id")), adj, kept)
          .join(units.select("unit", "player_id", "group", pl.col("minutes").alias("full_minutes")), on="unit")
          .filter(pl.col("full_minutes") >= MIN_MIN_SPLIT))
    sh = split_half(hv, choices, fold_of)
    equal = {f: {t: ("pos", 1.0) for t in kept} for f in (0, 1)}
    sh_equal = split_half(hv, equal, fold_of)
    pass_sh = sh.get("n", 0) > 0 and sh["top5"] >= 0.50
    print(f"\n== V3 split-half re-ID (units >= {MIN_MIN_SPLIT:.0f} min) ==")
    print(f"  primary (portability-weighted): {fmt(sh)}   {'PASS' if pass_sh else 'FAIL'} (bar: top5 >= 0.50)")
    print(f"  descriptive (all kept traits, position-relative, equal weight): {fmt(sh_equal)}")

    # V3 cross-team
    cu = full.filter(pl.col("minutes") >= MIN_MIN_CROSS)
    ct = cross_team(cu, pairs, choices, fold_of, CROSS_TEAM)
    lo, base = (ct["top5_ci90"][0], ct["position_baseline_top5"]) if ct.get("n") else (0, 1)
    pass_ct = ct.get("n", 0) > 0 and ct["top5"] >= 2 * base and lo > base
    print(f"\n== V3 cross-team re-ID (units >= {MIN_MIN_CROSS:.0f} min, gallery {cu.height}) ==")
    print(f"  primary (all cross-team pairs): {fmt(ct)}   {'PASS' if pass_ct else 'FAIL'} "
          f"(bar: top5 >= 2x position-label and CI above it)")
    by_type = {t: cross_team(cu, pairs, choices, fold_of, (t,)) for t in CROSS_TEAM}
    for t, r in by_type.items():
        print(f"  {t:13s} {fmt(r)}")
    ct_equal = cross_team(cu, pairs, equal, fold_of, CROSS_TEAM)
    print(f"  descriptive (equal weights, position-relative): {fmt(ct_equal)}")

    (OUT / "portability.json").write_text(json.dumps(
        {"pairs": counts, "table": port,
         "weights": {str(f): {t: list(c) for t, c in choices[f].items()} for f in (0, 1)}}, indent=1),
        encoding="utf-8")
    (OUT / "reid.json").write_text(json.dumps(
        {"split_half": sh, "split_half_equal": sh_equal, "split_half_pass": bool(pass_sh),
         "cross_team": ct, "cross_team_pass": bool(pass_ct), "cross_team_by_type": by_type,
         "cross_team_equal": ct_equal}, indent=1), encoding="utf-8")
    print(f"\ndone in {time.time() - t0:.0f}s")
