"""League strength and player level (V7, see README).

Reads out/traits.parquet (stage 1, per gender). Writes out/level.parquet and out/level.json.

    ROLEFIT_OUT=out/male python scripts/run_level.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from rolefit.level import crossfit_v7, fit_two_way, linked_players, quality_index  # noqa: E402

OUT = Path(os.environ.get("ROLEFIT_OUT", "out"))
MIN_MINUTES = 450.0
MIN_LINKED = 10

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    units = pl.read_parquet(OUT / "traits.parquet").filter(
        (pl.col("group").fill_null("OTHER") != "GK") & (pl.col("minutes") >= MIN_MINUTES))
    q = quality_index(units)
    df = units.select("unit", "player_id", "key", "competition", "season", "group", "minutes").join(q, on="unit") \
              .filter(pl.col("q").is_not_null() & pl.col("q").is_finite())
    players, keys = df["player_id"].to_list(), df["key"].to_list()
    qv, w = df["q"].to_numpy(), (df["minutes"] / 900.0).to_numpy()
    print(f"{df.height} units >= {MIN_MINUTES:.0f} min from {len(set(players))} players in {len(set(keys))} competition-seasons")

    v7 = crossfit_v7(players, keys, qv, w)
    print(f"\n== V7 held-out cross-competition prediction of the quality index ==\n  {v7}")
    passed = bool(v7.get("n_pairs", 0) > 0 and v7["ci90"][0] > 0)
    print(f"V7 {'PASS' if passed else 'FAIL'}: league adjustment lowers held-out error, 90% CI above 0")

    theta, lam = fit_two_way(None, players, keys, qv, w)
    links = linked_players(players, keys)
    meta = df.group_by("key").agg(pl.col("competition").first(), pl.col("season").first(), pl.len().alias("units"))
    table = []
    for r in meta.iter_rows(named=True):
        k = r["key"]
        known = links.get(k, 0) >= MIN_LINKED
        table.append({"key": k, "competition": r["competition"], "season": r["season"], "units": r["units"],
                      "linked_players": links.get(k, 0), "strength": round(-lam[k], 3) if known else None})
    table.sort(key=lambda x: (x["strength"] is None, -(x["strength"] or 0)))
    print(f"\n== competition strength (higher = harder; needs >= {MIN_LINKED} players also seen elsewhere) ==")
    for t in table:
        s = "unknown" if t["strength"] is None else f"{t['strength']:+.2f}"
        print(f"  {s:>8s}  {t['competition'][:28]:28s} {t['season']:10s} linked={t['linked_players']:4d} units={t['units']}")

    known = {t["key"] for t in table if t["strength"] is not None}
    lvl = df.with_columns(
        level=pl.col("q") - pl.col("key").replace_strict({k: (lam[k] if k in known else 0.0) for k in lam}, default=0.0),
        strength_known=pl.col("key").is_in(list(known)))
    lvl = lvl.with_columns(level_pct=(pl.col("level").rank("average").over("group") / pl.len().over("group") * 100).round(0))
    lvl.select("unit", "q", "level", "level_pct", "strength_known").write_parquet(OUT / "level.parquet")
    (OUT / "level.json").write_text(json.dumps({"v7": v7, "v7_pass": passed, "strength": table}, indent=1),
                                    encoding="utf-8")
    top = lvl.sort("level", descending=True).head(15).join(units.select("unit", "player"), on="unit")
    print("\nhighest level units: " + "; ".join(f"{r['player']} ({r['competition']} {r['season']})" for r in top.iter_rows(named=True)))
