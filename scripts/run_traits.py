"""Stage 1: player traits for every player-team-season in the lake, plus the
pre-registered split-half reliability test (V1, see README).

Writes out/traits.parquet, out/traits_halves.parquet, out/reliability.json.
The workflow copies out/ to Drive; nothing is kept on a laptop.

    python scripts/run_traits.py [--keys 37_4 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl  # noqa: E402

from pitchgraph.data.lake import Lake  # noqa: E402
from pitchgraph.library.build import fit_value_model  # noqa: E402

from rolefit.roles import POSITION_GROUPS  # noqa: E402
from rolefit.traits import (TRAITS, add_derived, aggregate, event_flags, on_pitch_possessions,  # noqa: E402
                            spells, split_half_reliability)
from rolefit.xpass import crossfit_xpass, pass_features  # noqa: E402

MIN_MINUTES = 900.0
KEEP_R = 0.5
OUT = Path("out")


def key_expr() -> pl.Expr:
    return pl.col("competition_id").cast(pl.Utf8) + "_" + pl.col("season_id").cast(pl.Utf8)


def unit_expr() -> pl.Expr:
    return pl.format("{}|{}|{}|{}", "player_id", "team_id", "competition_id", "season_id")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keys", nargs="*")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    lake = Lake.open()
    matches = lake.matches().with_columns(key=key_expr())
    if args.keys:
        matches = matches.filter(pl.col("key").is_in(args.keys))
    mids = matches["match_id"].to_list()
    sp_all = spells(lake.scan("lineups").filter(pl.col("match_id").is_in(mids)).collect())
    names = (lake.scan("lineups").select("player_id", "player", "nickname").unique("player_id").collect())
    print(f"{len(mids)} matches, {sp_all.height} lineup spells [{time.time() - t0:.0f}s]", flush=True)

    model = fit_value_model(lake)
    print(f"value model fitted [{time.time() - t0:.0f}s]", flush=True)

    passes = (pass_features(
        lake.events().filter((pl.col("type") == "Pass") & pl.col("x").is_not_null()
                             & pl.col("end_x").is_not_null())
        .select("event_id", "x", "y", "end_x", "end_y", "height", "body_part", "under_pressure",
                "sub_type", "play_pattern", "outcome", "competition_id", "season_id"))
        .with_columns(key=key_expr()).collect())
    xp = crossfit_xpass(passes)
    del passes
    print(f"xpass done [{time.time() - t0:.0f}s]", flush=True)

    full_parts, half_parts = [], []
    seasons = matches.select("key", "competition_id", "season_id", "competition", "season", "gender").unique().sort("key")
    for r in seasons.iter_rows(named=True):
        ids = matches.filter(pl.col("key") == r["key"]).select("match_id", "match_date")
        raw = event_flags(lake.events().filter(pl.col("match_id").is_in(ids["match_id"].to_list()))).collect()
        vals = model.add_action_values(raw.lazy()).select("event_id", "value").collect()
        ev = add_derived(raw.join(vals, on="event_id", how="left").join(xp, on="event_id", how="left"))

        meta = pl.DataFrame({"competition_id": [r["competition_id"]], "season_id": [r["season_id"]]})
        onp = (on_pitch_possessions(raw, sp_all.filter(pl.col("match_id").is_in(ids["match_id"].to_list())))
               .join(meta, how="cross").with_columns(unit=unit_expr())
               .join(ids, on="match_id"))
        onp = onp.sort("unit", "match_date", "match_id").with_columns(
            half=(pl.int_range(pl.len()).over("unit") % 2).cast(pl.Int8))
        ev = (ev.filter(pl.col("player_id").is_not_null()).with_columns(unit=unit_expr())
                .join(onp.select("unit", "match_id", "half"), on=["unit", "match_id"], how="inner"))

        pos = (ev.filter(pl.col("position").is_not_null()).group_by("unit", "position").len()
                 .sort("len", descending=True).group_by("unit", maintain_order=True).first()
                 .select("unit", group=pl.col("position").replace_strict(POSITION_GROUPS, default="OTHER"),
                         position="position"))
        full = aggregate(ev, onp, ["unit"]).join(pos, on="unit", how="left")
        halves = aggregate(ev, onp, ["unit", "half"]).join(pos, on="unit", how="left")
        full = full.with_columns(key=pl.lit(r["key"]), competition=pl.lit(r["competition"]),
                                 season=pl.lit(r["season"]), gender=pl.lit(r["gender"]))
        full_parts.append(full)
        half_parts.append(halves)
        print(f"{r['competition']} {r['season']} ({r['key']}): {full.height} units, "
              f"{full.filter(pl.col('minutes') >= MIN_MINUTES).height} with >= {MIN_MINUTES:.0f} min "
              f"[{time.time() - t0:.0f}s]", flush=True)

    full = pl.concat(full_parts, how="diagonal_relaxed")
    parts = full["unit"].str.split("|")
    full = full.with_columns(player_id=parts.list.get(0).cast(pl.Int64), team_id=parts.list.get(1).cast(pl.Int64))
    full = full.join(names, on="player_id", how="left")
    halves = pl.concat(half_parts, how="diagonal_relaxed")
    full.write_parquet(OUT / "traits.parquet")
    halves.write_parquet(OUT / "traits_halves.parquet")

    genders = [g for g in sorted(full["gender"].drop_nulls().unique().to_list())]
    for g in genders:
        d = OUT / g
        d.mkdir(exist_ok=True)
        fg = full.filter(pl.col("gender") == g)
        hg = halves.join(fg.select("unit"), on="unit", how="inner")
        fg.write_parquet(d / "traits.parquet")
        hg.write_parquet(d / "traits_halves.parquet")
        eligible = fg.filter(pl.col("minutes") >= MIN_MINUTES)["unit"]
        rel = split_half_reliability(hg.filter(pl.col("unit").is_in(eligible.to_list())), "unit", "group")
        print(f"\n== V1 split-half reliability, {g} ({eligible.len()} units with >= {MIN_MINUTES:.0f} min) ==")
        print(f"   keep if within-position-group r >= {KEEP_R}")
        kept = []
        for t, fam in TRAITS.items():
            x = rel[t]
            keep = x["r_within"] is not None and x["r_within"] >= KEEP_R
            kept += [t] if keep else []
            print(f"  {fam:12s} {t:24s} n={x['n']:5d} r={x['r']}  r_within={x['r_within']}" + ("  KEEP" if keep else ""))
        print(f"\n[{g}] kept {len(kept)}/{len(TRAITS)} traits")
        (d / "reliability.json").write_text(json.dumps(
            {"gender": g, "min_minutes": MIN_MINUTES, "keep_r": KEEP_R, "n_units": eligible.len(),
             "traits": rel, "kept": kept}, indent=1), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s")
