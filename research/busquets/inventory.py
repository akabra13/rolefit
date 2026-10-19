"""What Busquets (and comparable holding midfielders) data does the lake hold?

Prints per competition-season: Busquets' matches, minutes, events, and how many
of those matches have 360 freeze-frames; plus the size of the comparison pool
of holding midfielders with 360 data. Writes out/results/inventory.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import polars as pl  # noqa: E402

from pitchgraph.data.lake import Lake  # noqa: E402
from rolefit.traits import spells  # noqa: E402

DM_POSITIONS = ("Center Defensive Midfield", "Left Defensive Midfield", "Right Defensive Midfield")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path("out/results")
    out.mkdir(parents=True, exist_ok=True)
    lake = Lake.open()
    m = lake.matches().with_columns(key=pl.col("competition_id").cast(pl.Utf8) + "_" + pl.col("season_id").cast(pl.Utf8))
    lu = lake.scan("lineups").collect()
    busi = lu.filter(pl.col("player").str.contains("Busquets"))
    print("players matching 'Busquets':", busi.select("player_id", "player").unique().to_dicts())
    pid = busi["player_id"].mode()[0]
    sp = spells(lu.filter(pl.col("player_id") == pid))
    mins = sp.group_by("match_id").agg(pl.col("minutes").sum())
    ev = lake.events().filter(pl.col("player_id") == pid).select(
        "match_id", "competition", "season", "type", "has_360", "position").collect()
    per = (ev.group_by("match_id").agg(pl.col("competition").first(), pl.col("season").first(), pl.len().alias("events"),
                                       pl.col("has_360").any().alias("has_360"))
             .join(mins, on="match_id", how="left"))
    summary = per.group_by("competition", "season").agg(
        pl.len().alias("matches"), pl.col("minutes").sum().round(0), pl.col("events").sum(),
        pl.col("has_360").sum().alias("matches_360")).sort("season")
    print(summary.to_pandas().to_string(index=False))
    pos = ev.group_by("position").len().sort("len", descending=True)
    print("\npositions he played (events):\n", pos.head(6).to_pandas().to_string(index=False))

    # comparison pool: players who played mostly as a defensive midfielder, by competition-season
    dm = (lake.events().filter(pl.col("position").is_in(DM_POSITIONS) & pl.col("player_id").is_not_null())
          .group_by("competition", "season", "player_id").agg(pl.len().alias("events"), pl.col("has_360").sum().alias("ev_360"))
          .collect())
    pool = dm.filter(pl.col("events") >= 400).group_by("competition", "season").agg(
        pl.len().alias("dm_players_400ev"), (pl.col("ev_360") >= 200).sum().alias("with_360")).sort("season")
    print("\nholding-midfielder pool (>= 400 events as DM):\n", pool.to_pandas().to_string(index=False))
    (out / "inventory.json").write_text(json.dumps({"busquets_id": int(pid), "busquets": summary.to_dicts(),
                                                    "dm_pool": pool.to_dicts()}, indent=1, default=str), encoding="utf-8")
