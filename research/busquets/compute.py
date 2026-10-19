"""Per player-match candidate metrics for every men's competition-season in the
lake. Writes out/results/player_match.parquet (to Drive via the workflow)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

from metrics import COLS, player_match_table  # noqa: E402
from pitchgraph.data.lake import Lake  # noqa: E402
from rolefit.traits import spells  # noqa: E402

DM_POSITIONS = ("Center Defensive Midfield", "Left Defensive Midfield", "Right Defensive Midfield")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    out = Path("out/results")
    out.mkdir(parents=True, exist_ok=True)
    lake = Lake.open()
    m = lake.matches().filter(pl.col("gender") == "male").with_columns(
        key=pl.col("competition_id").cast(pl.Utf8) + "_" + pl.col("season_id").cast(pl.Utf8))
    lu = lake.scan("lineups").filter(pl.col("match_id").is_in(m["match_id"].to_list())).collect()
    sp_all = spells(lu)
    parts = []
    for (key,), mm in m.group_by(["key"]):
        ids = mm["match_id"].to_list()
        ev = lake.scan("events").filter(pl.col("match_id").is_in(ids)).select(COLS + ["position"]).collect()
        fr = (lake.scan("frames").filter(pl.col("match_id").is_in(ids)).select("event_id", "teammate", "x", "y").collect()
              if lake.has("frames") else None)
        sp = sp_all.filter(pl.col("match_id").is_in(ids))
        t = player_match_table(ev.drop("position"), sp, fr)
        pos = (ev.filter(pl.col("position").is_not_null()).group_by("match_id", "player_id", "position").len()
                 .sort("len", descending=True).group_by("match_id", "player_id", maintain_order=True).first()
                 .select("match_id", "player_id", "position"))
        mins = sp.group_by("match_id", "player_id", "team_id").agg(pl.col("minutes").sum())
        t = (t.join(pos, on=["match_id", "player_id"], how="left").join(mins, on=["match_id", "player_id"], how="left")
               .with_columns(key=pl.lit(key), competition=pl.lit(mm["competition"][0]), season=pl.lit(mm["season"][0]),
                             is_dm=pl.col("position").is_in(DM_POSITIONS)))
        parts.append(t)
        print(f"{mm['competition'][0]} {mm['season'][0]}: {len(ids)} matches, {t.height} player-matches, "
              f"360={'yes' if fr is not None and fr.height else 'no'} [{time.time() - t0:.0f}s]", flush=True)
    allm = pl.concat(parts, how="diagonal_relaxed")
    names = lu.select("player_id", "player").unique("player_id")
    allm = allm.join(names, on="player_id", how="left")
    allm.write_parquet(out / "player_match.parquet")
    print(f"{allm.height} player-matches written [{time.time() - t0:.0f}s]")
