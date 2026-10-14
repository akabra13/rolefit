"""Team context tables: formation shares per team-season from Starting XI events.

Writes out/team_formations.parquet (shared by both genders; filtered later).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl  # noqa: E402

from rolefit.context import formation_shares, starting_formations  # noqa: E402

OUT = Path(os.environ.get("ROLEFIT_OUT", "out"))

if __name__ == "__main__":
    root = Path(os.environ["PITCHGRAPH_DATA"]) / "lake" / "events"
    ev = pl.scan_parquet(str(root / "*.parquet")).filter(pl.col("type") == "Starting XI").select(
        "match_id", "team_id", "type", "extra").collect()
    m = pl.read_parquet(str(Path(os.environ["PITCHGRAPH_DATA"]) / "lake" / "matches" / "*.parquet")).select(
        "match_id", "competition_id", "season_id")
    ev = ev.join(m, on="match_id")
    f = starting_formations(ev)
    sh = formation_shares(f)
    OUT.mkdir(exist_ok=True)
    sh.write_parquet(OUT / "team_formations.parquet")
    print(f"{f.height} starting XIs, {sh.height} team-seasons")
    print(f.group_by("family").len().sort("len", descending=True).to_pandas().to_string(index=False))
