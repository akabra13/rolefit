"""Build out/library.json for the Library page from stages 1-2 tables."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl  # noqa: E402

from pitchgraph.data.lake import Lake  # noqa: E402

from rolefit.export import build_payload  # noqa: E402

OUT = Path("out")

if __name__ == "__main__":
    kept = json.loads((OUT / "reliability.json").read_text(encoding="utf-8"))["kept"]
    units = pl.read_parquet(OUT / "traits.parquet")
    adj = pl.read_parquet(OUT / "traits_adjusted.parquet")
    weights = json.loads((OUT / "portability.json").read_text(encoding="utf-8"))["weights"]
    m = Lake.open().matches()
    names = {}
    for a, b in (("home_team_id", "home_team"), ("away_team_id", "away_team")):
        names.update(dict(zip(m[a].to_list(), m[b].to_list())))
    payload = build_payload(units, adj, weights, kept, names)
    (OUT / "library.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                                      encoding="utf-8")
    size = (OUT / "library.json").stat().st_size / 1e6
    print(f"{len(payload['players'])} players, {len(kept)} traits, {payload['n_embed_dims']} embedding dims, "
          f"{size:.2f} MB; translation fitted on {payload['translation']['n_pairs']} pairs")
