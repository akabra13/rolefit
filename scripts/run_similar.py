"""V6 face validity: 'plays like' lists for well-known players (see README).

Reads out/traits.parquet, out/traits_adjusted.parquet, out/portability.json.
Writes out/similar.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl  # noqa: E402

from rolefit.similar import build_space, find_units, plays_like  # noqa: E402

OUT = Path(os.environ.get("ROLEFIT_OUT", "out"))
QUERIES = ["Sergio Busquets", "Lionel Messi", "Kevin De Bruyne", "Kanté", "Jamie Vardy", "Virgil van Dijk",
           "Jordi Alba", "Philipp Lahm", "Alexia Putellas", "Lauren Hemp", "Cristiano Ronaldo", "Gerard Piqu",
           "Luka Modri", "Toni Kroos", "Neymar", "Sergio Ramos", "Dani Alves", "Luis Suárez",
           "Aitana Bonmat", "Vivianne Miedema", "Keira Walsh", "Lucy Bronze", "Alex Greenwood", "Ada Hegerberg",
           "Mapi", "Irene Paredes", "Ewa Pajor", "Lena Oberdorf"]

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    units = pl.read_parquet(OUT / "traits.parquet")
    names = units.select("unit", "player", "nickname", "competition", "season", "position")
    adj = pl.read_parquet(OUT / "traits_adjusted.parquet")
    weights = json.loads((OUT / "portability.json").read_text(encoding="utf-8"))["weights"]
    frame, E = build_space(adj, weights)
    frame = frame.join(names, on="unit", how="left")
    ids = frame["unit"].to_list()
    out, same_group, total = {}, 0, 0
    for q in QUERIES:
        hit = find_units(frame, q)
        if hit.is_empty():
            print(f"\n[{q}] no unit with >= 900 minutes")
            continue
        r = hit.row(0, named=True)
        # embedding rows align with `frame`
        res = plays_like(frame, E, r["unit"]).join(frame.select("unit", "player", "team_id", "competition",
                                                                  "season", "group", "position"), on="unit")
        print(f"\n[{r['player']}] {r['competition']} {r['season']}, {r['group']} ({r['position']}), {r['minutes']:.0f} min")
        for x in res.iter_rows(named=True):
            print(f"    {x['distance']:5.2f}  {x['player']}  ({x['competition']} {x['season']}, {x['group']})")
            total += 1
            same_group += int(x["group"] == r["group"])
        out[q] = {"unit": r["unit"], "player": r["player"], "neighbours": res.to_dicts()}
    print(f"\nshare of neighbours in the query's position group: {same_group}/{total}")
    (OUT / "similar.json").write_text(json.dumps(out, indent=1, default=str, ensure_ascii=False), encoding="utf-8")
