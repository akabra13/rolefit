"""Stage 3: slot demand profiles and the V3b persistence test (see README).

Reads out/traits.parquet and out/reliability.json (pulled from Drive).
Writes out/slot_demand.parquet and out/demand.json.

    python scripts/run_demand.py
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

from rolefit.demand import persistence, relative_demand, slot_card, slot_table  # noqa: E402

OUT = Path(os.environ.get("ROLEFIT_OUT", "out"))
MIN_SLOT_MINUTES = 900.0
MEDIAN_R_BAR = 0.20


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    rng = np.random.default_rng(0)
    kept = json.loads((OUT / "reliability.json").read_text(encoding="utf-8"))["kept"]
    units = pl.read_parquet(OUT / "traits.parquet")
    slots = relative_demand(slot_table(units, kept, MIN_SLOT_MINUTES), kept)
    slots.write_parquet(OUT / "slot_demand.parquet")
    print(f"{slots.height} slots (>= {MIN_SLOT_MINUTES:.0f} slot-minutes) from {units.height} units; "
          f"{slots['team_id'].n_unique()} teams, {slots['key'].n_unique()} competition-seasons "
          f"[{time.time() - t0:.0f}s]")
    print(slots.group_by("group").len().sort("group").to_pandas().to_string(index=False))

    per = persistence(slots, kept, rng)
    print(f"\n== V3b slot-demand persistence: same team + position group across two seasons ==")
    print(f"   pairs={per['n_pairs']}, of which main player changed={per.get('n_player_changed')}")
    tr = per["traits"]
    for t in kept:
        x = tr.get(t)
        if not x:
            continue
        f = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        print(f"  {t:22s} same team {f(x['r_same_team'])}   same team, new main player "
              f"{f(x['r_same_team_new_player'])}   other team {f(x['r_other_team'])}")
    m_new = med([x["r_same_team_new_player"] for x in tr.values()])
    m_base = med([x["r_other_team"] for x in tr.values()])
    m_same = med([x["r_same_team"] for x in tr.values()])
    passed = bool(m_new is not None and m_base is not None and m_new >= MEDIAN_R_BAR and m_new > m_base + 0.1)
    print(f"\nmedian over traits: same team {m_same}, same team new main player {m_new}, other team {m_base}")
    print(f"V3b {'PASS' if passed else 'FAIL'} (bar: median same-team-new-player r >= {MEDIAN_R_BAR} "
          f"and > other-team median + 0.1)")

    # a few readable cards: largest slots for the biggest teams in the most recent full seasons
    sample = (slots.sort("minutes", descending=True)
                   .join(units.select("team_id", "key", "competition", "season").unique(
                       subset=["team_id", "key"]), on=["team_id", "key"], how="left")
                   .head(12))
    cards = []
    for r in sample.iter_rows(named=True):
        c = slot_card(r, kept)
        cards.append({"key": r["key"], "team_id": r["team_id"], "competition": r["competition"],
                      "season": r["season"], "group": r["group"], "main_player": r["main_player"], **c})
    (OUT / "demand.json").write_text(json.dumps(
        {"n_slots": slots.height, "persistence": per, "median_r": {"same_team": m_same,
         "same_team_new_player": m_new, "other_team": m_base}, "pass": passed, "cards": cards}, indent=1),
        encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s")
