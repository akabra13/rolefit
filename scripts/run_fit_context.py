"""V4b: does the new team's context (formation, style, the rest of the lineup)
improve prediction of how a mover will play? See README. Needs out/team_formations.parquet.

    ROLEFIT_OUT=out/male python scripts/run_fit_context.py
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

from rolefit.context import others_demand  # noqa: E402
from rolefit.fit import (CTX_MODELS, add_context, build_samples, context_bootstrap, crossfit,  # noqa: E402
                         loo_slot_demand, player_bootstrap, standardised_error)
from rolefit.movers import CROSS_TEAM, mover_pairs  # noqa: E402

OUT = Path(os.environ.get("ROLEFIT_OUT", "out"))
ROOT = Path(os.environ.get("ROLEFIT_ROOT", "out"))
MIN_MINUTES = 270.0

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    kept = json.loads((OUT / "reliability.json").read_text(encoding="utf-8"))["kept"]
    units = pl.read_parquet(OUT / "traits.parquet").filter(pl.col("group").fill_null("OTHER") != "GK")
    adj = pl.read_parquet(OUT / "traits_adjusted.parquet")
    shares = pl.read_parquet(ROOT / "team_formations.parquet")
    loo = loo_slot_demand(adj, kept)
    others = others_demand(adj, kept)
    pairs = mover_pairs(units, MIN_MINUTES)
    S0 = build_samples(adj, loo, pairs, kept, CROSS_TEAM)
    S = add_context(S0, others, shares, kept)
    n, players = len(S["y"]), len(np.unique(S["player"]))
    print(f"{n} directed pairs from {players} players with full team context (of {len(S0['y'])} without) "
          f"[{time.time() - t0:.0f}s]")
    err = crossfit(S)
    base = player_bootstrap(S, {m: err[m] for m in ("full", "player_only", "system_only", "position")})
    ctx = context_bootstrap(S, err)
    print(f"  full (no team context)    median standardised error {base['full']['median_std_error']:.3f}")
    for m in CTX_MODELS:
        c = ctx[m]
        print(f"  {m:15s} error {c['median_std_error']:.3f}  improvement over full {c['improvement_over_full']:+.4f} "
              f"(90% CI {c['ci90'][0]:+.4f} to {c['ci90'][1]:+.4f})")
    helps = ctx["full_ctx"]["ci90"][0] > 0
    per = {m: standardised_error(err[m], S["y"]) for m in err}
    gains = per["full"] - per["full_ctx"]
    order = np.argsort(-gains)
    print("  traits helped most by context: " + ", ".join(f"{kept[i]} ({gains[i]:+.3f})" for i in order[:8]))
    print("  traits hurt most by context:   " + ", ".join(f"{kept[i]} ({gains[i]:+.3f})" for i in order[::-1][:5]))
    print(f"V4b {'PASS' if helps else 'FAIL'}: full_ctx beats full with the 90% CI of the improvement above 0")
    (OUT / "fit_context.json").write_text(json.dumps(
        {"n_pairs": n, "n_players": players, "base": base, "context": ctx, "v4b_pass": bool(helps)}, indent=1),
        encoding="utf-8")
