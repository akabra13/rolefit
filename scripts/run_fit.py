"""Stage 4: role translation (V4, see README).

Reads out/traits_adjusted.parquet (stage 2), out/traits.parquet and
out/reliability.json (stage 1). Writes out/fit.json.

    python scripts/run_fit.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from rolefit.fit import (build_samples, crossfit, fit_coefficients, loo_slot_demand,  # noqa: E402
                         player_bootstrap, standardised_error)
from rolefit.movers import CROSS_TEAM, mover_pairs  # noqa: E402

OUT = Path("out")
MIN_MINUTES = 270.0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    kept = json.loads((OUT / "reliability.json").read_text(encoding="utf-8"))["kept"]
    units = pl.read_parquet(OUT / "traits.parquet").filter(pl.col("group").fill_null("OTHER") != "GK")
    adj = pl.read_parquet(OUT / "traits_adjusted.parquet")
    loo = loo_slot_demand(adj, kept)
    pairs = mover_pairs(units, MIN_MINUTES)

    results = {"traits": kept}
    for name, types in (("cross_team", CROSS_TEAM), ("same_team_ceiling", ("same_team",))):
        S = build_samples(adj, loo, pairs, kept, types)
        n, players = len(S["y"]), len(np.unique(S["player"]))
        print(f"\n== {name}: {n} directed pairs from {players} players "
              f"(both units >= {MIN_MINUTES:.0f} min, other slot-mates >= 270 min) [{time.time() - t0:.0f}s]")
        if n < 100:
            print("  too few pairs")
            continue
        err = crossfit(S)
        boot = player_bootstrap(S, err)
        for m in ("full", "player_only", "system_only", "position"):
            print(f"  {m:12s} median standardised error {boot[m]['median_std_error']:.3f} "
                  f"(90% CI {boot[m]['ci90'][0]:.3f}-{boot[m]['ci90'][1]:.3f})")
        for m in ("player_only", "system_only", "position"):
            b = boot[f"full_vs_{m}"]
            print(f"  full better than {m:12s} by {b['improvement']:+.3f} (90% CI {b['ci90'][0]:+.3f} to {b['ci90'][1]:+.3f})")
        per_trait = {m: standardised_error(err[m], S["y"]) for m in err}
        coefs = fit_coefficients(S, kept)
        print(f"  {'trait':22s} {'full':>6s} {'player':>7s} {'system':>7s} {'pos':>6s}   a(player) b(new slot) c(old slot)")
        for i, t in enumerate(kept):
            c = coefs[t]
            print(f"  {t:22s} {per_trait['full'][i]:6.3f} {per_trait['player_only'][i]:7.3f} "
                  f"{per_trait['system_only'][i]:7.3f} {per_trait['position'][i]:6.3f}   "
                  f"{c['a_player']:+.2f}      {c['b_new_slot']:+.2f}        {c['c_old_slot']:+.2f}")
        wins = {m: int(np.sum(per_trait["full"] < per_trait[m])) for m in ("player_only", "system_only", "position")}
        print(f"  traits where full beats: {wins} of {len(kept)}")
        results[name] = {"n_pairs": n, "n_players": players, "models": boot, "coefficients": coefs,
                         "per_trait": {m: [round(float(x), 4) for x in per_trait[m]] for m in per_trait},
                         "full_wins": wins}

    if "cross_team" in results:
        b = results["cross_team"]["models"]
        passed = all(b[f"full_vs_{m}"]["ci90"][0] > 0 for m in ("player_only", "system_only"))
        results["v4_pass"] = bool(passed)
        print(f"\nV4 {'PASS' if passed else 'FAIL'}: full model beats both single-source baselines with the "
              f"90% bootstrap-by-player CI of the improvement above 0")
    (OUT / "fit.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s")
