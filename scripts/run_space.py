"""V8: which similarity space should the page use? (see README)

Compares candidate "plays like" spaces on the units the page shows (outfield,
>= 900 minutes) by (a) cross-team re-identification: does a player's nearest
neighbours include the same player at another team, and (b) position
agreement: share of the 5 nearest other players in the same position group.

    ROLEFIT_OUT=out/male python scripts/run_space.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from rolefit.export import NOT_STYLE  # noqa: E402
from rolefit.movers import CROSS_TEAM, mover_pairs  # noqa: E402
from rolefit.portability import embed  # noqa: E402
from rolefit.similar import add_absolute, merge_weights  # noqa: E402
from rolefit.validate import reid  # noqa: E402

OUT = Path(os.environ.get("ROLEFIT_OUT", "out"))
MIN_MINUTES = 900.0
MIN_POSITION_AGREEMENT = 0.70


def position_agreement(E: np.ndarray, groups: list, pids: list, k: int = 5) -> float:
    g = np.array(groups, dtype=object)
    p = np.array(pids)
    sq = (E ** 2).sum(1)
    agree, n = 0, 0
    for i in range(len(E)):
        d = sq + sq[i] - 2 * E @ E[i]
        d[p == p[i]] = np.inf
        nn = np.argpartition(d, k)[:k]
        agree += int((g[nn] == g[i]).sum())
        n += k
    return agree / n


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    units = pl.read_parquet(OUT / "traits.parquet")
    adj = pl.read_parquet(OUT / "traits_adjusted.parquet")
    w = merge_weights(json.loads((OUT / "portability.json").read_text(encoding="utf-8"))["weights"])
    traits = [t for t in w if t not in NOT_STYLE]
    frame = add_absolute(adj.filter(pl.col("minutes") >= MIN_MINUTES), traits)
    frame = frame.with_columns([(pl.col(f"{t}__abs") - pl.col(f"{t}__team").fill_null(0.0)).alias(f"{t}__abst")
                                for t in traits])
    spaces = {
        "A_position_kept (current page)": {t: ("abs", w[t][1]) for t in traits},
        "B_position_kept_minus_team_style": {t: ("abst", w[t][1]) for t in traits},
        "C_B_with_equal_weights": {t: ("abst", 1.0) for t in traits},
    }
    ulist = frame["unit"].to_list()
    idx = {u: i for i, u in enumerate(ulist)}
    groups = frame["group"].fill_null("OTHER").to_list()
    pids = frame["player_id"].to_list()
    pairs = mover_pairs(units.filter(pl.col("group").fill_null("OTHER") != "GK"), MIN_MINUTES)
    partners: dict = {}
    for a, b, t in pairs.select("unit_a", "unit_b", "type").iter_rows():
        if a in idx and b in idx:
            partners.setdefault(idx[a], []).append((idx[b], t))
            partners.setdefault(idx[b], []).append((idx[a], t))
    q = [i for i, pl_ in partners.items() if any(t in CROSS_TEAM for _, t in pl_)]
    correct = [{j for j, t in partners[i] if t in CROSS_TEAM} for i in q]
    exclude = [{i} | {j for j, t in partners[i] if t not in CROSS_TEAM} for i in q]
    print(f"{frame.height} units, {len(q)} cross-team queries")
    res = {}
    for name, choice in spaces.items():
        E = embed(frame, choice)
        r = reid(E[q], E, correct, exclude, groups_q=[groups[i] for i in q], groups_g=groups, metric="euclidean")
        pa = position_agreement(E, groups, pids)
        res[name] = {"cross_team_top5": round(r["top5"], 4), "ci90": [round(x, 4) for x in r["top5_ci90"]],
                     "top1": round(r["top1"], 4), "position_agreement": round(pa, 3), "n": r["n"]}
        print(f"  {name:38s} cross-team top5 {r['top5']:.3f} (90% CI {r['top5_ci90'][0]:.3f}-{r['top5_ci90'][1]:.3f}) "
              f"top1 {r['top1']:.3f}  position agreement {pa:.3f}")
    ok = {k: v for k, v in res.items() if v["position_agreement"] >= MIN_POSITION_AGREEMENT}
    chosen = max(ok, key=lambda k: ok[k]["cross_team_top5"]) if ok else "A_position_kept (current page)"
    print(f"chosen: {chosen}")
    (OUT / "space.json").write_text(json.dumps({"results": res, "chosen": chosen}, indent=1), encoding="utf-8")
