"""Pre-registered back-test of the team-needs assumption (see README.md).

Reads (pulled by the research workflow):
  out/team/<g>/team_metrics.parquet
  out/traits/<g>/traits.parquet
  out/portability/<g>/traits_adjusted.parquet
Writes out/results/backtest.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rolefit.needs import TRAIT_LINKS  # noqa: E402
from rolefit.team import GOOD_HIGH  # noqa: E402

AREAS = [m for m, g in GOOD_HIGH.items() if g is not None and m in TRAIT_LINKS]
MIN_MIN = 450.0
OUT = Path("out")


def year(season: str) -> int | None:
    try:
        return int(str(season)[:4])
    except ValueError:
        return None


def area_scores(adj: pl.DataFrame, units: pl.DataFrame) -> pl.DataFrame:
    """Per unit: strength in every area from its position-relative traits."""
    ref = adj.filter(pl.col("minutes") >= 900)
    sd = {c: float(ref[c].std()) for c in adj.columns if c.endswith("__pos")}
    cols = []
    for a in AREAS:
        parts = []
        for t, sgn in TRAIT_LINKS[a].items():
            c = f"{t}__pos"
            if c in adj.columns and sd.get(c):
                parts.append(sgn * pl.col(c) / sd[c])
        cols.append(pl.mean_horizontal(parts).alias(f"S_{a}") if parts else pl.lit(None).alias(f"S_{a}"))
    s = adj.select(["unit"] + cols)
    return units.select("unit", "player_id", "team_id", "key", "competition", "season", "minutes", "group") \
                .join(s, on="unit", how="left").with_columns(yr=pl.col("season").map_elements(year, return_dtype=pl.Int32))


def run(gender: str, rng) -> dict:
    tm = pl.read_parquet(OUT / "team" / gender / "team_metrics.parquet").filter(pl.col("matches") >= 10)
    units = pl.read_parquet(OUT / "traits" / gender / "traits.parquet").filter(pl.col("group").fill_null("OTHER") != "GK")
    adj = pl.read_parquet(OUT / "portability" / gender / "traits_adjusted.parquet")
    pu = area_scores(adj, units)
    tm = tm.with_columns(yr=pl.col("season").map_elements(year, return_dtype=pl.Int32),
                         comp=pl.col("key").str.split("_").list.get(0))
    sd_m = {a: float(tm[a].std()) for a in AREAS}
    rows, n_arr, n_arr_prof = [], 0, 0
    for r in tm.iter_rows(named=True):
        nxt = tm.filter((pl.col("team_id") == r["team_id"]) & (pl.col("comp") == r["comp"]) & (pl.col("yr") == r["yr"] + 1))
        if nxt.height != 1:
            continue
        r2 = nxt.row(0, named=True)
        cur = pu.filter((pl.col("team_id") == r["team_id"]) & (pl.col("key") == r["key"]))
        new = pu.filter((pl.col("team_id") == r["team_id"]) & (pl.col("key") == r2["key"]))
        cur_p, new_p = set(cur.filter(pl.col("minutes") > 0)["player_id"].to_list()), set(new["player_id"].to_list())
        arr = new.filter(~pl.col("player_id").is_in(list(cur_p)) & (pl.col("minutes") >= MIN_MIN))
        dep = cur.filter(~pl.col("player_id").is_in(list(new_p)) & (pl.col("minutes") >= MIN_MIN))
        tot_new, tot_cur = float(new["minutes"].sum()) or 1.0, float(cur["minutes"].sum()) or 1.0
        prior = {}
        for pid in arr["player_id"].to_list():
            n_arr += 1
            prev = pu.filter((pl.col("player_id") == pid) & (pl.col("yr") < r2["yr"]) & (pl.col("minutes") >= MIN_MIN)) \
                     .sort(["yr", "minutes"], descending=[True, True])
            if prev.height:
                n_arr_prof += 1
                prior[pid] = prev.row(0, named=True)
        tid = f"{r['team_id']}|{r['key']}"
        for a in AREAS:
            g = GOOD_HIGH[a]
            if r[a] is None or r2[a] is None or not sd_m[a]:
                continue
            delta = (r2[a] - r[a]) * (1 if g else -1) / sd_m[a]
            zt = r.get(f"z_{a}")
            zt = (zt if g else -zt) if zt is not None and np.isfinite(zt) else 0.0
            a_s = sum(float(m) / tot_new * (prior[p][f"S_{a}"] or 0.0)
                      for p, m in zip(arr["player_id"].to_list(), arr["minutes"].to_list()) if p in prior)
            d_s = sum(float(m) / tot_cur * (s or 0.0) for m, s in zip(dep["minutes"].to_list(), dep[f"S_{a}"].to_list()))
            rows.append((tid, a, delta, a_s, d_s, zt))
    if len(rows) < 30:
        return {"rows": len(rows)}
    tids = np.array([x[0] for x in rows])
    y = np.array([x[2] for x in rows])
    X = np.column_stack([np.ones(len(rows)), [x[3] for x in rows], [x[4] for x in rows], [x[5] for x in rows]])
    Xw = np.column_stack([X, X[:, 1] * (X[:, 3] < -0.5)])

    def fit(idx, M):
        b, *_ = np.linalg.lstsq(M[idx], y[idx], rcond=None)
        return b

    allidx = np.arange(len(rows))
    b = fit(allidx, X)
    bw = fit(allidx, Xw)
    ut = np.unique(tids)
    members = {t: np.flatnonzero(tids == t) for t in ut}
    boots, bootsw = [], []
    for _ in range(2000):
        pick = rng.choice(ut, size=len(ut), replace=True)
        idx = np.concatenate([members[t] for t in pick])
        boots.append(fit(idx, X)[1])
        bootsw.append(fit(idx, Xw)[4])
    ci = [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))]
    return {"transitions": int(len(ut)), "rows": int(len(rows)), "arrivals": n_arr, "arrivals_with_prior_profile": n_arr_prof,
            "b_arrivals": round(float(b[1]), 3), "ci90": [round(c, 3) for c in ci],
            "b_departures": round(float(b[2]), 3), "b_start_z": round(float(b[3]), 3),
            "pass": bool(ci[0] > 0),
            "weak_area_extra": round(float(bw[4]), 3),
            "weak_area_extra_ci90": [round(float(np.percentile(bootsw, 5)), 3), round(float(np.percentile(bootsw, 95)), 3)],
            "mean_arrival_strength": round(float(X[:, 1].mean()), 3), "sd_arrival_strength": round(float(X[:, 1].std()), 3)}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(0)
    res = {}
    for g in ("female", "male"):
        res[g] = run(g, rng)
        print(f"\n== {g} ==\n  {res[g]}")
        if "pass" in res[g]:
            print(f"  primary: {'PASS' if res[g]['pass'] else 'FAIL'} (b_arrivals > 0 with 90% CI above 0)")
    Path("out/results").mkdir(parents=True, exist_ok=True)
    Path("out/results/backtest.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
