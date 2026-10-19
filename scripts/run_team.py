"""Team needs: team-season metrics, V9 (do they explain results?), weaknesses,
and the type of player each team needs. See README (V9).

Needs the lake (events, matches) and, per gender, out/<g>/reliability.json,
traits.parquet, traits_adjusted.parquet, slot_demand.parquet.
Writes out/<g>/team.json and out/<g>/team_metrics.parquet.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from pitchgraph.data.lake import Lake  # noqa: E402

from rolefit.fit import _ridge  # noqa: E402
from rolefit.needs import TRAIT_LINKS, peer_z, slot_for_need, weaknesses  # noqa: E402
from rolefit.team import METRICS, team_match_counts, team_season_metrics  # noqa: E402

OUT = Path("out")
OUTCOME = "points_pm"
PREDICTORS = tuple(m for m in METRICS if m not in ("xg_for_pp", "xg_against_pp"))
MIN_R2 = 0.25
COLS = ["match_id", "team_id", "possession", "possession_team_id", "period", "type", "sub_type", "outcome", "x", "y",
        "end_x", "end_y", "xg", "play_pattern", "extra"]


def v9(ts: pl.DataFrame, n_boot: int = 300, seed: int = 0) -> dict:
    """Leave-one-competition-season-out ridge of points per match on peer
    z-scored team metrics (league-scope teams with >= 10 matches only)."""
    d = ts.filter((pl.col("peer_scope") == "league") & (pl.col("matches") >= 10))
    cols = [f"z_{m}" for m in PREDICTORS]
    d = d.drop_nulls(cols + [OUTCOME])
    X, y, keys = d.select(cols).to_numpy(), d[OUTCOME].to_numpy(), d["key"].to_numpy()
    y = y - np.array([y[keys == k].mean() for k in keys])       # within-league points
    pred = np.zeros(len(y))
    for k in np.unique(keys):
        tr, te = keys != k, keys == k
        c, b, _, _ = _ridge(X[tr], y[tr], lam=5.0)
        pred[te] = X[te] @ c + b
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rng = np.random.default_rng(seed)
    uk = np.unique(keys)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uk, size=len(uk), replace=True)
        idx = np.concatenate([np.flatnonzero(keys == k) for k in pick])
        boots.append(1 - ((y[idx] - pred[idx]) ** 2).sum() / ((y[idx] - y[idx].mean()) ** 2).sum())
    c, _, _, _ = _ridge(X, y, lam=5.0)
    sd = X.std(0)
    imp = {m: float(ci * s / (y.std() or 1)) for m, ci, s in zip(PREDICTORS, c, sd)}
    return {"n_teams": int(len(y)), "n_seasons": int(len(uk)), "cv_r2": round(float(r2), 3),
            "ci90": [round(float(np.percentile(boots, 5)), 3), round(float(np.percentile(boots, 95)), 3)],
            "importance": {k: round(v, 3) for k, v in imp.items()}}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    lake = Lake.open()
    matches = lake.matches().with_columns(key=pl.col("competition_id").cast(pl.Utf8) + "_" + pl.col("season_id").cast(pl.Utf8))
    parts = []
    for (key,), mm in matches.group_by(["key"]):
        ev = lake.scan("events").filter(pl.col("match_id").is_in(mm["match_id"].to_list())).select(COLS).collect()
        parts.append(team_match_counts(ev, mm).with_columns(key=pl.lit(key)))
    tm = pl.concat(parts, how="diagonal_relaxed")
    print(f"{tm.height} team-matches [{time.time() - t0:.0f}s]")
    info = matches.select("key", "competition", "season", "gender").unique("key")
    names = {}
    for a, b in (("home_team_id", "home_team"), ("away_team_id", "away_team")):
        names.update(dict(zip(matches[a].to_list(), matches[b].to_list())))
    ts = team_season_metrics(tm).join(info, on="key", how="left")

    for g in ("male", "female"):
        d = OUT / g
        tg = peer_z(ts.filter(pl.col("gender") == g))
        res = v9(tg)
        passed = res["cv_r2"] >= MIN_R2 and res["ci90"][0] > 0
        print(f"\n== V9 {g}: points per match from team metrics, leave-one-season-out ==\n  {res}\n  "
              f"{'PASS' if passed else 'FAIL'} (bar: CV R2 >= {MIN_R2} and CI above 0)")
        imp = {m: abs(v) for m, v in res["importance"].items()} if passed else None
        kept = json.loads((d / "reliability.json").read_text(encoding="utf-8"))["kept"]
        slots = pl.read_parquet(d / "slot_demand.parquet") if (d / "slot_demand.parquet").exists() else pl.DataFrame()
        adj = pl.read_parquet(d / "traits_adjusted.parquet")
        teams = []
        for r in tg.filter(pl.col("matches") >= 8).iter_rows(named=True):
            weak = weaknesses(r, imp)
            ts_slots = slots.filter((pl.col("key") == r["key"]) & (pl.col("team_id") == r["team_id"])) if slots.height else slots
            for w in weak:
                w["slot"] = slot_for_need(w["metric"], ts_slots)
                w["traits"] = {t: s for t, s in TRAIT_LINKS.get(w["metric"], {}).items() if t in kept}
            team_units = adj.filter((pl.col("key") == r["key"]) & (pl.col("team_id") == r["team_id"]))
            sv = {}
            for grp, gdf in team_units.group_by("group"):
                wts = gdf["minutes"].to_numpy()
                sv[grp[0]] = [None if not np.isfinite(v) else round(float(v), 3) for v in
                              [np.nansum(gdf[f"{t}__pos"].to_numpy() * wts) / max(np.nansum(wts * np.isfinite(gdf[f"{t}__pos"].to_numpy())), 1e-9)
                               for t in kept]]
            allw = team_units["minutes"].to_numpy()
            ov = [None if not np.isfinite(v) else round(float(v), 3) for v in
                  [np.nansum(team_units[f"{t}__pos"].to_numpy() * allw) / max(np.nansum(allw * np.isfinite(team_units[f"{t}__pos"].to_numpy())), 1e-9)
                   for t in kept]] if team_units.height else None
            teams.append({"key": r["key"], "team_id": r["team_id"], "team": names.get(r["team_id"], str(r["team_id"])),
                          "competition": r["competition"], "season": r["season"], "matches": r["matches"],
                          "peer_scope": r["peer_scope"], "points_pm": round(r["points_pm"], 2),
                          "xg_diff_pm": round(r["xg_diff_pm"], 2),
                          "z": {m: None if r[f"z_{m}"] is None or not np.isfinite(r[f"z_{m}"]) else round(r[f"z_{m}"], 2)
                                for m in METRICS},
                          "weaknesses": weak, "slot_demand": sv, "squad": ov})
        tg.write_parquet(d / "team_metrics.parquet")
        (d / "team.json").write_text(json.dumps({"v9": res, "v9_pass": passed, "weighting": "importance" if passed else "equal",
                                                 "teams": teams}, indent=1, default=str), encoding="utf-8")
        ex = [t for t in teams if t["peer_scope"] == "league"][:3]
        for t in ex:
            print(f"  {t['team']} {t['competition']} {t['season']}: " +
                  "; ".join(f"{w['label']} (z {w['z_good']}) -> {w['slot']['group'] if w['slot'] else '?'}" for w in t["weaknesses"]))
    print(f"done in {time.time() - t0:.0f}s")
