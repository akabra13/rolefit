"""Pre-registered analysis of the Busquets signature study (see README.md).

Reads out/research/busquets_metrics/player_match.parquet. Writes
out/results/analysis.json and prints the tables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

BUSQUETS = 5203
MIN_MATCHES = 10
MIN_MATCHES_360 = 4
MIN_SEASON_MATCHES = 8
TOURNAMENTS = ("FIFA World Cup", "UEFA Euro", "Copa America", "African Cup of Nations")
MID_POSITIONS = ("Center Defensive Midfield", "Left Defensive Midfield", "Right Defensive Midfield",
                 "Center Midfield", "Left Center Midfield", "Right Center Midfield")

# metric -> (numerator, denominator, kind); kind "rate" = num/den, "p90" = num per 90 minutes
METRICS = {
    "fwd_after_recv": ("rp_fwd", "rp", "rate"),
    "turn_after_back": ("rb_fwd", "rb", "rate"),
    "bounce": ("bounce", "rp", "rate"),
    "redirect": ("redirect", "rp", "rate"),
    "central_break": ("central_break", "op", "rate"),
    "quick_release": ("quick", "rp", "rate"),
    "press_keep": ("up_keep", "up_recv", "rate"),
    "buildup_share": ("buildup_touched", "buildup_poss", "rate"),
    "involvement_p90": ("sequences", "minutes", "p90"),
    "counter_foul_p90": ("counter_foul", "minutes", "p90"),
    "mid_intercept_p90": ("mid_intercept", "minutes", "p90"),
    "bypassed": ("bypassed_sum", "f_passes", "rate"),
    "line_break": ("line_breaks", "f_passes", "rate"),
    "pass_space": ("space_sum", "f_passes", "rate"),
    "behind_recv": ("behind_sum", "f_recv", "rate"),
    "crowd_recv": ("crowd_sum", "f_recv", "rate"),
}
FRAME_METRICS = {"bypassed", "line_break", "pass_space", "behind_recv", "crowd_recv"}


def rate(df: pl.DataFrame, metric: str) -> float | None:
    num, den, kind = METRICS[metric]
    n, d = df[num].fill_null(0).sum(), df[den].fill_null(0).sum()
    if d is None or d <= 0:
        return None
    return float(n / d * (90.0 if kind == "p90" else 1.0))


def player_rates(df: pl.DataFrame, metric: str, min_matches: int) -> pl.DataFrame:
    num, den, kind = METRICS[metric]
    d = df.filter(pl.col(den).fill_null(0) > 0)
    g = d.group_by("player_id").agg(pl.col(num).fill_null(0).sum().alias("n"), pl.col(den).sum().alias("d"),
                                    pl.len().alias("matches"))
    g = g.filter(pl.col("matches") >= min_matches)
    return g.with_columns(r=pl.col("n") / pl.col("d") * (90.0 if kind == "p90" else 1.0))


def split_half_r(df: pl.DataFrame, metric: str, min_matches: int) -> tuple[float | None, int]:
    num, den, kind = METRICS[metric]
    d = df.filter(pl.col(den).fill_null(0) > 0).sort("player_id", "match_id").with_columns(
        h=(pl.int_range(pl.len()).over("player_id") % 2), cnt=pl.len().over("player_id"))
    d = d.filter(pl.col("cnt") >= min_matches)
    k = 90.0 if kind == "p90" else 1.0
    g = d.group_by("player_id", "h").agg((pl.col(num).fill_null(0).sum() / pl.col(den).sum() * k).alias("r"))
    w = g.pivot(on="h", index="player_id", values="r").drop_nulls()
    if w.height < 20:
        return None, w.height
    return float(np.corrcoef(w["0"].to_numpy(), w["1"].to_numpy())[0, 1]), w.height


def percentile(value: float, pool: np.ndarray) -> float:
    return float((pool < value).mean() * 100 + (pool == value).mean() * 50)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    Path("out/results").mkdir(parents=True, exist_ok=True)
    pm = pl.read_parquet("out/research/busquets_metrics/player_match.parquet").filter(
        pl.col("minutes").fill_null(0) >= 45)
    dm = pm.filter(pl.col("is_dm"))
    busi = dm.filter(pl.col("player_id") == BUSQUETS)
    barca_team = busi.filter(~pl.col("competition").is_in(TOURNAMENTS))["team_id"].mode()[0]
    b_club = busi.filter(pl.col("team_id") == barca_team)
    b_spain = busi.filter(pl.col("competition").is_in(TOURNAMENTS))
    pool = dm.filter(pl.col("player_id") != BUSQUETS)
    tour_pool = pool.filter(pl.col("competition").is_in(TOURNAMENTS))
    mates = pm.filter((pl.col("team_id") == barca_team) & (pl.col("player_id") != BUSQUETS)
                      & pl.col("position").is_in(MID_POSITIONS)
                      & pl.col("match_id").is_in(b_club["match_id"].to_list()))
    print(f"Busquets: {b_club.height} Barcelona and {b_spain.height} Spain holding-midfield matches; "
          f"pool {pool['player_id'].n_unique()} players / {pool.height} matches")

    rows = {}
    for mname in METRICS:
        is360 = mname in FRAME_METRICS
        mm = MIN_MATCHES_360 if is360 else MIN_MATCHES
        rel, nrel = split_half_r(pool, mname, mm)
        pr = player_rates(pool, mname, mm)
        his = rate(b_club, mname)
        if his is None or pr.height < 20:
            rows[mname] = {"reliability": rel, "n_rel": nrel, "pool_players": pr.height, "busquets": his}
            continue
        vals = pr["r"].to_numpy()
        z = (his - vals.mean()) / vals.std()
        med = float(np.median(vals))
        # season consistency
        seas = b_club.group_by("season").agg(pl.len().alias("n")).filter(pl.col("n") >= MIN_SEASON_MATCHES)["season"].to_list()
        same = []
        for s in seas:
            r_s = rate(b_club.filter(pl.col("season") == s), mname)
            if r_s is not None:
                same.append((r_s > med) == (z > 0))
        cons = float(np.mean(same)) if same else None
        # Spain vs tournament pool
        tp = player_rates(tour_pool, mname, 3)
        sp_rate = rate(b_spain, mname)
        sp_pct = percentile(sp_rate, tp["r"].to_numpy()) if sp_rate is not None and tp.height >= 20 else None
        # Barcelona midfield team-mates in the same matches
        mt = player_rates(mates, mname, 10)
        mates_pct = percentile(his, mt["r"].to_numpy()) if mt.height >= 3 else None
        port_ok = sp_pct is not None and ((z > 0 and sp_pct >= 75) or (z < 0 and sp_pct <= 25))
        sig = bool(rel is not None and rel >= 0.5 and abs(z) >= 1.5 and cons is not None and cons >= 0.75 and port_ok)
        rows[mname] = {"reliability": None if rel is None else round(rel, 3), "n_rel": nrel, "pool_players": pr.height,
                       "busquets": round(his, 4), "pool_mean": round(float(vals.mean()), 4), "z": round(float(z), 2),
                       "pct": round(percentile(his, vals), 1), "seasons": len(seas),
                       "consistency": None if cons is None else round(cons, 2),
                       "spain_rate": None if sp_rate is None else round(sp_rate, 4),
                       "spain_pct_in_tournament_pool": None if sp_pct is None else round(sp_pct, 1),
                       "pct_among_barca_midfielders": None if mates_pct is None else round(mates_pct, 1),
                       "barca_midfielders": mt.height, "signature": sig}

    print(f"\n{'metric':18s} {'rel':>5s} {'his':>8s} {'pool':>8s} {'z':>6s} {'pct':>5s} {'cons':>5s} {'Spain%':>7s} {'vsBarca%':>8s}  sig")
    for k, v in rows.items():
        if "z" not in v:
            print(f"{k:18s} {str(v['reliability']):>5s}  (not enough data: pool {v['pool_players']}, his {v['busquets']})")
            continue
        print(f"{k:18s} {str(v['reliability']):>5s} {v['busquets']:8.3f} {v['pool_mean']:8.3f} {v['z']:6.2f} {v['pct']:5.0f} "
              f"{str(v['consistency']):>5s} {str(v['spain_pct_in_tournament_pool']):>7s} {str(v['pct_among_barca_midfielders']):>8s}"
              f"  {'SIGNATURE' if v['signature'] else ''}")
    sigs = [k for k, v in rows.items() if v.get("signature")]
    print(f"\nsignature metrics: {sigs or 'none'}")

    # retrieval: holding-midfield player-team-season units, z-scored over the pool
    use = sigs or [k for k, v in rows.items() if v.get("reliability") and v["reliability"] >= 0.5 and "z" in v
                   and abs(v["z"]) >= 1.0]
    label = "signature" if sigs else "fallback: reliable metrics with |z| >= 1"
    units = dm.with_columns(unit=pl.format("{}|{}|{}", "player_id", "team_id", "key"))
    cnt = units.group_by("unit").agg(pl.len().alias("n"), pl.col("player_id").first(), pl.col("team_id").first(),
                                     pl.col("player").first(), pl.col("competition").first(), pl.col("season").first())
    cnt = cnt.filter(pl.col("n") >= 8)
    feats = {}
    for mname in use:
        num, den, kind = METRICS[mname]
        k = 90.0 if kind == "p90" else 1.0
        g = units.filter(pl.col(den).fill_null(0) > 0).group_by("unit").agg(
            (pl.col(num).fill_null(0).sum() / pl.col(den).sum() * k).alias(mname))
        cnt = cnt.join(g, on="unit", how="left")
    if use:
        X = cnt.select(use).to_numpy().astype(float)
        mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
        Z = np.nan_to_num((X - mu) / np.where(sd > 0, sd, 1))
        cp, cn = cnt["player_id"].to_list(), cnt["n"].to_list()
        cpl, ccomp, cseas = cnt["player"].to_list(), cnt["competition"].to_list(), cnt["season"].to_list()
        bi = [i for i, p in enumerate(cp) if p == BUSQUETS]
        best_b = max(bi, key=lambda i: cn[i])
        d = np.sqrt(((Z - Z[best_b]) ** 2).sum(1))
        order = [int(i) for i in np.argsort(d) if cp[int(i)] != BUSQUETS]
        seen, nn = set(), []
        for i in order:
            if cp[i] in seen:
                continue
            seen.add(cp[i])
            nn.append({"player": cpl[i], "competition": ccomp[i], "season": cseas[i],
                       "distance": round(float(d[i]), 2)})
            if len(nn) == 12:
                break
        print(f"\nnearest holding midfielders to Busquets ({cnt['season'][best_b]}) in the {label} space ({len(use)} metrics):")
        for r in nn:
            print(f"   {r['distance']:5.2f}  {r['player']}  ({r['competition']} {r['season']})")
        # Busquets' own other seasons: how highly does he rank himself?
        ranks = []
        for j in bi:
            if j == best_b:
                continue
            ranks.append(int((d < d[j]).sum()) - 1)
        print(f"rank of Busquets' other seasons among all units (0 = nearest): {sorted(ranks)}")
    else:
        nn, ranks = [], []

    # cross-team re-identification among holding midfielders: signature space vs all reliable candidates
    def space(cols):
        X = cnt.select(cols).to_numpy().astype(float)
        mu_, sd_ = np.nanmean(X, 0), np.nanstd(X, 0)
        return np.nan_to_num((X - mu_) / np.where(sd_ > 0, sd_, 1))

    for mname in METRICS:
        if mname not in cnt.columns:
            num, den, kind = METRICS[mname]
            k = 90.0 if kind == "p90" else 1.0
            g = units.filter(pl.col(den).fill_null(0) > 0).group_by("unit").agg(
                (pl.col(num).fill_null(0).sum() / pl.col(den).sum() * k).alias(mname))
            cnt = cnt.join(g, on="unit", how="left")
    reliable = [k for k, v in rows.items() if v.get("reliability") and v["reliability"] >= 0.5]
    pids, tids = cnt["player_id"].to_list(), cnt["team_id"].to_list()
    reid = {}
    for name, cols in (("signature_or_fallback", use), ("all_reliable_candidates", reliable)):
        if not cols:
            continue
        Z = space(cols)
        hits = n = 0
        for i in range(len(pids)):
            right = [j for j in range(len(pids)) if pids[j] == pids[i] and tids[j] != tids[i]]
            if not right:
                continue
            excl = {j for j in range(len(pids)) if pids[j] == pids[i] and tids[j] == tids[i]}
            d = np.sqrt(((Z - Z[i]) ** 2).sum(1))
            order = [j for j in np.argsort(d) if j not in excl]
            hits += any(j in right for j in order[:5])
            n += 1
        reid[name] = {"n": n, "top5": round(hits / n, 3) if n else None, "metrics": len(cols)}
    print(f"\ncross-team re-identification among holding midfielders (top-5): {reid}")
    Path("out/results").mkdir(parents=True, exist_ok=True)
    Path("out/results/analysis.json").write_text(json.dumps(
        {"metrics": rows, "signature": sigs, "retrieval_space": label, "retrieval_metrics": use,
         "nearest": nn, "self_ranks": ranks, "cross_team_reid": reid}, indent=1, default=str), encoding="utf-8")
