"""Current squads (Wikipedia + Wikidata) matched to StatsBomb players.

Writes out/current/squads.parquet and out/current/squads_report.json. Needs the
lake's lineups and matches tables (for StatsBomb names, countries, gender).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl  # noqa: E402

import os  # noqa: E402
from rolefit.squads import entity_facts, keep_clubs, norm, parse_squad, season_links, wikidata_for_titles, wikitext  # noqa: E402

OUT = Path("out/current")
LEAGUES = [
    ("Premier League", "male", ["2026–27 Premier League", "2025–26 Premier League"]),
    ("La Liga", "male", ["2026–27 La Liga", "2025–26 La Liga"]),
    ("Serie A", "male", ["2026–27 Serie A", "2025–26 Serie A"]),
    ("Bundesliga", "male", ["2026–27 Bundesliga", "2025–26 Bundesliga"]),
    ("Ligue 1", "male", ["2026–27 Ligue 1", "2025–26 Ligue 1"]),
    ("Eredivisie", "male", ["2026–27 Eredivisie", "2025–26 Eredivisie"]),
    ("Primeira Liga", "male", ["2026–27 Primeira Liga", "2025–26 Primeira Liga"]),
    ("Major League Soccer", "male", ["2026 Major League Soccer season"]),
    ("Women's Super League", "female", ["2026–27 Women's Super League", "2025–26 Women's Super League"]),
    ("NWSL", "female", ["2026 National Women's Soccer League season"]),
    ("Liga F", "female", ["2026–27 Liga F", "2025–26 Liga F"]),
    ("Frauen-Bundesliga", "female", ["2026–27 Frauen-Bundesliga", "2025–26 Frauen-Bundesliga"]),
    ("Serie A Women", "female", ["2026–27 Serie A (women)", "2025–26 Serie A (women)"]),
    ("Première Ligue", "female", ["2026–27 Première Ligue", "2025–26 Première Ligue"]),
]
MIN_SQUAD = 11


def statsbomb_index(root: Path) -> tuple[dict, dict]:
    """normalised name -> list of (player_id, country, gender); player_id -> display name."""
    m = pl.read_parquet(str(root / "matches" / "*.parquet")).select("match_id", "gender")
    lu = pl.read_parquet(str(root / "lineups" / "*.parquet")).select(
        "match_id", "player_id", "player", "nickname", "country").join(m, on="match_id", how="left")
    lu = lu.unique(subset=["player_id"], keep="last")
    idx: dict = {}
    names = {}
    for r in lu.iter_rows(named=True):
        names[r["player_id"]] = r["nickname"] or r["player"]
        for n in {norm(r["player"]), norm(r["nickname"])} - {""}:
            idx.setdefault(n, []).append((r["player_id"], r["country"], r["gender"], set(norm(r["player"]).split())))
    return idx, names


def match(player: dict, facts: dict | None, idx: dict, gender: str, country_names: dict) -> tuple[int | None, str]:
    cands = {}
    keys = {norm(player["name"]), norm(player["title"] or "")}
    if facts:
        keys |= {norm(n) for n in facts["names"]}
    for k in keys - {""}:
        for pid, ctry, g, toks in idx.get(k, []):
            if g == gender:
                cands[pid] = ctry
    cit = {country_names.get(c, "").lower() for c in (facts or {}).get("citizenship", [])}
    if len(cands) > 1 and cit:
        cands = {p: c for p, c in cands.items() if (c or "").lower() in cit} or cands
    if len(cands) == 1:
        return next(iter(cands)), "name"
    if len(cands) > 1:
        return None, "ambiguous"
    # fallback: every token of the common name appears in a StatsBomb full name, same nationality
    toks = set(norm(player["name"]).split())
    if len(toks) >= 2 and cit:
        hits = {pid for lst in idx.values() for pid, ctry, g, full in lst
                if g == gender and toks <= full and (ctry or "").lower() in cit}
        if len(hits) == 1:
            return next(iter(hits)), "tokens"
    return None, "none"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    idx, sb_names = statsbomb_index(Path(os.environ["PITCHGRAPH_DATA"]) / "lake")
    rows, report = [], {}
    for league, gender, articles in LEAGUES:
        text, used = None, None
        for a in articles:
            text = wikitext(a)
            if text:
                used = a
                break
        if not text:
            print(f"{league}: no season article found"); report[league] = {"article": None}; continue
        clubs, seen_q = [], set()
        cq = wikidata_for_titles(keep_clubs(season_links(text)))
        for c, q in cq.items():          # redirects can list one club twice
            if q not in seen_q:
                seen_q.add(q)
                clubs.append(c)
        got = 0
        for club in clubs:
            ct = wikitext(club)
            squad = parse_squad(ct or "")
            if len(squad) < MIN_SQUAD:
                continue
            got += 1
            for p in squad:
                rows.append({"league": league, "gender": gender, "season_article": used, "club": club, **p})
        report[league] = {"article": used, "candidate_links": len(clubs), "clubs_with_squad": got}
        print(f"{league} ({used}): {got} clubs with squads of {len(clubs)} links [{time.time() - t0:.0f}s]", flush=True)

    sq = pl.DataFrame(rows)
    titles = sorted({t for t in sq["title"].to_list() if t})
    qid_of = wikidata_for_titles(titles)
    qids = sorted({q for q in qid_of.values() if q})
    facts = entity_facts(qids)
    cit_qids = sorted({c for f in facts.values() for c in f["citizenship"]})
    country_names = {q: f["label"] for q, f in entity_facts(cit_qids).items()} if cit_qids else {}
    out = []
    for r in sq.iter_rows(named=True):
        q = qid_of.get(r["title"]) if r["title"] else None
        f = facts.get(q) if q else None
        pid, how = match(r, f, idx, r["gender"], country_names)
        out.append({**r, "qid": q, "dob": f["dob"] if f else None,
                    "citizenship": ", ".join(country_names.get(c, c) for c in (f or {}).get("citizenship", [])) or None,
                    "statsbomb_player_id": pid, "statsbomb_name": sb_names.get(pid) if pid else None, "match": how})
    df = pl.DataFrame(out, infer_schema_length=None)
    df.write_parquet(OUT / "squads.parquet")
    summ = df.group_by("league").agg(pl.len().alias("players"), pl.col("club").n_unique().alias("clubs"),
                                     pl.col("statsbomb_player_id").is_not_null().sum().alias("matched"),
                                     (pl.col("match") == "ambiguous").sum().alias("ambiguous")).sort("league")
    print(summ.to_pandas().to_string(index=False))
    for lg, v in report.items():
        s = summ.filter(pl.col("league") == lg)
        if s.height:
            v.update(s.row(0, named=True))
    (OUT / "squads_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s")
