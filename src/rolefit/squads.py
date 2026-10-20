"""Current squads from Wikipedia and Wikidata, matched to StatsBomb players.

Free and licensed for this use: Wikipedia text is CC BY-SA (attribution
required) and its API permits automated reads; Wikidata is CC0. Requests are
sequential, batched where the API allows, with a descriptive User-Agent and
`maxlag`, as Wikimedia's API etiquette asks.

Pipeline: season article -> clubs -> each club's current-squad section
(squad-player templates, stopping before loaned-out lists) -> each player's
Wikidata item (labels, aliases, birth name, birth date, citizenship) ->
match to StatsBomb lineups by normalised name, with citizenship as tie-breaker.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request

UA = "RoleFitResearch/0.1 (https://github.com/akabra13/rolefit; non-commercial football research)"
WP = "https://en.wikipedia.org/w/api.php"
WD = "https://www.wikidata.org/w/api.php"
PAUSE = 0.5
SQUAD_HEADINGS = ("current squad", "first-team squad", "first team squad", "players", "squad", "roster",
                  "current roster")
STOP_HEADINGS = ("out on loan", "on loan", "loan", "reserves", "academy", "youth", "former", "retired",
                 "notable", "player records", "staff", "management", "coaching")
PLAYER_TEMPLATE = re.compile(r"(fs player|football squad player|fs2 player|nat fs player)", re.I)


def _get(url: str, params: dict, retries: int = 4) -> dict:
    params = dict(params, format="json", maxlag=5)
    for i in range(retries):
        req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.load(r)
            if d.get("error", {}).get("code") == "maxlag":
                time.sleep(5 * (i + 1))
                continue
            time.sleep(PAUSE)
            return d
        except Exception:
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"request failed: {url} {params.get('page') or params.get('titles') or params.get('ids')}")


def wikitext(title: str) -> str | None:
    d = _get(WP, {"action": "parse", "page": title, "prop": "wikitext", "redirects": 1})
    return d.get("parse", {}).get("wikitext", {}).get("*")


def norm(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    return " ".join(re.sub(r"[^a-z ]+", " ", s).split())


def split_params(body: str) -> list[str]:
    """Split template parameters on top-level '|' (not inside [[...]] or {{...}})."""
    out, depth, cur, i = [], 0, [], 0
    while i < len(body):
        two = body[i:i + 2]
        if two in ("[[", "{{"):
            depth += 1; cur.append(two); i += 2; continue
        if two in ("]]", "}}"):
            depth -= 1; cur.append(two); i += 2; continue
        if body[i] == "|" and depth == 0:
            out.append("".join(cur)); cur = []; i += 1; continue
        cur.append(body[i]); i += 1
    out.append("".join(cur))
    return out


def templates(text: str) -> list[tuple[str, str]]:
    """Top-level templates as (name, body)."""
    res, i = [], 0
    while True:
        s = text.find("{{", i)
        if s < 0:
            return res
        depth, j = 0, s
        while j < len(text):
            if text.startswith("{{", j):
                depth += 1; j += 2
            elif text.startswith("}}", j):
                depth -= 1; j += 2
                if depth == 0:
                    break
            else:
                j += 1
        inner = text[s + 2:j - 2]
        parts = split_params(inner)
        res.append((parts[0].strip(), inner))
        i = j


def first_link(s: str) -> tuple[str | None, str]:
    m = re.search(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]", s or "")
    if not m:
        return None, re.sub(r"<[^>]+>|\{\{[^}]*\}\}", "", s or "").strip()
    return m.group(1).strip(), (m.group(2) or m.group(1)).strip()


def squad_section(text: str) -> str:
    """Text of the current-squad section, stopping at the next heading of the
    same or higher level, or at a loan/youth/staff subheading."""
    heads = [(m.start(), len(m.group(1)), m.group(2).strip().lower()) for m in
             re.finditer(r"^(=+)\s*(.*?)\s*=+\s*$", text, re.M)]
    for k, (pos, lvl, title) in enumerate(heads):
        if any(title.startswith(h) for h in SQUAD_HEADINGS) and not any(s in title for s in STOP_HEADINGS):
            end = len(text)
            for pos2, lvl2, title2 in heads[k + 1:]:
                if lvl2 <= lvl or any(s in title2 for s in STOP_HEADINGS):
                    end = pos2
                    break
            return text[pos:end]
    return ""


def player_templates(text: str) -> list[tuple[str, str]]:
    """Every squad-player template in `text`, at any nesting depth."""
    res = []
    for m in re.finditer(r"\{\{\s*(" + PLAYER_TEMPLATE.pattern[1:-1] + r")[^|}]*", text, re.I):
        s, depth, j = m.start(), 0, m.start()
        while j < len(text):
            if text.startswith("{{", j):
                depth += 1; j += 2
            elif text.startswith("}}", j):
                depth -= 1; j += 2
                if depth == 0:
                    break
            else:
                j += 1
        inner = text[s + 2:j - 2]
        res.append((split_params(inner)[0].strip(), inner))
    return res


def parse_squad(text: str) -> list[dict]:
    out = []
    for name, body in player_templates(squad_section(text)):
        f = {}
        for p in split_params(body)[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                f[k.strip().lower()] = v.strip()
        raw = f.get("name") or " ".join(x for x in (f.get("first", ""), f.get("last", "")) if x)
        title, shown = first_link(raw)
        if not title and f.get("link"):
            title = f["link"]
        if not shown:
            continue
        out.append({"pos": f.get("pos"), "nat": f.get("nat"), "no": f.get("no"), "title": title, "name": shown})
    return out


def season_clubs(text: str) -> list[str]:
    """Club article titles from a league-season article: first link of each
    row in the 'Teams' / 'Stadiums and locations' tables."""
    heads = [(m.start(), m.group(2).strip().lower()) for m in re.finditer(r"^(=+)\s*(.*?)\s*=+\s*$", text, re.M)]
    seg = ""
    for k, (pos, title) in enumerate(heads):
        if title.startswith(("teams", "stadiums", "clubs", "personnel", "team changes", "participating")):
            nxt = heads[k + 1][0] if k + 1 < len(heads) else len(text)
            seg += text[pos:nxt]
    clubs = []
    for row in re.split(r"\n\|-", seg):
        for line in row.split("\n"):
            line = line.strip()
            if line.startswith(("|", "!")) and "[[" in line:
                t, _ = first_link(line)
                if t and not t.lower().startswith(("file:", "image:")) and t not in clubs:
                    clubs.append(t)
                break
    return clubs


CLUB_WORDS = ("football club", "soccer club", "football team", "soccer team", "sports club",
              "association football", "women's association football")


def keep_clubs(titles: list[str]) -> list[str]:
    """Titles whose Wikidata item is an instance of some kind of football club."""
    q = wikidata_for_titles(titles)
    facts = entity_facts([v for v in q.values() if v])
    classes = sorted({c for f in facts.values() for c in f["instance"]})
    labels = {k: (v["label"] or "").lower() for k, v in entity_facts(classes).items()} if classes else {}
    keep = []
    for t in titles:
        f = facts.get(q.get(t))
        bad = ("season", "venue", "stadium", "league", "division", "competition", "cup", "tournament")
        if f and any(any(w in labels.get(c, "") for w in CLUB_WORDS) and not any(b in labels.get(c, "") for b in bad)
                     for c in f["instance"]):
            keep.append(t)
    return keep


def season_links(text: str) -> list[str]:
    """All wiki links in a league-season article's team/stadium/personnel sections."""
    heads = [(m.start(), m.group(2).strip().lower()) for m in re.finditer(r"^(=+)\s*(.*?)\s*=+\s*$", text, re.M)]
    seg = ""
    for k, (pos, title) in enumerate(heads):
        if title.startswith(("teams", "stadium", "clubs", "personnel", "participating", "location"))                 and "change" not in title:
            nxt = heads[k + 1][0] if k + 1 < len(heads) else len(text)
            seg += text[pos:nxt]
    out = []
    for line in seg.split("\n"):
        ls = line.strip()
        # table rows and map labels only; prose (e.g. who was relegated last season) is skipped
        if not (ls.startswith(("|", "!")) or "label=" in ls):
            continue
        for m in re.finditer(r"\[\[([^\]|#]+)", ls):
            t = m.group(1).strip()
            if not t.lower().startswith(("file:", "image:", "category:")) and t not in out:
                out.append(t)
    return out


def wikidata_for_titles(titles: list[str]) -> dict:
    """title -> Wikidata QID (via page props, following redirects)."""
    out = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        d = _get(WP, {"action": "query", "prop": "pageprops", "ppprop": "wikibase_item", "redirects": 1,
                      "titles": "|".join(chunk)})
        q = d.get("query", {})
        redirect = {r["from"]: r["to"] for r in q.get("redirects", [])}
        normalized = {r["from"]: r["to"] for r in q.get("normalized", [])}
        by_title = {p["title"]: p.get("pageprops", {}).get("wikibase_item") for p in q.get("pages", {}).values()}
        for t in chunk:
            t2 = normalized.get(t, t)
            t2 = redirect.get(t2, t2)
            out[t] = by_title.get(t2)
    return out


def entity_facts(qids: list[str]) -> dict:
    """QID -> {names: set, dob, citizenship: [QIDs], instance: [QIDs]}."""
    out = {}
    for i in range(0, len(qids), 50):
        d = _get(WD, {"action": "wbgetentities", "ids": "|".join(qids[i:i + 50]),
                      "props": "labels|aliases|claims", "languages": "en|es|de|fr|it|pt|nl"})
        for q, e in d.get("entities", {}).items():
            names = {v["value"] for v in e.get("labels", {}).values()}
            for al in e.get("aliases", {}).values():
                names |= {a["value"] for a in al}
            cl = e.get("claims", {})

            def vals(pid):
                res = []
                for c in cl.get(pid, []):
                    dv = c.get("mainsnak", {}).get("datavalue", {}).get("value")
                    if isinstance(dv, dict) and "id" in dv:
                        res.append(dv["id"])
                    elif isinstance(dv, dict) and "text" in dv:
                        res.append(dv["text"])
                    elif isinstance(dv, dict) and "time" in dv:
                        res.append(dv["time"])
                return res
            names |= set(vals("P1477"))           # birth name
            dob = (vals("P569") or [None])[0]
            out[q] = {"names": names, "dob": dob[1:11] if dob else None, "citizenship": vals("P27"),
                      "instance": vals("P31"), "label": e.get("labels", {}).get("en", {}).get("value")}
    return out
