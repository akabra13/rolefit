"""Pairs of units belonging to the same player: the natural experiments that
separate a player's habits from their team's system.

  same_team     same player, same team, another season or competition
                (the ceiling: how stable is a trait with the system held fixed)
  club_country  one club unit and one national-team unit
  transfer      two different club teams
  other_team    two different national teams (e.g. youth and senior sides)
"""
from __future__ import annotations

from itertools import combinations

import polars as pl

INTERNATIONAL = frozenset({
    "FIFA World Cup", "UEFA Euro", "Women's World Cup", "UEFA Women's Euro",
    "Copa America", "African Cup of Nations", "FIFA U20 World Cup",
})
CROSS_TEAM = ("club_country", "transfer", "other_team")


def is_international(competition: str | None) -> bool:
    return competition in INTERNATIONAL


def pair_type(team_a, team_b, intl_a: bool, intl_b: bool) -> str:
    if team_a == team_b:
        return "same_team"
    if intl_a != intl_b:
        return "club_country"
    return "other_team" if intl_a else "transfer"


def mover_pairs(units: pl.DataFrame, min_minutes: float) -> pl.DataFrame:
    """All unordered pairs of units of the same player, both with at least
    `min_minutes`. `units` needs unit, player_id, team_id, competition, minutes."""
    u = units.filter(pl.col("minutes") >= min_minutes).select(
        "unit", "player_id", "team_id", "competition").sort("unit")
    rows = []
    for (pid,), g in u.group_by(["player_id"]):
        recs = g.iter_rows(named=True)
        for a, b in combinations(list(recs), 2):
            rows.append((pid, a["unit"], b["unit"],
                         pair_type(a["team_id"], b["team_id"],
                                   is_international(a["competition"]), is_international(b["competition"]))))
    return pl.DataFrame(rows, schema=["player_id", "unit_a", "unit_b", "type"], orient="row")
