"""Role vectors: how a player plays, per player-team-season.

One vector per unit, made only of shares/compositions so a high-possession
team's player does not look different just from touching the ball more:

  act    where on the 12x8 pitch grid the player acts (on-ball events)
  rec    where they receive the ball
  dst    where their passes go
  def    where they defend (pressure, duel, interception, recovery, ...)
  style  pass length/direction mix and the composition of their actions

Coordinates are StatsBomb's, in the acting team's frame (attacking +x).
Each grid is a distribution (sums to 1).
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

GX, GY = 12, 8
CELLS = GX * GY
ON_BALL = ("Pass", "Carry", "Dribble", "Shot", "Ball Receipt*", "Duel", "Ball Recovery",
           "Interception", "Clearance", "Miscontrol", "Dispossessed", "Foul Won")
DEFENSIVE = ("Pressure", "Duel", "Interception", "Ball Recovery", "Clearance", "Block")
UNIT = ("player_id", "team_id", "competition_id", "season_id")
STYLE_NAMES = ("pass_len", "pass_fwd", "pass_long", "pass_share", "carry_share",
               "dribble_share", "shot_share", "def_share", "recv_share")
BLOCKS = {"act": CELLS, "rec": CELLS, "dst": CELLS, "def": CELLS, "style": len(STYLE_NAMES)}
LONG_PASS = 30.0
POSITION_GROUPS = {
    "Goalkeeper": "GK", "Center Back": "CB", "Left Center Back": "CB", "Right Center Back": "CB",
    "Left Back": "FB", "Right Back": "FB", "Left Wing Back": "FB", "Right Wing Back": "FB",
    "Center Defensive Midfield": "DM", "Left Defensive Midfield": "DM", "Right Defensive Midfield": "DM",
    "Center Midfield": "CM", "Left Center Midfield": "CM", "Right Center Midfield": "CM",
    "Left Midfield": "W", "Right Midfield": "W",
    "Center Attacking Midfield": "AM", "Left Attacking Midfield": "AM", "Right Attacking Midfield": "AM",
    "Left Wing": "W", "Right Wing": "W",
    "Center Forward": "ST", "Left Center Forward": "ST", "Right Center Forward": "ST",
    "Secondary Striker": "ST",
}


def cell_expr(xc: str, yc: str) -> pl.Expr:
    gx = (pl.col(xc) / (120.0 / GX)).floor().clip(0, GX - 1)
    gy = (pl.col(yc) / (80.0 / GY)).floor().clip(0, GY - 1)
    return (gy * GX + gx).cast(pl.Int32)


def _spell_minutes(s: str | None) -> float:
    """Minutes from a lineup 'positions' JSON; an open-ended spell counts as 90."""
    def mins(t):
        if not t:
            return None
        a = t.split(":")
        return int(a[0]) + int(a[1]) / 60.0

    total = 0.0
    for p in json.loads(s) if s else []:
        a, b = mins(p.get("from")), mins(p.get("to"))
        total += max(0.0, (90.0 if b is None else b) - (a or 0.0))
    return total


def minutes_by_unit(lineups: pl.DataFrame, matches: pl.DataFrame) -> pl.DataFrame:
    """Approximate minutes and match count per (player, team, competition, season)."""
    m = matches.select("match_id", "competition_id", "season_id")
    df = lineups.join(m, on="match_id", how="inner").with_columns(
        pl.col("positions").map_elements(_spell_minutes, return_dtype=pl.Float64).alias("minutes"))
    return df.group_by(*UNIT).agg(pl.col("minutes").sum(), pl.col("match_id").n_unique().alias("matches"))


def _grid(df: pl.DataFrame, types, xc: str, yc: str) -> pl.DataFrame:
    g = (df.filter(pl.col("type").is_in(types) & pl.col(xc).is_not_null() & pl.col(yc).is_not_null())
           .with_columns(cell=cell_expr(xc, yc))
           .group_by("unit", "cell").len())
    return g.with_columns(share=pl.col("len") / pl.col("len").sum().over("unit"))


def role_vectors(events: pl.DataFrame) -> tuple[list, np.ndarray, dict]:
    """Role vector for every `unit` in `events`.

    events needs: unit (hashable id), type, x, y, end_x, end_y.
    Returns (unit_ids, matrix [n_units, dim], blocks {name: slice}).
    """
    df = events.filter(pl.col("x").is_not_null())
    units = sorted(df["unit"].unique().to_list())
    index = {u: i for i, u in enumerate(units)}
    offsets, o = {}, 0
    for b, w in BLOCKS.items():
        offsets[b] = slice(o, o + w)
        o += w
    X = np.zeros((len(units), o))

    for name, types, xc, yc in (("act", ON_BALL, "x", "y"), ("rec", ("Ball Receipt*",), "x", "y"),
                                ("dst", ("Pass",), "end_x", "end_y"), ("def", DEFENSIVE, "x", "y")):
        g = _grid(df, types, xc, yc)
        base = offsets[name].start
        for u, c, s in zip(g["unit"].to_list(), g["cell"].to_list(), g["share"].to_list()):
            X[index[u], base + c] = s

    passes = df.filter((pl.col("type") == "Pass") & pl.col("end_x").is_not_null()).with_columns(
        dx=pl.col("end_x") - pl.col("x"), dy=pl.col("end_y") - pl.col("y"))
    passes = passes.with_columns(plen=(pl.col("dx") ** 2 + pl.col("dy") ** 2).sqrt())
    ps = passes.filter(pl.col("plen") > 1e-9).group_by("unit").agg(
        pl.col("plen").mean().alias("pass_len"),
        (pl.col("dx") / pl.col("plen")).mean().alias("pass_fwd"),
        (pl.col("plen") >= LONG_PASS).mean().alias("pass_long"))
    st = offsets["style"].start
    for u, a, b, c in ps.iter_rows():
        X[index[u], st:st + 3] = (a, b, c)

    comp = df.group_by("unit").agg(
        pl.len().alias("n"),
        (pl.col("type") == "Pass").sum().alias("pass"), (pl.col("type") == "Carry").sum().alias("carry"),
        (pl.col("type") == "Dribble").sum().alias("dribble"), (pl.col("type") == "Shot").sum().alias("shot"),
        pl.col("type").is_in(DEFENSIVE).sum().alias("deff"),
        (pl.col("type") == "Ball Receipt*").sum().alias("recv"))
    for r in comp.iter_rows(named=True):
        i = index[r["unit"]]
        for k, key in enumerate(("pass", "carry", "dribble", "shot", "deff", "recv")):
            X[i, st + 3 + k] = r[key] / r["n"]
    return units, X, offsets


def primary_group(events: pl.DataFrame) -> dict:
    """unit -> position group of the position the unit played most events in."""
    df = events.filter(pl.col("position").is_not_null())
    top = (df.group_by("unit", "position").len().sort("len", descending=True)
             .group_by("unit", maintain_order=True).first())
    return {u: POSITION_GROUPS.get(p, "OTHER") for u, p in zip(top["unit"].to_list(), top["position"].to_list())}


def embed(X: np.ndarray, offsets: dict, n_components: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Standardise, give every block equal total weight, then PCA.
    Returns (weighted standardised matrix, PCA scores). Unsupervised."""
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / np.where(sd > 1e-12, sd, 1.0)
    for sl in offsets.values():
        Z[:, sl] /= np.sqrt(sl.stop - sl.start)
    U, S, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    k = min(n_components, len(S))
    return Z, U[:, :k] * S[:k]
