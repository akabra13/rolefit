"""Player traits per player-team-season unit (and per half of its matches).

Every trait is a share, a mean, an over-expected residual, or a rate per
100 possessions *while the player was on the pitch* (own possessions for
on-ball traits, opponent possessions for defending). Raw counts are never
traits, so a player on a high-possession team, or one who played more
minutes, does not look different for that reason alone.

Coordinates are StatsBomb's, in the acting team's frame (attacking +x).
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

RESTARTS = ("Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in")
ON_BALL = ("Pass", "Carry", "Dribble", "Shot", "Ball Receipt*", "Miscontrol", "Dispossessed",
           "Ball Recovery", "Interception", "Clearance", "Duel", "Foul Won")
DEFENSIVE = ("Pressure", "Duel", "Interception", "Ball Recovery", "Clearance", "Block")
UNIT = ("player_id", "team_id", "competition_id", "season_id")
LONG_PASS = 30.0
PERIOD_KEY = 100_000

# trait -> family. Order is the reporting order.
TRAITS = {
    "avg_x": "territory", "avg_y": "territory", "width": "territory", "x_spread": "territory",
    "y_spread": "territory", "def_height": "territory",
    "touches_p100": "involvement", "recv_p100": "involvement", "pass_share": "involvement",
    "prog_pass_p100": "progression", "prog_carry_p100": "progression", "value_p100": "progression",
    "value_per_action": "progression",
    "shot_assist_p100": "creation", "box_entry_p100": "creation", "cross_p100": "creation",
    "through_p100": "creation",
    "pass_cmp": "retention", "pass_cmp_oe": "retention", "loss_per_touch": "retention",
    "pressured_pass_share": "retention",
    "pass_len": "passing", "pass_fwd": "passing", "long_share": "passing", "switch_share": "passing",
    "back_share": "passing", "cross_share": "passing", "high_share": "passing",
    "carry_p100": "carrying", "carry_len": "carrying", "takeon_p100": "carrying",
    "takeon_success": "carrying",
    "shots_p100": "finishing", "xg_per_shot": "finishing", "shot_dist": "finishing",
    "xg_p100": "finishing",
    "pressures_p100o": "defending", "regains_p100o": "defending", "tackles_p100o": "defending",
    "aerials_p100": "defending", "aerial_win": "defending", "counterpress_share": "defending",
    "fouls_p100o": "defending",
}

EVENT_COLS = ["match_id", "match_date", "event_id", "player_id", "team_id", "possession",
              "possession_team_id", "period", "minute", "second", "type", "sub_type", "outcome",
              "x", "y", "end_x", "end_y", "under_pressure", "counterpress", "height", "body_part",
              "xg", "play_pattern", "position", "competition_id", "season_id"]


def _has(flag: str) -> pl.Expr:
    return pl.col("extra").fill_null("").str.contains(flag, literal=True)


def event_flags(events: pl.LazyFrame) -> pl.LazyFrame:
    """Select the columns traits need and turn the JSON `extra` payload into
    boolean flags (so the large string column can be dropped early)."""
    return events.with_columns(
        f_cross=_has('"cross": true'),
        f_switch=_has('"switch": true'),
        f_through=_has('"through_ball": true') | _has('"name": "Through Ball"'),
        f_assist=_has('"shot_assist": true') | _has('"goal_assist": true'),
        f_aerial_won=_has('"aerial_won": true'),
    ).select(EVENT_COLS + ["f_cross", "f_switch", "f_through", "f_assist", "f_aerial_won"])


def _clock_key(period: int | None, mmss: str | None, default: int) -> int:
    if not mmss or period is None:
        return default
    m, s = mmss.split(":")
    return int(period) * PERIOD_KEY + int(m) * 60 + int(s)


def spells(lineups: pl.DataFrame) -> pl.DataFrame:
    """One row per on-pitch spell: match_id, team_id, player_id, start, end
    (period-keyed clock), minutes."""
    rows = []
    for mid, tid, pid, pos in lineups.select("match_id", "team_id", "player_id", "positions").iter_rows():
        for p in json.loads(pos) if pos else []:
            a = _clock_key(p.get("from_period"), p.get("from"), 0)
            b = _clock_key(p.get("to_period"), p.get("to"), 10 * PERIOD_KEY)
            fm = int(p["from"].split(":")[0]) if p.get("from") else 0
            tm = int(p["to"].split(":")[0]) if p.get("to") else 90
            rows.append((mid, tid, pid, a, b, max(0.0, float(tm - fm))))
    return pl.DataFrame(rows, schema=["match_id", "team_id", "player_id", "start", "end", "minutes"],
                        orient="row")


def on_pitch_possessions(events: pl.DataFrame, sp: pl.DataFrame) -> pl.DataFrame:
    """Per (match, team, player): own and opponent possessions that started
    while the player was on the pitch, plus minutes."""
    poss = (events.filter(pl.col("period") <= 4)
                  .group_by("match_id", "possession")
                  .agg(pl.col("possession_team_id").first(),
                       (pl.col("period").first() * PERIOD_KEY + pl.col("minute").first() * 60
                        + pl.col("second").first()).alias("t")))
    j = sp.join(poss, on="match_id").filter((pl.col("t") >= pl.col("start")) & (pl.col("t") < pl.col("end")))
    counts = j.group_by("match_id", "team_id", "player_id").agg(
        (pl.col("possession_team_id") == pl.col("team_id")).sum().alias("own_poss"),
        (pl.col("possession_team_id") != pl.col("team_id")).sum().alias("opp_poss"))
    mins = sp.group_by("match_id", "team_id", "player_id").agg(pl.col("minutes").sum())
    return mins.join(counts, on=["match_id", "team_id", "player_id"], how="left").fill_null(0)


def add_derived(ev: pl.DataFrame) -> pl.DataFrame:
    """Per-event indicator and measurement columns used by `aggregate`."""
    t = pl.col("type")
    sub = pl.col("sub_type").fill_null("")
    is_pass = t == "Pass"
    open_pass = is_pass & ~sub.is_in(RESTARTS)
    ok_pass = is_pass & pl.col("outcome").is_null()
    dx, dy = pl.col("end_x") - pl.col("x"), pl.col("end_y") - pl.col("y")
    plen = (dx ** 2 + dy ** 2).sqrt()
    d0 = ((120 - pl.col("x")) ** 2 + (40 - pl.col("y")) ** 2).sqrt()
    d1 = ((120 - pl.col("end_x")) ** 2 + (40 - pl.col("end_y")) ** 2).sqrt()
    in_box = lambda xc, yc: (pl.col(xc) >= 102) & pl.col(yc).is_between(18, 62)  # noqa: E731
    move = (ok_pass & open_pass) | ((t == "Carry") & pl.col("end_x").is_not_null())
    shot = (t == "Shot") & (sub != "Penalty")
    return ev.with_columns(
        touch=t.is_in(ON_BALL), recv=(t == "Ball Receipt*") & pl.col("outcome").is_null(),
        is_pass=is_pass, open_pass=open_pass, ok_open=ok_pass & open_pass,
        plen=pl.when(is_pass & pl.col("end_x").is_not_null()).then(plen),
        pcos=pl.when(is_pass & (plen > 0)).then(dx / plen),
        prog_pass=ok_pass & open_pass & (d1 <= 0.75 * d0) & (pl.col("end_x") >= 60),
        prog_carry=(t == "Carry") & (d1 <= 0.75 * d0) & (pl.col("end_x") >= 60) & (dx >= 5),
        box_entry=move & in_box("end_x", "end_y") & ~in_box("x", "y"),
        is_carry=t == "Carry", carry_len=pl.when(t == "Carry").then(plen),
        takeon=t == "Dribble", takeon_ok=(t == "Dribble") & (pl.col("outcome") == "Complete"),
        loss=t.is_in(["Miscontrol", "Dispossessed"]) | ((t == "Dribble") & (pl.col("outcome") == "Incomplete")),
        shot=shot, shot_xg=pl.when(shot).then(pl.col("xg").fill_null(0.0)),
        shot_dist=pl.when(shot).then(d0),
        pressure=t == "Pressure", cp=(t == "Pressure") & pl.col("counterpress"),
        regain=t.is_in(["Ball Recovery", "Interception"]),
        tackle=(t == "Duel") & (sub == "Tackle"),
        aerial_lost=(t == "Duel") & (sub == "Aerial Lost"),
        aerial_won=pl.col("f_aerial_won"),
        foul=t == "Foul Committed",
        defn=t.is_in(DEFENSIVE), def_x=pl.when(t.is_in(DEFENSIVE)).then(pl.col("x")),
        tx=pl.when(t.is_in(ON_BALL)).then(pl.col("x")), ty=pl.when(t.is_in(ON_BALL)).then(pl.col("y")),
        cross=open_pass & pl.col("f_cross"), switch=open_pass & pl.col("f_switch"),
        through=open_pass & pl.col("f_through"), assist=is_pass & pl.col("f_assist"),
        long=open_pass & (plen >= LONG_PASS), back=open_pass & (dx < 0),
        high=open_pass & (pl.col("height") == "High Pass"),
        pressured_pass=open_pass & pl.col("under_pressure"),
    )


def aggregate(ev: pl.DataFrame, onp: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Traits per group `by`. `ev` from add_derived (plus `value`, `xp`
    columns, nullable) carrying the `by` columns; `onp` carries `by` plus
    own_poss, opp_poss, minutes, already summed or per match."""
    s = lambda c: pl.col(c).sum()  # noqa: E731
    g = ev.group_by(by).agg(
        s("touch").alias("n_touch"), s("recv").alias("n_recv"), s("is_pass").alias("n_pass"),
        s("open_pass").alias("n_open"), s("ok_open").alias("n_ok_open"),
        pl.col("tx").mean().alias("avg_x"), pl.col("ty").mean().alias("avg_y"),
        (pl.col("ty") - 40).abs().mean().alias("width"),
        pl.col("tx").std().alias("x_spread"), pl.col("ty").std().alias("y_spread"),
        pl.col("def_x").mean().alias("def_height"),
        s("prog_pass").alias("n_prog_pass"), s("prog_carry").alias("n_prog_carry"),
        pl.col("value").sum().alias("value_sum"), pl.col("value").is_not_null().sum().alias("n_valued"),
        s("assist").alias("n_assist"), s("box_entry").alias("n_box"), s("cross").alias("n_cross"),
        s("through").alias("n_through"), s("switch").alias("n_switch"), s("long").alias("n_long"),
        s("back").alias("n_back"), s("high").alias("n_high"), s("pressured_pass").alias("n_pp"),
        pl.when(pl.col("open_pass")).then(pl.col("ok_open").cast(pl.Float64) - pl.col("xp"))
          .mean().alias("pass_cmp_oe"),
        s("loss").alias("n_loss"),
        pl.col("plen").filter(pl.col("open_pass")).mean().alias("pass_len"),
        pl.col("pcos").filter(pl.col("open_pass")).mean().alias("pass_fwd"),
        s("is_carry").alias("n_carry"), pl.col("carry_len").mean().alias("carry_len"),
        s("takeon").alias("n_takeon"), s("takeon_ok").alias("n_takeon_ok"),
        s("shot").alias("n_shot"), pl.col("shot_xg").sum().alias("xg_sum"),
        pl.col("shot_dist").mean().alias("shot_dist"),
        s("pressure").alias("n_pressure"), s("cp").alias("n_cp"), s("regain").alias("n_regain"),
        s("tackle").alias("n_tackle"), s("aerial_lost").alias("n_al"), s("aerial_won").alias("n_aw"),
        s("foul").alias("n_foul"), s("defn").alias("n_def"),
    )
    p = onp.group_by(by).agg(pl.col("own_poss").sum(), pl.col("opp_poss").sum(), pl.col("minutes").sum(),
                             pl.col("match_id").n_unique().alias("matches"))
    g = g.join(p, on=by, how="inner")

    def per(col, den, k=100.0):
        return pl.when(pl.col(den) > 0).then(k * pl.col(col) / pl.col(den))

    def share(col, den):
        return pl.when(pl.col(den) > 0).then(pl.col(col) / pl.col(den))

    out = g.with_columns(
        touches_p100=per("n_touch", "own_poss"), recv_p100=per("n_recv", "own_poss"),
        pass_share=share("n_pass", "n_touch"),
        prog_pass_p100=per("n_prog_pass", "own_poss"), prog_carry_p100=per("n_prog_carry", "own_poss"),
        value_p100=per("value_sum", "own_poss"), value_per_action=share("value_sum", "n_valued"),
        shot_assist_p100=per("n_assist", "own_poss"), box_entry_p100=per("n_box", "own_poss"),
        cross_p100=per("n_cross", "own_poss"), through_p100=per("n_through", "own_poss"),
        pass_cmp=share("n_ok_open", "n_open"), loss_per_touch=share("n_loss", "n_touch"),
        pressured_pass_share=share("n_pp", "n_open"),
        long_share=share("n_long", "n_open"), switch_share=share("n_switch", "n_open"),
        back_share=share("n_back", "n_open"), cross_share=share("n_cross", "n_open"),
        high_share=share("n_high", "n_open"),
        carry_p100=per("n_carry", "own_poss"), takeon_p100=per("n_takeon", "own_poss"),
        takeon_success=share("n_takeon_ok", "n_takeon"),
        shots_p100=per("n_shot", "own_poss"), xg_per_shot=share("xg_sum", "n_shot"),
        xg_p100=per("xg_sum", "own_poss"),
        pressures_p100o=per("n_pressure", "opp_poss"), regains_p100o=per("n_regain", "opp_poss"),
        tackles_p100o=per("n_tackle", "opp_poss"), fouls_p100o=per("n_foul", "opp_poss"),
        aerials_p100=pl.when((pl.col("own_poss") + pl.col("opp_poss")) > 0)
          .then(100.0 * (pl.col("n_al") + pl.col("n_aw")) / (pl.col("own_poss") + pl.col("opp_poss"))),
        aerial_win=pl.when((pl.col("n_al") + pl.col("n_aw")) > 0)
          .then(pl.col("n_aw") / (pl.col("n_al") + pl.col("n_aw"))),
        counterpress_share=share("n_cp", "n_pressure"),
    )
    return out.select(by + ["minutes", "matches", "own_poss", "opp_poss", "n_touch"] + list(TRAITS))


def split_half_reliability(halves: pl.DataFrame, unit_col: str, group_col: str,
                           traits=tuple(TRAITS)) -> dict:
    """Per trait: Pearson r between half A and half B across units (overall)
    and after removing each position group's mean (within-group, the
    stricter one: can it tell apart two players in the same role?)."""
    a = halves.filter(pl.col("half") == 0)
    b = halves.filter(pl.col("half") == 1)
    j = a.join(b, on=[unit_col, group_col], suffix="_b")
    out = {}
    for t in traits:
        d = j.select(group_col, pl.col(t).alias("a"), pl.col(f"{t}_b").alias("b")).drop_nulls()
        d = d.filter(pl.col("a").is_finite() & pl.col("b").is_finite())
        if d.height < 20:
            out[t] = {"n": d.height, "r": None, "r_within": None}
            continue
        r = float(np.corrcoef(d["a"].to_numpy(), d["b"].to_numpy())[0, 1])
        w = d.with_columns(a=pl.col("a") - pl.col("a").mean().over(group_col),
                           b=pl.col("b") - pl.col("b").mean().over(group_col))
        rw = float(np.corrcoef(w["a"].to_numpy(), w["b"].to_numpy())[0, 1])
        out[t] = {"n": d.height, "r": round(r, 3), "r_within": round(rw, 3)}
    return out
