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
    # --- playstyle block (added 2026-10-14 after feedback that characterisation was too thin) ---
    "recv_def_share": "reception", "recv_att_share": "reception", "recv_wide_share": "reception",
    "recv_halfspace_share": "reception", "touch_box_p100": "reception",
    "dst_def_share": "destination", "dst_att_share": "destination", "dst_box_share": "destination",
    "dst_wide_share": "destination", "dst_halfspace_share": "destination",
    "pass_sideways_share": "destination", "pass_forward_share": "destination",
    "cutback_p100": "technique", "header_share": "technique", "left_foot_share": "technique",
    "backheel_p100": "technique", "up_touch_share": "technique",
    "carry_att_entry_p100": "carrying", "carry_box_entry_p100": "carrying", "carry_fwd": "carrying",
    "carry_speed": "carrying",
    "dribbled_past_p100o": "defending", "def_own_third_share": "defending", "def_att_third_share": "defending",
    "intercept_p100o": "defending", "recovery_p100o": "defending", "clearance_p100o": "defending",
    "block_p100o": "defending", "tackle_win_share": "defending", "press_att_share": "defending",
    "foul_won_p100": "involvement", "corner_p100": "setpiece", "fk_pass_p100": "setpiece", "throwin_p100": "setpiece",
    "fk_shot_p100": "setpiece", "cards_p100o": "discipline",
    "poss_touch_share": "sequence", "xgchain_p100": "sequence", "xgbuildup_p100": "sequence",
    "shot_box_share": "finishing", "header_shot_share": "finishing", "goals_p100": "finishing",
    "finishing_oe": "finishing",
    # --- tempo and press resistance (added 2026-10-19: what separates a Busquets from a Coquelin) ---
    "one_touch_share": "tempo", "hold_time": "tempo", "up_pass_cmp": "press", "up_retain": "press",
    "recv_space": "press", "recv_crowd": "press",
}

EVENT_COLS = ["match_id", "match_date", "event_id", "player_id", "team_id", "possession",
              "possession_team_id", "period", "minute", "second", "type", "sub_type", "outcome",
              "x", "y", "end_x", "end_y", "under_pressure", "counterpress", "height", "body_part",
              "xg", "play_pattern", "position", "duration", "idx", "period_seconds", "competition_id", "season_id"]


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
        f_cutback=_has('"cut_back": true'),
        f_backheel=_has('"backheel": true'),
        extra_card=_has('"card"') & pl.col("type").is_in(["Foul Committed", "Bad Behaviour"]),
    ).select(EVENT_COLS + ["f_cross", "f_switch", "f_through", "f_assist", "f_aerial_won", "f_cutback",
                           "f_backheel", "extra_card"])


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


IN_POSS_TYPES = ("Pass", "Carry", "Ball Receipt*", "Dribble", "Shot", "Miscontrol", "Dispossessed")
HOLD_CAP_S = 15.0
CROWD_RADIUS = 5.0


def tempo_columns(ev: pl.DataFrame) -> pl.DataFrame:
    """Per event: one_touch (a pass whose previous in-possession event was the
    same player's ball receipt, i.e. no carry in between) and hold_s (seconds
    from the player's last receipt in the possession to a pass or shot,
    capped). Computed on the team's in-possession event stream in match order."""
    s = ev.select("event_id", "match_id", "idx", "possession", "player_id", "team_id", "possession_team_id",
                  "type", "period", "period_seconds").sort("match_id", "idx")
    s = s.filter((pl.col("team_id") == pl.col("possession_team_id")) & pl.col("type").is_in(IN_POSS_TYPES))
    s = s.with_columns(
        _pt=pl.col("type").shift(1).over("match_id"), _pp=pl.col("player_id").shift(1).over("match_id"),
        _pper=pl.col("period").shift(1).over("match_id"),
        _rt=pl.when(pl.col("type") == "Ball Receipt*").then(pl.col("period_seconds")))
    s = s.with_columns(_rt=pl.col("_rt").forward_fill().over("match_id", "possession", "player_id", "period"))
    release = pl.col("type").is_in(["Pass", "Shot"])
    s = s.select(
        "event_id",
        one_touch=(pl.col("type") == "Pass") & (pl.col("_pt") == "Ball Receipt*") & (pl.col("_pp") == pl.col("player_id"))
        & (pl.col("_pper") == pl.col("period")),
        hold_s=pl.when(release & pl.col("_rt").is_not_null())
        .then((pl.col("period_seconds") - pl.col("_rt")).clip(0.0, HOLD_CAP_S)))
    ev = ev.drop([c for c in ("one_touch", "hold_s") if c in ev.columns])
    return ev.join(s, on="event_id", how="left").with_columns(pl.col("one_touch").fill_null(False))


def receipt_space(ev: pl.DataFrame, frames: pl.DataFrame | None) -> pl.DataFrame:
    """360 only: at each successful ball receipt with a freeze-frame, the
    distance to the nearest visible opponent (recv_space_e) and how many
    opponents are within CROWD_RADIUS (recv_crowd_e). Null elsewhere."""
    if frames is None or frames.is_empty():
        ev = ev.drop([c for c in ("recv_space_e", "recv_crowd_e") if c in ev.columns])
        return ev.with_columns(recv_space_e=pl.lit(None, pl.Float64), recv_crowd_e=pl.lit(None, pl.Float64))
    rec = ev.filter((pl.col("type") == "Ball Receipt*") & pl.col("outcome").is_null() & pl.col("x").is_not_null())             .select("event_id", pl.col("x").alias("rx"), pl.col("y").alias("ry"))
    opp = frames.filter(~pl.col("teammate")).select("event_id", "x", "y")
    j = rec.join(opp, on="event_id").with_columns(
        d=((pl.col("x") - pl.col("rx")) ** 2 + (pl.col("y") - pl.col("ry")) ** 2).sqrt())
    g = j.group_by("event_id").agg(pl.col("d").min().alias("recv_space_e"),
                                   (pl.col("d") < CROWD_RADIUS).sum().cast(pl.Float64).alias("recv_crowd_e"))
    ev = ev.drop([c for c in ("recv_space_e", "recv_crowd_e") if c in ev.columns])
    return ev.join(g, on="event_id", how="left")


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
    ).with_columns(**_playstyle_columns())


def _playstyle_columns() -> dict:
    t = pl.col("type")
    sub = pl.col("sub_type").fill_null("")
    x, y = pl.col("x"), pl.col("y")
    ex, ey = pl.col("end_x"), pl.col("end_y")
    open_pass = (t == "Pass") & ~sub.is_in(RESTARTS)
    own = pl.col("team_id") == pl.col("possession_team_id")
    box = lambda a, b: (a >= 102) & b.is_between(18, 62)  # noqa: E731
    lane_wide = lambda b: (b < 18) | (b > 62)  # noqa: E731
    lane_half = lambda b: b.is_between(18, 30, closed="left") | b.is_between(50, 62, closed="right")  # noqa: E731
    recv = (t == "Ball Receipt*") & pl.col("outcome").is_null()
    shot = (t == "Shot") & (sub != "Penalty")
    carry = (t == "Carry") & ex.is_not_null()
    dx, dy = ex - x, ey - y
    plen = (dx ** 2 + dy ** 2).sqrt()
    defn = t.is_in(DEFENSIVE)
    foot_pass = open_pass & pl.col("body_part").is_in(["Left Foot", "Right Foot"])
    return dict(
        recv_def=recv & (x < 40), recv_att=recv & (x >= 80), recv_wide=recv & lane_wide(y),
        recv_half=recv & lane_half(y), touch_box=pl.col("touch") & box(x, y),
        dst_def=open_pass & (ex < 40), dst_att=open_pass & (ex >= 80), dst_box=open_pass & box(ex, ey),
        dst_wide=open_pass & lane_wide(ey), dst_half=open_pass & lane_half(ey),
        sideways=open_pass & (dy.abs() > dx.abs()), forward=open_pass & (dx > 0.5 * plen) & (plen > 0),
        cutback=pl.col("f_cutback"), backheel=pl.col("f_backheel"),
        header=pl.col("touch") & (pl.col("body_part") == "Head"),
        foot_pass=foot_pass, left_pass=foot_pass & (pl.col("body_part") == "Left Foot"),
        up_touch=pl.col("touch") & pl.col("under_pressure"),
        carry_att=carry & (ex >= 80) & (x < 80), carry_box=carry & box(ex, ey) & ~box(x, y),
        carry_dx=pl.when(carry).then(dx),
        carry_spd=pl.when(carry & (pl.col("duration") > 0.5)).then(plen / pl.col("duration")),
        dpast=t == "Dribbled Past",
        def_own=defn & (x < 40), def_att=defn & (x >= 80),
        intercept=t == "Interception", recovery=t == "Ball Recovery", clearance=t == "Clearance",
        blk=t == "Block", press_att=(t == "Pressure") & (x >= 80), foul_won=t == "Foul Won",
        tackle_won=(t == "Duel") & (sub == "Tackle") & pl.col("outcome").is_in(["Won", "Success In Play", "Success Out"]),
        shot_box=shot & box(x, y), header_shot=shot & (pl.col("body_part") == "Head"),
        goal=shot & (pl.col("outcome") == "Goal"),
        touch_own=pl.col("touch") & own & t.is_in(["Pass", "Carry", "Dribble", "Shot", "Ball Receipt*",
                                                    "Miscontrol", "Dispossessed"]),
        poss_xg=pl.when(own).then(pl.when(shot).then(pl.col("xg").fill_null(0.0))).sum().over("match_id", "possession"),
        corner=(t == "Pass") & (sub == "Corner"), fk_pass=(t == "Pass") & (sub == "Free Kick"),
        throw=(t == "Pass") & (sub == "Throw-in"), fk_shot=shot & (sub == "Free Kick"),
        card=pl.col("extra_card"),
    )

def aggregate(ev: pl.DataFrame, onp: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Traits per group `by`. `ev` from add_derived (plus `value`, `xp`
    columns, nullable) carrying the `by` columns; `onp` carries `by` plus
    own_poss, opp_poss, minutes, already summed or per match."""
    s = lambda c: pl.col(c).sum()  # noqa: E731
    keys = by + ["match_id", "possession"]
    ev = ev.with_columns(
        _ft=pl.col("touch_own") & (pl.col("touch_own").cast(pl.Int8).cum_sum().over(keys) == 1),
        _own_shot=(pl.col("shot") | pl.col("assist")).any().over(keys))
    g = ev.group_by(by).agg(
        s("recv_def").alias("n_recv_def"), s("recv_att").alias("n_recv_att"), s("recv_wide").alias("n_recv_wide"),
        s("recv_half").alias("n_recv_half"), s("touch_box").alias("n_tbox"),
        s("dst_def").alias("n_dst_def"), s("dst_att").alias("n_dst_att"), s("dst_box").alias("n_dst_box"),
        s("dst_wide").alias("n_dst_wide"), s("dst_half").alias("n_dst_half"),
        s("sideways").alias("n_side"), s("forward").alias("n_fwd"), s("cutback").alias("n_cutback"),
        s("backheel").alias("n_backheel"), s("header").alias("n_header"), s("foot_pass").alias("n_foot"),
        s("left_pass").alias("n_left"), s("up_touch").alias("n_up"),
        s("carry_att").alias("n_carry_att"), s("carry_box").alias("n_carry_box"),
        pl.col("carry_dx").mean().alias("carry_fwd"), pl.col("carry_spd").mean().alias("carry_speed"),
        s("dpast").alias("n_dpast"), s("def_own").alias("n_def_own"), s("def_att").alias("n_def_att"),
        s("intercept").alias("n_int"), s("recovery").alias("n_rec"), s("clearance").alias("n_clr"),
        s("blk").alias("n_blk"), s("press_att").alias("n_press_att"), s("foul_won").alias("n_fw"),
        s("tackle_won").alias("n_tackle_won"), s("shot_box").alias("n_shot_box"),
        s("header_shot").alias("n_header_shot"), s("goal").alias("n_goal"),
        s("corner").alias("n_corner"), s("fk_pass").alias("n_fk"), s("throw").alias("n_throw"),
        s("fk_shot").alias("n_fk_shot"), s("card").alias("n_card"),
        (pl.col("one_touch") & pl.col("open_pass")).sum().alias("n_one_touch"),
        pl.col("hold_s").mean().alias("hold_time"),
        (pl.col("open_pass") & pl.col("under_pressure")).sum().alias("n_up_pass"),
        (pl.col("ok_open") & pl.col("under_pressure")).sum().alias("n_up_pass_ok"),
        (pl.col("loss") & pl.col("under_pressure")).sum().alias("n_up_loss"),
        pl.col("recv_space_e").mean().alias("recv_space"), pl.col("recv_crowd_e").mean().alias("recv_crowd"),
        s("_ft").alias("n_ptouch"), (pl.col("poss_xg") * pl.col("_ft")).sum().alias("xgchain_sum"),
        (pl.col("poss_xg") * (pl.col("_ft") & ~pl.col("_own_shot"))).sum().alias("xgbuild_sum"),
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
        recv_def_share=share("n_recv_def", "n_recv"), recv_att_share=share("n_recv_att", "n_recv"),
        recv_wide_share=share("n_recv_wide", "n_recv"), recv_halfspace_share=share("n_recv_half", "n_recv"),
        touch_box_p100=per("n_tbox", "own_poss"),
        dst_def_share=share("n_dst_def", "n_open"), dst_att_share=share("n_dst_att", "n_open"),
        dst_box_share=share("n_dst_box", "n_open"), dst_wide_share=share("n_dst_wide", "n_open"),
        dst_halfspace_share=share("n_dst_half", "n_open"), pass_sideways_share=share("n_side", "n_open"),
        pass_forward_share=share("n_fwd", "n_open"), cutback_p100=per("n_cutback", "own_poss"),
        header_share=share("n_header", "n_touch"), left_foot_share=share("n_left", "n_foot"),
        backheel_p100=per("n_backheel", "own_poss"), up_touch_share=share("n_up", "n_touch"),
        carry_att_entry_p100=per("n_carry_att", "own_poss"), carry_box_entry_p100=per("n_carry_box", "own_poss"),
        dribbled_past_p100o=per("n_dpast", "opp_poss"),
        def_own_third_share=share("n_def_own", "n_def"), def_att_third_share=share("n_def_att", "n_def"),
        intercept_p100o=per("n_int", "opp_poss"), recovery_p100o=per("n_rec", "opp_poss"),
        clearance_p100o=per("n_clr", "opp_poss"), block_p100o=per("n_blk", "opp_poss"),
        tackle_win_share=share("n_tackle_won", "n_tackle"), press_att_share=share("n_press_att", "n_pressure"),
        foul_won_p100=per("n_fw", "own_poss"),
        poss_touch_share=pl.when(pl.col("own_poss") > 0).then(pl.col("n_ptouch") / pl.col("own_poss")),
        xgchain_p100=per("xgchain_sum", "own_poss"), xgbuildup_p100=per("xgbuild_sum", "own_poss"),
        shot_box_share=share("n_shot_box", "n_shot"), header_shot_share=share("n_header_shot", "n_shot"),
        goals_p100=per("n_goal", "own_poss"),
        finishing_oe=pl.when(pl.col("n_shot") > 0).then((pl.col("n_goal") - pl.col("xg_sum")) / pl.col("n_shot")),
        corner_p100=per("n_corner", "own_poss"), fk_pass_p100=per("n_fk", "own_poss"),
        throwin_p100=per("n_throw", "own_poss"), fk_shot_p100=per("n_fk_shot", "own_poss"),
        cards_p100o=per("n_card", "opp_poss"),
        one_touch_share=share("n_one_touch", "n_open"),
        up_pass_cmp=share("n_up_pass_ok", "n_up_pass"),
        up_retain=pl.when(pl.col("n_up") > 0).then(1.0 - (pl.col("n_up_loss") + pl.col("n_up_pass") - pl.col("n_up_pass_ok"))
                                                 / pl.col("n_up")),
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
