"""Candidate signature metrics for holding midfielders, per player-match.

All counts are numerators/denominators, so rates can be pooled over any set of
matches. Coordinates are StatsBomb's, in the acting team's frame (attacking +x).

Touch sequences: in a team's in-possession event stream (match order), a run of
consecutive events by the same player is one "touch sequence". If it starts with
a ball receipt and the event before it is a team-mate's pass, that pass is the
incoming ball; if it ends with a pass, that is the outgoing ball.

  fwd_after_recv      outgoing pass is forward (>= 5 units) | received, then passed
  turn_after_back     same, but only when the incoming ball travelled forward
                      into him (>= 5 units): he received facing his own goal
  bounce              outgoing pass goes straight back to the incoming passer
  redirect            incoming and outgoing balls go to opposite flanks
                      (|dy_in| >= 5, |dy_out| >= 10)
  central_break       completed pass from own half travelling >= 15 units
                      forward and ending in the central 44 units
  buildup_share       share of the team's possessions that start in its own
                      third (while he is on) in which he touches the ball
  counter_foul        fouls committed within 8 s of the opponent winning the ball
  mid_intercept       interceptions in the middle third
  quick               received and passed on within 2 s (carry logged or not)
  up_keep / up_recv   received under pressure and completed the next pass
360 (pass/receipt freeze-frames, visible players only):
  bypassed            opponents between passer and pass end (in x), per
                      completed forward pass
  line_break          completed passes bypassing >= 2 opponents
  pass_to_space       distance from pass end to nearest opponent (completed passes)
  opp_behind_recv     opponents already behind him (x < his x) at his receipt
  crowd_recv          opponents within 5 units at his receipt
"""
from __future__ import annotations

import polars as pl

IN_POSS = ("Pass", "Carry", "Ball Receipt*", "Dribble", "Shot", "Miscontrol", "Dispossessed")
RESTARTS = ("Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in")
COLS = ["match_id", "event_id", "idx", "period", "period_seconds", "possession", "possession_team_id", "team_id",
        "player_id", "type", "sub_type", "outcome", "x", "y", "end_x", "end_y", "under_pressure", "recipient_id"]


def touch_sequences(ev: pl.DataFrame) -> pl.DataFrame:
    """One row per touch sequence with incoming/outgoing ball attributes."""
    s = (ev.filter((pl.col("team_id") == pl.col("possession_team_id")) & pl.col("type").is_in(IN_POSS)
                   & pl.col("player_id").is_not_null())
           .sort("match_id", "idx"))
    s = s.with_columns(
        _new=((pl.col("player_id") != pl.col("player_id").shift(1).over("match_id"))
              | (pl.col("possession") != pl.col("possession").shift(1).over("match_id"))).fill_null(True))
    s = s.with_columns(seq=pl.col("_new").cast(pl.Int32).cum_sum().over("match_id"))
    prev = s.select("match_id", "seq", p_type="type", p_player="player_id", p_x="x", p_y="y", p_ex="end_x",
                    p_ey="end_y", p_out="outcome", p_poss="possession").with_columns(seq=pl.col("seq") + 1)
    prev = prev.group_by("match_id", "seq").agg(pl.all().last())
    g = s.group_by("match_id", "seq").agg(
        pl.col("player_id").first(), pl.col("team_id").first(), pl.col("possession").first(),
        pl.col("type").first().alias("f_type"), pl.col("under_pressure").first().alias("f_up"),
        pl.col("type").last().alias("l_type"), pl.col("sub_type").last().alias("l_sub"),
        pl.col("outcome").last().alias("l_out"), pl.col("x").last().alias("l_x"), pl.col("y").last().alias("l_y"),
        pl.col("end_x").last().alias("l_ex"), pl.col("end_y").last().alias("l_ey"),
        pl.col("recipient_id").last().alias("l_rec"), pl.col("event_id").last().alias("l_event"),
        pl.col("event_id").first().alias("f_event"),
        (pl.col("period_seconds").last() - pl.col("period_seconds").first()).alias("dur"))
    g = g.join(prev, on=["match_id", "seq"], how="left")
    received = (pl.col("f_type") == "Ball Receipt*") & (pl.col("p_type") == "Pass") & pl.col("p_out").is_null() \
        & (pl.col("p_poss") == pl.col("possession"))
    out_pass = (pl.col("l_type") == "Pass") & ~pl.col("l_sub").fill_null("").is_in(RESTARTS) & pl.col("l_ex").is_not_null()
    in_dx, in_dy = pl.col("p_ex") - pl.col("p_x"), pl.col("p_ey") - pl.col("p_y")
    out_dx, out_dy = pl.col("l_ex") - pl.col("l_x"), pl.col("l_ey") - pl.col("l_y")
    rp = received & out_pass
    return g.with_columns(
        rp=rp, rp_fwd=rp & (out_dx >= 5),
        rb=rp & (in_dx >= 5), rb_fwd=rp & (in_dx >= 5) & (out_dx >= 5),
        bounce=rp & (pl.col("l_rec") == pl.col("p_player")),
        redirect=rp & (in_dy.abs() >= 5) & (out_dy.abs() >= 10) & (in_dy.sign() != out_dy.sign()),
        op=out_pass, op_ok=out_pass & pl.col("l_out").is_null(),
        quick=rp & (pl.col("dur") <= 2.0),
        up_recv=received & pl.col("f_up").fill_null(False),
        up_keep=received & pl.col("f_up").fill_null(False) & out_pass & pl.col("l_out").is_null(),
        central_break=out_pass & pl.col("l_out").is_null() & (pl.col("l_x") < 60) & (out_dx >= 15)
        & pl.col("l_ey").is_between(18, 62),
    )


def buildup_involvement(ev: pl.DataFrame, spells: pl.DataFrame) -> pl.DataFrame:
    """Per (match, player): own-third-start possessions while on the pitch, and
    how many of them he touched. `spells` has match_id, player_id, start, end
    in the period-keyed clock used by rolefit.traits."""
    from rolefit.traits import PERIOD_KEY
    first = (ev.sort("match_id", "idx").group_by("match_id", "possession")
               .agg(pl.col("possession_team_id").first().alias("team"), pl.col("x").drop_nulls().first().alias("x0"),
                    (pl.col("period").first() * PERIOD_KEY + pl.col("period_seconds").first().round(0)).alias("t")))
    deep = first.filter(pl.col("x0") < 40)
    touched = (ev.filter(pl.col("type").is_in(IN_POSS) & (pl.col("team_id") == pl.col("possession_team_id")))
                 .select("match_id", "possession", "player_id").unique())
    sp = spells.select("match_id", "team_id", "player_id", "start", "end")
    on = sp.join(deep, left_on=["match_id", "team_id"], right_on=["match_id", "team"]).filter(
        (pl.col("t") >= pl.col("start")) & (pl.col("t") < pl.col("end")))
    on = on.join(touched.with_columns(hit=pl.lit(True)), on=["match_id", "possession", "player_id"], how="left")
    return on.group_by("match_id", "player_id").agg(pl.len().alias("buildup_poss"),
                                                    pl.col("hit").fill_null(False).sum().alias("buildup_touched"))


def defensive_counts(ev: pl.DataFrame) -> pl.DataFrame:
    start = (ev.sort("match_id", "idx").group_by("match_id", "possession")
               .agg(pl.col("period_seconds").first().alias("p_start"), pl.col("period").first().alias("p_period")))
    d = ev.join(start, on=["match_id", "possession"], how="left")
    foul = (d["type"] == "Foul Committed")
    return d.with_columns(
        counter_foul=foul & (pl.col("team_id") != pl.col("possession_team_id")) & (pl.col("period") == pl.col("p_period"))
        & ((pl.col("period_seconds") - pl.col("p_start")) <= 8.0),
        mid_intercept=(pl.col("type") == "Interception") & pl.col("x").is_between(40, 80, closed="left"),
    ).group_by("match_id", "player_id").agg(pl.col("counter_foul").sum(), pl.col("mid_intercept").sum())


def frame_metrics(seqs: pl.DataFrame, ev: pl.DataFrame, frames: pl.DataFrame) -> pl.DataFrame:
    """360 metrics per (match, player). Pass frames are the passer's; receipt
    frames the receiver's (StatsBomb stores both in the actor's frame)."""
    opp = frames.filter(~pl.col("teammate")).select("event_id", ox="x", oy="y")
    passes = seqs.filter(pl.col("op_ok") & (pl.col("l_ex") > pl.col("l_x"))).select(
        "match_id", "player_id", event_id="l_event", sx="l_x", ex="l_ex", ey="l_ey")
    pj = passes.join(opp, on="event_id")
    per_pass = pj.group_by("match_id", "player_id", "event_id").agg(
        ((pl.col("ox") > pl.col("sx")) & (pl.col("ox") < pl.col("ex"))).sum().alias("byp"),
        (((pl.col("ox") - pl.col("ex")) ** 2 + (pl.col("oy") - pl.col("ey")) ** 2).sqrt().min()).alias("space"))
    pm = per_pass.group_by("match_id", "player_id").agg(
        pl.len().alias("f_passes"), pl.col("byp").sum().alias("bypassed_sum"),
        (pl.col("byp") >= 2).sum().alias("line_breaks"), pl.col("space").sum().alias("space_sum"))
    rec = ev.filter((pl.col("type") == "Ball Receipt*") & pl.col("outcome").is_null() & pl.col("x").is_not_null()).select(
        "match_id", "player_id", "event_id", rx="x", ry="y")
    rj = rec.join(opp, on="event_id")
    per_rec = rj.group_by("match_id", "player_id", "event_id").agg(
        (pl.col("ox") < pl.col("rx")).sum().alias("behind"),
        ((((pl.col("ox") - pl.col("rx")) ** 2 + (pl.col("oy") - pl.col("ry")) ** 2).sqrt()) < 5).sum().alias("crowd"))
    rm = per_rec.group_by("match_id", "player_id").agg(
        pl.len().alias("f_recv"), pl.col("behind").sum().alias("behind_sum"), pl.col("crowd").sum().alias("crowd_sum"))
    return pm.join(rm, on=["match_id", "player_id"], how="full", coalesce=True)


def player_match_table(ev: pl.DataFrame, spells: pl.DataFrame, frames: pl.DataFrame | None) -> pl.DataFrame:
    seqs = touch_sequences(ev)
    s = seqs.group_by("match_id", "player_id").agg(
        pl.col("rp").sum(), pl.col("rp_fwd").sum(), pl.col("rb").sum(), pl.col("rb_fwd").sum(),
        pl.col("bounce").sum(), pl.col("redirect").sum(), pl.col("op").sum(), pl.col("op_ok").sum(),
        pl.col("central_break").sum(), pl.col("quick").sum(), pl.col("up_recv").sum(), pl.col("up_keep").sum(),
        pl.len().alias("sequences"))
    out = s.join(buildup_involvement(ev, spells), on=["match_id", "player_id"], how="left") \
           .join(defensive_counts(ev), on=["match_id", "player_id"], how="left")
    if frames is not None and not frames.is_empty():
        out = out.join(frame_metrics(seqs, ev, frames), on=["match_id", "player_id"], how="left")
    return out
