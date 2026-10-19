"""Team-season profiles: how a team plays and where it falls short.

Per team-match we count, for the team and against it, then pool over the
season. All metrics are per possession, per opponent possession or shares, so
they are comparable between teams that see more or less of the ball.

attack     xg_for_pp, shots_pp, box_entries_pp, final_third_pp, prog_pp,
           own_third_loss_pp (lost in own third, per own possession),
           pass_cmp, possession_share, cross_share, setpiece_xg_pp
defence    xg_against_pp (per opponent possession), box_entries_against_pp,
           high_regain_pp (regains in the attacking 40%, per opp. possession),
           ppda (opponent passes in their own 60% per our defensive action there),
           counter_xg_against_pp, setpiece_xg_against_pp, aerial_win_share
outcomes   points_pm, goal_diff_pm, xg_diff_pm (per match)
"""
from __future__ import annotations

import polars as pl

SETPIECE = ("From Corner", "From Free Kick", "From Throw In")
DEF_TYPES = ("Duel", "Interception", "Foul Committed", "Ball Recovery", "Block", "Pressure")
METRICS = ("xg_for_pp", "shots_pp", "box_entries_pp", "final_third_pp", "prog_pp", "own_third_loss_pp", "pass_cmp",
           "possession_share", "cross_share", "setpiece_xg_pp", "xg_against_pp", "box_entries_against_pp",
           "high_regain_pp", "ppda", "counter_xg_against_pp", "setpiece_xg_against_pp", "aerial_win_share")
# higher is better for the team? (used to turn gaps into weaknesses)
GOOD_HIGH = {"xg_for_pp": True, "shots_pp": True, "box_entries_pp": True, "final_third_pp": True, "prog_pp": True,
             "own_third_loss_pp": False, "pass_cmp": True, "possession_share": True, "cross_share": None,
             "setpiece_xg_pp": True, "xg_against_pp": False, "box_entries_against_pp": False, "high_regain_pp": True,
             "ppda": None, "counter_xg_against_pp": False, "setpiece_xg_against_pp": False, "aerial_win_share": True}


def team_match_counts(ev: pl.DataFrame, matches: pl.DataFrame) -> pl.DataFrame:
    """One row per (match, team) with counts for and against. `ev` needs the
    lake event columns; `matches` match_id, home/away team ids and scores."""
    t = pl.col("type")
    sub = pl.col("sub_type").fill_null("")
    own = pl.col("team_id") == pl.col("possession_team_id")
    in_box = (pl.col("end_x") >= 102) & pl.col("end_y").is_between(18, 62)
    was_box = (pl.col("x") >= 102) & pl.col("y").is_between(18, 62)
    ok_move = ((t == "Pass") & pl.col("outcome").is_null()) | (t == "Carry")
    d0 = ((120 - pl.col("x")) ** 2 + (40 - pl.col("y")) ** 2).sqrt()
    d1 = ((120 - pl.col("end_x")) ** 2 + (40 - pl.col("end_y")) ** 2).sqrt()
    shot = (t == "Shot") & (sub != "Penalty")
    e = ev.with_columns(
        shot=shot, xg_=pl.when(shot).then(pl.col("xg").fill_null(0.0)).otherwise(0.0),
        box_entry=ok_move & in_box & ~was_box,
        ft_entry=ok_move & (pl.col("end_x") >= 80) & (pl.col("x") < 80),
        prog=ok_move & (d1 <= 0.75 * d0) & (pl.col("end_x") >= 60),
        loss_own3=own & (pl.col("x") < 40) & (t.is_in(["Miscontrol", "Dispossessed"])
                                              | ((t == "Pass") & pl.col("outcome").is_not_null())),
        is_pass=(t == "Pass") & ~sub.is_in(["Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in"]),
        pass_ok=(t == "Pass") & pl.col("outcome").is_null() & ~sub.is_in(["Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in"]),
        cross=(t == "Pass") & pl.col("extra").fill_null("").str.contains('"cross": true', literal=True),
        sp_xg=pl.when(shot & pl.col("play_pattern").is_in(SETPIECE)).then(pl.col("xg").fill_null(0.0)).otherwise(0.0),
        counter_xg=pl.when(shot & (pl.col("play_pattern") == "From Counter")).then(pl.col("xg").fill_null(0.0)).otherwise(0.0),
        high_regain=t.is_in(["Ball Recovery", "Interception"]) & (pl.col("x") >= 72),
        opp_build_pass=(t == "Pass") & (pl.col("x") < 72),          # in the passing team's own 60%
        def_high=t.is_in(DEF_TYPES) & (pl.col("x") >= 48),          # our actions in their 60%
        aerial_won=pl.col("extra").fill_null("").str.contains('"aerial_won": true', literal=True),
        aerial_lost=(t == "Duel") & (sub == "Aerial Lost"),
    )
    per = e.group_by("match_id", "team_id").agg(
        pl.col("shot").sum(), pl.col("xg_").sum().alias("xg"), pl.col("box_entry").sum(), pl.col("ft_entry").sum(),
        pl.col("prog").sum(), pl.col("loss_own3").sum(), pl.col("is_pass").sum().alias("passes"),
        pl.col("pass_ok").sum(), pl.col("cross").sum(), pl.col("sp_xg").sum(), pl.col("counter_xg").sum(),
        pl.col("high_regain").sum(), pl.col("opp_build_pass").sum().alias("build_passes"),
        pl.col("def_high").sum(), pl.col("aerial_won").sum(), pl.col("aerial_lost").sum())
    poss = (ev.filter(pl.col("period") <= 4).group_by("match_id", "possession")
              .agg(pl.col("possession_team_id").first().alias("team_id"))
              .group_by("match_id", "team_id").agg(pl.len().alias("possessions")))
    per = per.join(poss, on=["match_id", "team_id"], how="left").fill_null(0)
    opp = per.rename({c: f"opp_{c}" for c in per.columns if c not in ("match_id",)})
    pairs = per.join(opp, on="match_id").filter(pl.col("team_id") != pl.col("opp_team_id"))
    m = matches.select("match_id", "home_team_id", "away_team_id", "home_score", "away_score")
    pairs = pairs.join(m, on="match_id", how="left").with_columns(
        gf=pl.when(pl.col("team_id") == pl.col("home_team_id")).then(pl.col("home_score")).otherwise(pl.col("away_score")),
        ga=pl.when(pl.col("team_id") == pl.col("home_team_id")).then(pl.col("away_score")).otherwise(pl.col("home_score")))
    return pairs.with_columns(points=pl.when(pl.col("gf") > pl.col("ga")).then(3).when(pl.col("gf") == pl.col("ga")).then(1).otherwise(0))


def team_season_metrics(tm: pl.DataFrame, by=("key", "team_id")) -> pl.DataFrame:
    s = lambda c: pl.col(c).sum()  # noqa: E731
    g = tm.group_by(list(by)).agg(
        pl.len().alias("matches"), s("possessions"), s("opp_possessions"), s("xg"), s("opp_xg"), s("shot"),
        s("box_entry"), s("opp_box_entry"), s("ft_entry"), s("prog"), s("loss_own3"), s("passes"), s("pass_ok"),
        s("cross"), s("sp_xg"), s("opp_sp_xg"), s("opp_counter_xg"), s("high_regain"), s("opp_build_passes"),
        s("def_high"), s("aerial_won"), s("aerial_lost"), s("points"), s("gf"), s("ga"))
    pp = lambda c: pl.col(c) / pl.col("possessions")  # noqa: E731
    opp = lambda c: pl.col(c) / pl.col("opp_possessions")  # noqa: E731
    return g.with_columns(
        xg_for_pp=pp("xg"), shots_pp=pp("shot"), box_entries_pp=pp("box_entry"), final_third_pp=pp("ft_entry"),
        prog_pp=pp("prog"), own_third_loss_pp=pp("loss_own3"), pass_cmp=pl.col("pass_ok") / pl.col("passes"),
        possession_share=pl.col("possessions") / (pl.col("possessions") + pl.col("opp_possessions")),
        cross_share=pl.col("cross") / pl.col("passes"), setpiece_xg_pp=pp("sp_xg"),
        xg_against_pp=opp("opp_xg"), box_entries_against_pp=opp("opp_box_entry"), high_regain_pp=opp("high_regain"),
        ppda=pl.col("opp_build_passes") / pl.when(pl.col("def_high") > 0).then(pl.col("def_high")).otherwise(None),
        counter_xg_against_pp=opp("opp_counter_xg"), setpiece_xg_against_pp=opp("opp_sp_xg"),
        aerial_win_share=pl.col("aerial_won") / (pl.col("aerial_won") + pl.col("aerial_lost")),
        points_pm=pl.col("points") / pl.col("matches"), goal_diff_pm=(pl.col("gf") - pl.col("ga")) / pl.col("matches"),
        xg_diff_pm=(pl.col("xg") - pl.col("opp_xg")) / pl.col("matches"))
