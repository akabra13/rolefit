# Back-test: do signings that fit a team's weak area improve it?

The Library's Team needs and Current squads tabs suggest the type of player a team should add for its weakest area. That rests on an assumption: adding a player strong in an area improves the team in that area. Earlier results (V4b) found newcomers mostly adapt to their new team, which cuts against it. This study tests the assumption on the transitions the data allows.

## Data

Team-season transitions where the same team has StatsBomb data in two consecutive seasons of the same competition, with at least 10 matches in each: FA WSL 2018/19 -> 2019/20 -> 2020/21 (about 11-12 teams per transition) and Barcelona's La Liga seasons 2004/05 -> 2020/21 (one team, about 16 transitions; its metrics are compared with the pooled men's team-seasons). This is small, and the test is low-powered; a null result would not show the assumption is wrong.

## Pre-registered design (fixed 2026-10-21, before running)

- **Areas:** the 13 team metrics that have a direction (`team.py`, `GOOD_HIGH` not None): chances, shots, box and final-third entries, progression, own-third losses, set-piece threat, chances and box entries conceded, high regains, counter-attack and set-piece chances conceded, aerial win share.
- **Outcome:** change in the team's metric from season t to t+1, turned so that up is better, divided by the standard deviation of that metric over all team-seasons of the gender.
- **Arrivals:** players with at least 450 minutes for the team in t+1 and none in t. Each arrival's strength in an area is the mean, over the area's linked traits (`needs.TRAIT_LINKS`), of the sign-corrected trait (position-relative, in standard deviations) from their most recent profile **before** t+1 (any team or competition, at least 450 minutes). `arrivals` = sum over arrivals with a prior profile of (share of the team's outfield minutes in t+1) x (strength).
- **Departures:** players with at least 450 minutes in t and none in t+1, with their season-t profile, weighted by their share of t minutes.
- **Model:** outcome = b0 + b1 x arrivals + b2 x departures + b3 x (team's z in the area at t) over all transition x area rows; the last term absorbs regression to the mean.
- **Primary test:** b1 > 0 with the 90% bootstrap CI (resampling whole team-transitions) above 0. Read as: bringing in strength in an area is followed by improvement in that area.
- **Secondary (descriptive):** the same with an interaction for weak areas (team's z < -0.5 at t); share of arrivals with a prior profile.

Whatever comes out is reported, and the Library's wording follows it.
