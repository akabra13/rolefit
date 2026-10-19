# rolefit

*Who plays this player's role, how does a team play with them, what does a team need from a role, and who could do it?*

Third project in the PitchGraph programme (after pitchgraph v3 and dynamic_shift). It depends on pitchgraph's Parquet lake (StatsBomb Open Data on Drive). No data lives in this repo or on a laptop. Derived tables go to Drive (`rolefit/`).

## The problem it is built around

Every number we see for a player mixes the player with the system they play in. A full-back's crossing rate is partly their habit and partly their team's left-lane routes. RoleFit makes that **player vs system** split explicit. It measures the split on players seen at more than one team (club and country, transfers), and builds fit on top of it.

## Four layers

1. **Player traits** (`traits.py`, `xpass.py`). There are 43 interpretable traits in 9 families: territory, involvement, progression, creation, retention, passing profile, carrying, finishing and defending.
   - Each trait is a share, a mean, an over-expected residual, or a rate per 100 possessions **while the player was on the pitch** (own possessions for on-ball traits, opponent possessions for defending). On-pitch possessions come from lineup spells, keyed by period so stoppage time can't overlap.
   - Pass completion over expected uses a gradient-boosted completion model cross-fitted by competition-season.
   - Value added uses pitchgraph's `MarkovValue`.
   - Each trait will also get a **system-adjusted** version (stage 2): the player relative to their team's baseline in the same zones.
2. **How the team plays with the player** (`influence.py`, stage 3).
   - Hub role from the team's player chain: flow share, removal bounds, final-third dependency, partners.
   - On/off style shift: fingerprint distance, lanes and threat with vs without the player. This is exploratory and has low power.
3. **What the team needs from a role** (`demand.py`, stage 3). A slot is a team-season × position group.
   - Its demand profile is the minutes-weighted traits of the players who filled the slot.
   - It also records the slot's share of the system (flow, progression, creation, defending) and the team's style context.
   - All of these are compared with the league's same-position average.
4. **Fit** (`fit.py`, stage 4). Components are shown separately, with no fused score unless validated.
   - **Role translation:** predict how a player would play in a new slot. Per trait, `new = a·old + b·(demand_new − demand_old) + c`, learnt on movers and cross-fitted by player. The coefficients tell you which traits belong to the player and which belong to the system.
   - **Demand gap:** predicted traits against what the slot asks for, weighted by how much the system leans on each trait.
   - **"Plays like":** similarity on portability-weighted traits.
   - **Counterfactual Δ threat:** swap the player into the team's player chain, with no-adaptation and full-adaptation bounds. v3 found this kind of counterfactual weak (r=0.08 vs real absences), so it is shown as a bound, not a forecast.

## Pre-registered validation (written before any run)

| # | Test | Pass bar |
|---|---|---|
| V1 | Trait reliability. Each unit's matches are sorted by date and alternated into halves; units have ≥900 minutes. Per trait, Pearson r between halves, both overall and **within position group** (each group's mean removed, so a trait can't pass just by separating centre-backs from strikers). | A trait is **kept** if within-group r ≥ 0.5. The kept list is reported, and later stages use only kept traits. |
| V2 | Portability. For movers, the correlation of each trait across teams, raw vs system-adjusted. The ceiling is the same player at the same team in another season. | Descriptive. It produces the portability weights. |
| V3 | Re-identification on the portability-weighted embedding: split-half, and cross-team (club ↔ country, transfers). Baselines are gallery chance and the position label alone. | Split-half top-5 ≥ 0.50. Cross-team top-5 ≥ 2× the position-label baseline, with the 90% Wilson lower bound above it. |
| V4 | Role translation: error predicting movers' traits at the new team, cross-fitted by player. Baselines are old traits only, slot demand only, and the league position mean. | The full model beats **both** single-source baselines on median standardised error, with the bootstrap-by-player CI of the improvement excluding 0. |
| V5 | Counterfactual Δ threat vs the observed change in the new team's threat for movers. | Exploratory, reported with a low-power caveat. |
| V6 | Face validity: "plays like" lists for archetypes (deep playmaker, inverted full-back, target forward). | Reviewed by the user. |

**Stage 2 specifics (fixed 2026-10-13, before any real-data run):**
- **Goalkeepers excluded.** Outfield traits don't describe them.
- **System adjustment** (`system.py`): trait = league-position mean + team effect + player.
  - The league-position mean is the minutes-weighted mean for the same competition-season and position group.
  - The team effect is a leave-one-out, minutes-weighted mean of the teammates' (all positions) deviations from their own league-position means, shrunk with 10 pseudo-matches of zero effect.
  - Two versions per trait: `pos` (minus the league-position mean) and `sys` (also minus the team effect). Role-specific demands are deliberately left in; they are stage 3's demand profiles.
- **Mover pairs:** both units have at least 270 minutes. Pair types are `same_team`, `club_country`, `transfer` and `other_team`; international means the World Cup, Euro, Women's World Cup, Women's Euro, Copa America, AFCON or U20 World Cup.
- **Portability weights:** for each trait, use whichever of `pos`/`sys` has the higher cross-team r; the weight is max(0, r).
  - Weights are fitted on one random half of players and applied to the other half, so re-ID queries are never embedded with weights learnt from their own pairs.
  - In the embedding, traits are z-scored (a missing value counts as the mean) and scaled by √weight.
- **Similarity:** Euclidean distance on the embedding. Cosine was rejected before any real run: on z-scored traits it throws away magnitude, and synthetic tests showed it failing on low-dimensional profiles.
- **V3 split-half** uses the stage 1 halves of units with ≥900 minutes. Each half gets the same season-level offsets as its unit.
- **V3 cross-team primary:** a query's right answers are the player's units at other teams. The query itself and its same-team units are removed from the candidates. The per-pair-type results and the equal-weight, position-relative embedding are descriptive only.

Thresholds are fixed now. Negatives are reported plainly. Approximations are stated up front:
- Minutes come from lineup spells, with an open-ended spell counted to 90.
- Possessions in shoot-outs (period 5) are ignored.

## Stages

1. Traits + V1 (`scripts/run_traits.py`, `.github/workflows/traits.yml`). *Run 2026-10-14, see Results.*
2. Movers, system-adjusted traits, portability, re-ID (V2, V3) (`scripts/run_portability.py`, `.github/workflows/portability.yml`). It reads stage 1's tables from Drive. *Run 2026-10-14, see Results.*
3. Team side: slot demand profiles (`demand.py`, `scripts/run_demand.py`, `.github/workflows/demand.yml`). Hub-role and on/off influence (`influence.py`) come after, because they need event-level chains.
4. Fit: translation model (V4) and demand gap (`fit.py`, `scripts/run_fit.py`, `.github/workflows/fit.yml`). The chain-swap counterfactual (V5) needs event-level player chains and is deferred; it stays exploratory.
5. Export to the Library: player card, slot card, "replace X in team T" search, side-by-side. Pages state the caveats (eras differ, no age or contract data, showcase only) and carry StatsBomb attribution.

## Results (2026-10-14, full lake: 3,960 matches, 2,911 outfield-and-GK units with >= 900 minutes)

**V1 trait reliability.** 38 of 43 traits are kept (within-position split-half r >= 0.5); most are 0.6-0.9. Not kept: `value_per_action` (0.47), `takeon_success` (0.10), `xg_per_shot` (0.28), `aerial_win` (0.42), `counterpress_share` (0.28). Success rates and per-event quality are noisy, while volumes and locations are stable. The completion model (cross-fitted by competition-season) has out-of-fold AUC 0.916.

**V2 portability** (r between a player's two units; 90% CIs in `portability.json`). Pairs: 1,665 same-team, 1,101 club-country, 556 transfer. Typical trait: r about 0.5 with the team held fixed and about 0.35 across teams, so roughly two thirds of the stable signal travels.
- Most portable across teams: `avg_y` (0.57, the side of the pitch), `takeon_p100` (0.52), `aerials_p100` (0.55), `prog_carry_p100` (0.50), `box_entry_p100` (0.50), `through_p100` (0.49).
- Least portable: `shot_dist` (0.17), `y_spread` (0.17), `avg_x` (0.17-0.22), `def_height` (0.16-0.23), `pressured_pass_share` (0.19).
- **The team-effect adjustment does not help in general.** For volume traits it hurts: `touches_p100` 0.48 raw to 0.29 adjusted, `carry_p100` 0.48 to 0.31. It helps for passing direction: `pass_fwd` 0.35 to 0.44 and `back_share` 0.35 to 0.43 across teams. A likely reason is that a team's overall volume partly reflects the role mix its players fill, so removing it also removes part of the player. Weights therefore pick the better version per trait.

**V3 re-identification (both pre-registered bars pass):**
- **Split-half:** top-5 0.634 (90% CI 0.619-0.649), top-1 0.430, among 2,678 queries; chance 0.002, position label 0.013. Bar: top-5 >= 0.50. Equal-weight position-relative embedding: 0.603.
- **Cross-team:** top-5 0.070 (0.061-0.080), top-1 0.029, among 1,860 queries in a gallery of 5,536 units; chance 0.002, position label 0.006. That is 11x the position baseline and the CI sits above it, so the bar passes. Club-country 0.055, transfer 0.084. Equal weights: 0.064.
- **What it means:** within the same team, a player is highly recognisable. Across teams they are recognisable well above chance but rarely at the top. At 7% top-5 the vector is not a "who is this player" identifier across systems; it is a useful but weak signal, consistent with about two thirds portability per trait. Portability weighting added little over equal weights (0.070 vs 0.064).
- Caveat: the gallery includes thousands of low-minute units (270 minutes or more), which makes cross-team harder than if only established players were listed.

Implication for fit (stages 3-4): roughly a third of a trait's stable variation depends on the system, so translating a player to a new team is a real modelling problem, not an identity map.

## Pre-registered: V3b slot-demand persistence (written 2026-10-14, before running stage 3)

Question: is a slot's demand a property of the **system**, or just of whoever fills it?

- A slot is (competition-season, team, position group), outfield only, with at least 900 slot-minutes. Demand is the minutes-weighted mean of the V1-kept traits of its players, relative to the league's same-position mean (in standard deviations over all slots of the group).
- Test pairs: the same team and position group in two different competition-seasons. Per trait, the Pearson r of relative demand across the pair. Three numbers: all pairs; pairs where the **main player (most minutes) changed**; and a baseline pairing each slot with a random other team's slot of the same group in the second season (200 permutations, mean r).
- **Pass bar:** median over traits of r(same team, new main player) >= 0.20 AND greater than the median other-team r + 0.10. Otherwise slot demand is reported as not separable from the occupant.
- Also reported (descriptive): each slot's share of its team's volume traits (touches, progression, creation, defending), and cards listing a slot's most unusual traits.
- Limitation stated up front: only seasons with full minutes data pair up, which is mostly the four WSL seasons and Barcelona's La Liga seasons, so the test is low-powered and clubs with repeated seasons are not a random sample.

### Result: V3b slot-demand persistence (2026-10-14) — PASS

1,531 slots (>= 900 slot-minutes) over 210 teams and 48 competition-seasons; 1,214 same-team, same-position pairs across two seasons, of which 926 had a different main player.
- Median over the 38 traits: same team **0.25**, same team with a new main player **0.24**, random other team **-0.03**. Bar (>= 0.20 and > other-team + 0.10): passes.
- The slot's demand barely changes when its main player does. Most persistent: `avg_x` 0.47, `def_height` 0.44, `prog_pass_p100` 0.48, `pass_cmp` 0.43, `pass_cmp_oe` 0.49, `shots_p100` 0.38, `xg_p100` 0.38, `touches_p100` 0.32. Least: `through_p100` 0.00, `pass_share` 0.03, `pressures_p100o` 0.04, `takeon_p100` 0.07.
- Caveats: "main player changed" means the player with the most minutes changed. Other squad members, the manager and the club's resources often persist, so this shows demand attaches to the **team context**, not that it is a tactical instruction. The sample is dominated by teams with repeated full-minutes seasons (WSL, Barcelona, a few others), and a correlation of about 0.25 is modest: most of a slot's profile is still not predictable from the same slot's previous season.
- Consequence for stage 4: demand is a usable second input for the role-translation model, alongside the player's own traits.

## Pre-registered: V4 role translation specifics (written 2026-10-14, before running stage 4)

- **Samples:** every player with two units at different teams (club-country, transfer, other-team pairs), both >= 270 minutes, outfield; each pair in both directions (old -> new). Same-team pairs are scored separately as a ceiling, not used for the verdict.
- **Variables** (all position-relative, i.e. minus the league's same-position mean; the 38 V1-kept traits): `y` = the player's trait in the new unit; `x_old` = their trait in the old unit; `D_new`, `D_old` = what the slot asks for, defined as the minutes-weighted mean of the **other** players in that slot (leave-one-out, so a player's own new numbers cannot leak into the prediction). Slot-mates need >= 270 minutes between them, otherwise the pair is dropped for all models.
- **Models** (per trait, ridge with intercept, lambda fixed at 1.0 on standardised features): `full` y ~ x_old + D_new + D_old; `player_only` y ~ x_old; `system_only` y ~ D_new; `position` y = 0.
- **Cross-fitting:** 5 folds, assigned by player, so a player's moves are never in the training data that predicts them.
- **Metric:** per trait, out-of-fold MSE divided by the variance of `y` (1.0 = no better than the position mean); the headline is the median over traits.
- **Pass bar:** `full` beats **both** `player_only` and `system_only` on the median standardised error, with the 90% bootstrap-by-player CI of each improvement above 0.
- **Descriptive:** per-trait errors, how many traits `full` wins, and the standardised coefficients (a = player, b = new slot, c = old slot) which show which traits travel with the player and which come from the system.

### Result: V4 role translation (2026-10-14) — PASS, with a small margin

2,078 directed cross-team pairs from 465 players (club-country and transfers), 38 traits, folds by player.

| Model | Median standardised error (90% CI) |
|---|---|
| full (old traits + new slot + old slot) | **0.849** (0.824-0.868) |
| player only (old traits) | 0.873 (0.865-0.919) |
| system only (new slot's demand) | 0.987 (0.959-1.003) |
| position mean | 1.059 (1.032-1.113) |

- The full model beats player-only by **+0.025** (90% CI +0.016 to +0.079), system-only by +0.138 (+0.112 to +0.161) and the position mean by +0.210. Per trait it beats system-only on 38 of 38 and player-only on 29 of 38. Bar: passes.
- **Honest size of the effect:** adding the new slot to a player's own history improves prediction by about 3% of the variance. A player's old numbers do most of the work; the new system explains a little more on top. The picture is the same for players who stayed at the same team (full 0.736, player-only 0.749), where the system should add nothing: for those pairs the improvement is +0.013 and its CI includes 0. So the cross-team gain is small but is not just a modelling artefact; it is larger exactly where the system differs.
- **Where the system matters** (largest b on the new slot, standardised): `touches_p100` and `recv_p100` (+0.50, +0.56), `carry_p100` (+0.50), `pass_cmp` (+0.50), `avg_x` (+0.56), `def_height` (+0.54), `high_share` (+0.43). These are volume, height and accuracy traits, and a player's value moves with the slot. **Where the player dominates** (b about 0): `prog_pass`, `prog_carry`, `shot_assist`, `shots_p100`, `takeon_p100`, `aerials_p100`, `pressures_p100o`, `regains_p100o`, `tackles_p100o`. Style choices (how to progress, who to take on, how to defend) travel; how much and how safely you are used does not.
- **Caveats:**
  - The "position mean" baseline scores above 1.0 on several traits because movers are not typical (their `y` is shifted from the position mean), so its comparison is easy to beat and not a strong baseline.
  - `avg_y` has a large negative coefficient on the new slot because a position group mixes left and right sides, and a slot's other players sit on the opposite side. It is a grouping artefact, not a real effect.
  - Movers are a selected sample (players good enough to be at two teams with 270+ minutes). Prediction error for typical players could be larger.

### Result: V6 "plays like" lists (2026-10-14) — mixed, for the user to judge

`scripts/run_similar.py` lists each query player's 8 nearest players (Euclidean distance on the V3 portability-weighted embedding, one row per player, units >= 900 minutes). Full lists are in the run output and `similar.json` on Drive.

- **Convincing:** Busquets -> Verratti, Keira Walsh, Zadrazil, Stambouli (all defensive midfielders); Messi -> Neymar, Di Maria, Ibrahimovic, Miedema; Neymar -> Messi, Dembele, Mbappe, Iniesta; Kroos -> Modric, Xavi, Walsh; Luis Suarez -> Benzema, Eto'o, Kane, Sturridge, Lacazette; Putellas -> Bonmati; Jordi Alba -> full-backs (Kolarov, Grimaldo, Clichy) and women's full-backs; Van Dijk -> Pique and other centre-backs; Modric -> Rakitic, Xavi, Badelj.
- **Weak:** Kante's neighbours are mostly women's centre-backs and defensive midfielders, with no obvious box-to-box type; Vardy -> Thierry Henry, Dzemaili and women's strikers; Ronaldo -> Villa, Henry and women's wingers and strikers; Dani Alves -> Mascherano, Ramsey, Wirtz (a poor match for an attacking right-back); Sergio Ramos' list is plausible but unfocused.
- **Descriptive check:** 72 of 136 neighbours (53%) are in the query's position group. Roles overlap in practice (a Kroos-style DM sits near CMs), but a role vector that returned the wrong group half the time would not be useful.
- **Known limitation:** men's and women's players mix freely. Traits are position-relative within each competition, so level differences between leagues are removed by design; that makes cross-gender "plays like" lists possible but unfiltered. The Library should add filters (gender, league, season, position group).
- **Not yet done:** Lahm has no 900+ minute unit in this lake. The list has not been compared with a baseline (position label, raw traits), so face validity is the only claim.

## Pre-registered: V4b team context (written 2026-10-14, before running)

Question raised by the user: does the receiving team itself matter beyond its slot, i.e. its formation, how it plays as a whole, and who is already in the squad?

- **Context of the new team-season, built without the mover:**
  - *formation mix*: share of the team's matches started in each of 9 formation families, from Starting XI events;
  - *squad supply* `O_t`: minutes-weighted mean of every other outfield player's trait `t` at that team-season;
  - *style*: the first 5 principal components of `O` over all team-seasons.
- **Models**, all else identical to V4 (per-trait ridge, cross-fitted by player, same samples, restricted to those with full context): `full` (V4 model), `full_others` (+ `O_t` for the trait being predicted), `full_formstyle` (+ formation shares and style components), `full_ctx` (all of them).
- **Pass bar:** `full_ctx` beats `full` on the median standardised error, with the 90% bootstrap-by-player CI of the improvement above 0. Run separately for men and women.
- Descriptive: which traits gain or lose most from context.
- A "complements the team" score will only be built if V4b passes; otherwise the page shows the squad's coverage as information, not a score.

### Result: V4b team context (2026-10-14), men and women separately

Pairs: 1,072 (230 men) and 1,118 (260 women), cross-team movers with full context.

| Model | Men: improvement over V4 `full` (90% CI) | Women: improvement over V4 `full` (90% CI) |
|---|---|---|
| + squad supply `O_t` | **+0.034** (+0.005 to +0.044) | **+0.041** (+0.022 to +0.061) |
| + formation and style | -0.025 (-0.040 to +0.003) | **+0.036** (+0.009 to +0.053) |
| + all context (registered bar) | -0.014 (-0.034 to +0.015) **FAIL** | **+0.043** (+0.015 to +0.064) **PASS** |

- **Registered verdict:** the all-context model passes for women and fails for men. For men the formation and style features add 14 inputs per trait to 1,072 pairs and overfit; the formation mix alone is a weak signal. I am not claiming formation matters.
- **Squad supply helps both genders** (not the registered comparison, so exploratory): knowing what the rest of the new team already does improves the prediction of how a mover will play by about 3-4% of the variance.
- **The direction of that effect is assimilation, not complementarity.** The squad-supply coefficient is positive for 63 of 70 traits (men) and 61 of 69 (women), median +0.16 and +0.23. Strongest: receptions, touches, carries, chance-chain and build-up involvement (men +0.6 to +0.7), chance-chain, defensive height, build-up and progressive passing (women +0.5 to +0.9). Players move towards what their teammates do.
- **Where there is division of labour:** corners (-0.43 men, -0.25 women), free-kick shots and passes (-0.33, -0.25 men), side (`avg_y`). The set-piece takers are the only clearly complementary traits: a team that already has a corner taker gets less of it from a newcomer.
- **Consequence for "who complements the team best":** the data does not show that movers fill gaps; it shows they adapt to the team. I will not publish a complement score. The page shows what the rest of the squad already supplies and adapts the prediction to it. Caveat: this is observational, teams also recruit players who already fit, so assimilation and selection are mixed.

## Rework after user feedback (2026-10-14)

The first Library draft mixed men and women and described players with 38 aggregate traits; neighbours did not make sense. Changes:
- **Men and women are fully separate:** traits, reliability, portability weights, embeddings, archetypes and lists are computed per gender.
- **86 candidate traits (was 43)**, adding where a player receives and sends the ball (thirds, wide, half-spaces, box), pass technique (cut-backs, backheels, headers, footedness), actions under pressure, carries into the final third and box, defensive height and type (interceptions, recoveries, clearances, blocks, dribbled past), involvement in possessions and chance chains, set-piece duties and cards. 69 (women) and 70 (men) pass the V1 reliability bar. Finishing over expected (r about 0) and tackle success (r about 0.1) are not stable person-level traits in this data.
- **Result of the richer set** (same pre-registered bars): split-half top-5 0.775 men / 0.792 women (was 0.634 pooled); cross-team top-5 0.128 / 0.129 (was 0.070), 11x and 9x the position-label baseline.
- **"Plays like" now keeps the position in the space** (trait minus the competition-season mean over all positions, not minus the position mean). With the position removed, a defensive midfielder was close to any centre-back who was equally typical of his role. Neighbours in the query's position group: 76% men, 69% women (was 48%). Side of the pitch and footedness are left out of the plays-like space, since they say which flank, not how a player plays.
- **Playstyle archetypes** (k-means within gender and position group, k by silhouette, labelled by the traits that set the cluster apart): **loosely separated** (silhouette 0.10-0.16). Useful as tendencies (e.g. aerial target striker vs carrying striker; ball-playing vs long-ball centre-back; through-ball winger vs carrying winger), not as sharp categories.
- **Team context** (V4b) is above: squad supply helps both genders and is assimilation, formation and style help women only.
- **Library page:** one private artifact with a men/women switch, filters, plays-like, replace-a-player (slot demand, squad supply, predicted changes) and compare. It does not score "complementarity". The StatsBomb logo that their open-data terms ask for is not yet on the page.

## Pre-registered: V7 league strength and player level (written 2026-10-19, before running)

Why: every trait is relative to the player's own competition-season, so style lists mixed elite players with dominant players from weak leagues (e.g. Indian Super League strikers next to Harry Kane).

- **Quality index** per unit (outfield, >= 450 minutes, per gender): mean of z-scores, within position group pooled across all competitions, of value added per 100 possessions, chance-chain involvement, completion over expected, losses per touch (reversed) and retention under pressure.
- **Model:** q = player effect + competition-season effect, minutes-weighted, ridge-shrunk (2 and 5 pseudo-units of 900 minutes), competition effects centred. International tournaments link players from different leagues. Competition strength = minus its effect; reported only where at least 10 of its players also appear in another competition-season, otherwise "unknown".
- **V7 test:** players split into 5 folds; competition effects fitted without the fold. For each held-out player with units in two competition-seasons, predict q at B from q at A, with and without the league adjustment. **Pass bar:** the adjustment lowers the mean squared error, with the 90% bootstrap-by-player CI of the improvement above 0. Per gender.
- **Level** shown on the page: q minus the competition effect, as a percentile within gender and position group. Where competition strength is unknown, the level is the raw within-pool percentile and is marked as such.
- Caveats stated up front: the quality index is an output measure, not a full rating; it favours players on dominant teams; seasons of the same competition years apart are linked only through the same players, so ageing is mixed into era differences.

### Result: V7 (2026-10-19) — FAIL for both genders; no estimated league strength is used

- Men: 2,188 held-out pairs from 165 players; league adjustment improvement +0.035 (90% CI -0.004 to +0.069). Women: 1,488 pairs from 272 players; -0.008 (-0.023 to +0.004).
- Why it cannot work on open data: the Indian Super League (the competition that motivated this) has only 3 players also seen elsewhere, so its strength is unidentified; the 1. Bundesliga seasons have 1-2. The La Liga seasons before 2015/16 contain only Barcelona's matches, so their "strength" is really Barcelona's dominance (they come out as easy leagues). Team quality and league strength cannot be separated with this sample.
- Consequence, as registered: the page shows **level as a percentile within the player's own competition-season and position group** (pooled when fewer than 8 players), plus a **competition tier that is an assumption, labelled as such**, so users can filter weaker leagues out. The estimated strength table stays in `level.json` for transparency only.

## Fixes after the face-validity review (2026-10-19)

1. **Replace-a-player bug fixed:** a player alone in their slot (Busquets as Barcelona's only holding midfielder) left the slot's demand undefined and the list empty; the slot now falls back to what that player did.
2. **Per-player labels:** each card shows the player's own standout traits; the loose style-group label is secondary.
3. **Tempo and press resistance added** (all seasons): first-time pass share and time on the ball (V1 r = 0.60-0.72 and 0.70), pass completion under pressure, keeping the ball under pressure (men only pass V1), and from 360 frames the space and number of opponents near a receiver (men only, 59 player-seasons, because full-season 360 data covers few clubs). Stage 2 re-run: split-half top-5 0.777 / 0.801, cross-team 0.130 / 0.130 (men / women).
4. **Level:** V7 failed (above), so level is a within-competition percentile and competition tiers are an explicit assumption. Filters for tier and level remove most of the weak-league matches (Kane's list loses Roy Krishna and gains Lukaku and Bacca; Xavi's loses an Indian Super League midfielder and gains Trashorras).

**Still wrong, stated plainly:** Busquets' neighbours are still possession-side holding midfielders such as Stambouli, and the replacement list for him leads with destroyers (Gonalons, Casemiro, Obi Mikel). Busquets does stand out for keeping the ball and completing passes under pressure (+2.3 and +1.8 SD, alongside Kroos and Jorginho), but on first-time passing he is exactly average for a La Liga holding midfielder: StatsBomb logs even a small touch-and-turn as a carry, so this proxy does not see a one- and two-touch player. Volume traits shared with any possession side's holding midfielder still dominate the distance. Kante's lists are weak for a similar reason (his volume of ball-winning is not separated from position-typical defending).

## Pre-registered: V8 choice of the "plays like" space (written 2026-10-19, before running)

Problem: Busquets' neighbours are holding midfielders of other possession sides (Stambouli at PSG). Hypothesis: team-wide style (how much the team has the ball) inflates volume traits and pulls such players together.

- **Candidates**, on outfield units with >= 900 minutes, per gender, side and footedness excluded:
  A = current page space (trait minus competition-season mean, position kept, portability weights);
  B = A minus the team effect from stage 2 (what all of the player's teammates do relative to their positions);
  C = B with equal weights.
- **Criteria:** cross-team re-identification top-5 (the same player at another team, both >= 900 minutes) and position agreement (share of the 5 nearest other players in the same position group).
- **Rule:** use the space with the highest cross-team top-5 among those with position agreement >= 0.70. Fixed before seeing results. Busquets and Kante lists are then reported as they come out, good or bad.
- New traits in the same run: share of passes released within 2 seconds of receiving (whether or not a short carry was logged) and share of carries under 3 units (touch-and-turn).

### Result: V8 (2026-10-19)

New traits pass V1 for both genders: releases within 2 seconds (r_within 0.745 women / 0.77 men), touch-and-turn carries (0.67 / 0.69).

| Space | Men: cross-team top-5 (90% CI), position agreement | Women: cross-team top-5 (90% CI), position agreement |
|---|---|---|
| A position kept (previous page) | **0.219** (0.160-0.292), 0.794 | 0.231 (0.179-0.293), 0.697 (below the 0.70 bar) |
| B = A minus team style | 0.210 (0.152-0.282), 0.800 | **0.313** (0.254-0.379), 0.707 |
| C = B, equal weights | 0.210 (0.152-0.282), 0.807 | 0.293 (0.235-0.358), 0.717 |

- As registered: men keep A, women use B. For women, removing team style clearly helps find the same player at another team; for men it does not.
- **Face validity afterwards, reported as it came out:** Busquets is unchanged (Stambouli, Rabiot, Tiago, Coquelin; the replacement list still leads with Mendy, Gonalons, Stambouli, Casemiro). Putellas got worse in B (Bonmati drops out; Ella Toone and Nerea Eizaguirre come in). Walsh, Miedema and Bronze stay good. Giroud, Ozil, Kane (with the tier filter) stay good.
- Reading: the event data does not contain what makes Busquets distinct from other possession-side holding midfielders. Better proxies need body orientation, scanning and the positions of nearby opponents at every touch, which full-season 360 data would give for a few clubs only.

## Pre-registered: V9 team needs (written 2026-10-19, before running)

Goal (the user's): pick a team, see where it falls short, and what type of player would address it.

- **Team metrics** per team-season (`team.py`): chances, shots, box and final-third entries, progression, losses in own third, pass accuracy, possession share, cross share, set-piece threat; chances and box entries conceded, high regains, PPDA, counter-attack and set-piece chances conceded, aerial win share. All per possession, per opponent possession or shares.
- **Peers:** z-scores within the team's competition-season when it has at least 8 teams with 10+ matches; otherwise against the gender pool (flagged).
- **V9 test:** do the metrics (excluding chances created and conceded themselves) explain results? Ridge regression of points per match (within-league) on the peer z-scores, leave-one-competition-season-out. **Pass bar:** cross-validated R^2 >= 0.25 with the 90% bootstrap (over competition-seasons) lower bound above 0. Per gender.
- **If V9 passes,** weaknesses are weighted by each metric's importance for points; otherwise all metrics count equally and the page says so.
- **Need -> position -> player type:** each metric is linked to the player traits that produce it (a stated football judgement in `needs.py`, not learnt). The position group whose players are furthest below the league's same-position average on those traits is where the need sits. Candidates are ranked by their predicted traits in that slot (V4/V4b translation with the team's squad), with the page's tier and level filters.
- Not claimed: that signing the suggested type fixes the weakness. Players adapt to teams (V4b), and there is no test here of signings changing team metrics.
