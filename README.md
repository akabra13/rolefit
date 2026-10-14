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
3. Team side: demand profiles and influence.
4. Fit: translation model, demand gap, counterfactual (V4, V5).
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
