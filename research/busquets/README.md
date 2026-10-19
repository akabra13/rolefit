# What defines Sergio Busquets? A signature-metric study

Motivation: RoleFit's "plays like" lists put Busquets next to possession-side holding midfielders such as Stambouli and Coquelin. Football people would not. This study asks which measurable behaviours actually set him apart, and whether they are his own or Barcelona's.

## Data

- **StatsBomb Open Data (discovery):** about 415 Busquets matches, 2008/09 to 2023: 13 La Liga seasons with Barcelona (Barcelona's matches only, except La Liga 2015/16 which is complete), 12 matches for Spain (World Cup 2018, Euro 2020, World Cup 2022), 6 for Inter Miami. 44 of those matches have 360 freeze-frames (La Liga 2020/21, Euro 2020, World Cup 2022, MLS 2023).
- **Comparison pool:** men's holding midfielders (most events as a defensive midfielder in a match, at least 45 minutes): about 350 players in the complete 2015/16 big-five seasons, plus tournaments; with 360 data about 130 player-tournaments.
- **Replication (planned):** the Wyscout public dataset (Pappalardo et al. 2019, *Scientific Data*, CC BY 4.0): full big-five seasons, the 2018 World Cup and Euro 2016, from a different provider. Metrics that can be rebuilt from Wyscout events are re-tested there.

## Candidate metrics (`metrics.py`)

Defined before looking at any Busquets result. Event-based: forward pass after receiving, forward pass after receiving a forward ball (half-turn proxy), bounce pass back to the sender, redirection of play to the other flank, central line-breaking passes from his own half, share of the team's own-third build-ups he touches, counter-stopping fouls (within 8 s of losing the ball), midfield interceptions, receive-and-release within 2 s, completing the next pass after receiving under pressure. 360-based: opponents bypassed per forward pass, passes bypassing two or more, space at the pass end, opponents already behind him when he receives, opponents within 5 units when he receives. Plus the RoleFit stage-1 traits.

## Pre-registered analysis (fixed 2026-10-19, before any Busquets metric was computed)

1. **Reliability:** for every metric, split-half Pearson r across holding midfielders with at least 10 matches in the role (360 metrics: at least 4), alternating matches. Needed: r >= 0.5.
2. **Distinctiveness:** Busquets' pooled rate over his Barcelona holding-midfield matches vs the pool of player rates (players with at least 10 matches, excluding him): z-score and percentile.
3. **Consistency:** the share of his Barcelona seasons (at least 8 matches in the role) in which his rate is on the same side of the pool median as his overall z. Needed: >= 75%.
4. **Portability:** his Spain rate vs the tournament holding-midfield pool. Needed: same side, beyond the 75th (or below the 25th) percentile.
5. **Player vs system:** his rate vs Barcelona's other central and holding midfielders in the same matches (Xavi, Iniesta, Rakitic, etc.). Reported, not a gate: a trait he shares with every Barcelona midfielder is Barcelona's, not his.
6. **Signature metric** = passes 1, |z| >= 1.5, passes 3 and 4.
7. **Retrieval check:** a holding-midfielder-only similarity space built from the signature metrics. Reported: Busquets' nearest neighbours (face validity), and, for every holding midfielder seen at two teams, cross-team re-identification against the current RoleFit space.

Negative results are reported as they come out. If no metric qualifies, that is the finding.

## Results (2026-10-19)

Busquets: 311 Barcelona and 11 Spain holding-midfield matches (>= 45 minutes). Pool: 1,219 other men's holding midfielders.

| Metric | Split-half r | Busquets | Pool mean | z | Pool pct | Seasons same side | Spain pct (tournaments) | Pct among Barca midfielders |
|---|---|---|---|---|---|---|---|---|
| Touch sequences per 90 (involvement) | 0.90 | 85.5 | 57.9 | **+1.71** | 93 | 100% | 94 | 64 |
| Share of own-third build-ups touched | 0.78 | 0.44 | 0.35 | +1.40 | 91 | 100% | 95 | 79 |
| Completes next pass after receiving under pressure | 0.22 | 0.80 | 0.68 | +1.29 | 89 | 100% | 90 | 77 |
| Space at the pass end (360) | 0.45 | 9.95 | 11.17 | -1.04 | 10 | 100% | 16 | - |
| Releases within 2 s of receiving | 0.72 | 0.81 | 0.78 | +0.59 | 72 | 92% | 89 | 71 |
| Midfield interceptions per 90 | 0.33 | 0.65 | 0.89 | -0.68 | 28 | 85% | 87 | 100 |
| Forward pass after a forward ball (half-turn) | 0.48 | 0.45 | 0.48 | -0.44 | 32 | 77% | 69 | 79 |
| Forward pass after receiving | 0.68 | 0.47 | 0.49 | -0.29 | 34 | 69% | 44 | 93 |
| Bounce pass to the sender | 0.63 | 0.16 | 0.15 | +0.45 | 72 | 77% | 11 | 14 |
| Redirects play to the other flank | 0.41 | 0.16 | 0.16 | -0.13 | 45 | 62% | 44 | 36 |
| Central line-breaking passes | 0.55 | 0.034 | 0.035 | -0.02 | 51 | 54% | 38 | 100 |
| Opponents bypassed per forward pass (360) | 0.46 | 2.55 | 2.44 | +0.38 | 67 | 100% | 93 | - |
| Passes bypassing two or more (360) | 0.18 | 0.59 | 0.60 | -0.12 | 50 | 100% | 90 | - |
| Opponents behind him at receipt (360) | 0.70 | 1.26 | 1.43 | -0.42 | 31 | 100% | 67 | - |
| Opponents within 5 units at receipt (360) | 0.50 | 0.39 | 0.42 | -0.30 | 43 | 100% | 90 | - |
| Counter-stopping fouls per 90 | 0.30 | 0.87 | 0.81 | +0.17 | 58 | 46% | 42 | 79 |

**Only one metric meets every pre-registered criterion: involvement (touch sequences per 90), and it is partly Barcelona's.** Among Barcelona's own midfielders in the same matches he sits at the 64th percentile, so his volume is the system's as much as his. Build-up share just misses (z 1.40). Keeping the ball after receiving under pressure is where football people would expect him to stand out, and he does (89th percentile, 100% of seasons, 90th for Spain), but the metric is not reliable match to match (r 0.22), so it fails the gate.

**What he is not, in this data:** a player who turns and plays forward more than others (32nd-34th percentile), who breaks lines more (50th-51st), who redirects play more (45th) or who intercepts more (28th). On every "visible" playmaking action he is average for a holding midfielder.

**Retrieval:** with one qualifying metric the signature space is meaningless (nearest players are anyone with similar volume: Badelj, De Rossi, Andrich). Only 8 holding midfielders in the data have 8+ matches in the role at two teams, too few for a cross-team test.

**Reading.** The event record describes what Busquets does with the ball, and on that he is an above-average, very high-volume, secure holding midfielder, not an outlier. What makes him distinct in football analysis (scanning, body orientation on receipt, positioning that makes him available, timing) happens without an event and needs tracking or video. StatsBomb's 360 frames are a snapshot at each event, visible players only, and give only weak versions of these (the 360 metrics are unreliable over the few tournament matches most comparison players have). The planned Wyscout replication would test the same event-level behaviours from another provider, so it cannot add what is missing; it is deferred.
