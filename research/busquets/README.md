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
