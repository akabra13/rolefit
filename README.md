# rolefit

*Who plays this player's role, and who could do it somewhere else?*

Third project in the PitchGraph programme (after pitchgraph v3 and dynamic_shift). Depends on pitchgraph's Parquet lake (StatsBomb Open Data on Drive). No data lives in this repo.

## Role vector (`roles.py`)

One vector per player-team-season (unit), built only from shares so a high-possession team's player does not look different from touching the ball more: four 12x8 pitch-grid distributions (where they act, receive, send passes, defend) plus 9 style numbers (pass length, forward direction, long-pass share, action composition). Embedding: standardise, give each block equal total weight, PCA to 20 components, cosine similarity.

## Pre-registered: does the vector identify the player? (written before running)

Hyperparameters fixed in advance: 20 PCA components, cosine similarity, grid 12x8. The embedding is unsupervised (no labels used).

1. **Split-half re-ID.** Units with >= 900 minutes. Each unit's matches are sorted by date and alternated into halves A and B; vectors are built from each. Query = half A, gallery = all half-B vectors across the lake. Correct = same unit. **Pass bar:** top-5 >= 0.50 AND top-5 >= 2x the position-label baseline's top-5 (guessing a random gallery row with the same position group).
2. **Cross-team re-ID** (the key test: does the role belong to the player or to the system?). Units with >= 270 minutes. Query = a unit's full-season vector, gallery = all units; correct = the same player with a different team (club <-> country, or transfers); the query itself and the player's same-team seasons are removed from the candidate list. **Pass bar:** top-5 >= 2x the position-label baseline's top-5 AND the 90% Wilson CI lower bound of top-5 is above that baseline.

Anything else is reported as a failure, plainly. No changes to features or thresholds after seeing results. Minutes are approximated from lineup position spells (open-ended spells counted as 90).

Later (not yet built): contrastive embedding, system similarity from the v3 fingerprint, counterfactual swap of a player into a team's chain, face-validity lists, the Library pages.
