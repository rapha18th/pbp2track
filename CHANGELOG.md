# Changelog

## 0.1.0 (2026-10-19)

First release, extracted from Ghost Broadcast.

- Possessions and lineups from NBA stats v3 play-by-play: clock-based segmentation, starters per period, `SUB: X FOR Y` rows resolved by surname, `in` and `out` rows paired at the same clock.
- Library builder: SportVU logs (`.7z` or `.json`) joined to the play-by-play, one 25 Hz timeline per game, possessions with outcome, roles and attack side.
- Reconstruction: retrieval by outcome, shot spot, shot family, length and previous outcome; role assignment; time warp; shot pinning; attack direction; stitching across possessions.
- Held-out evaluation: shooter error, closest defender, offensive spacing, speed KS and overlap, against a same-outcome random baseline. On 40 held-out games and 4,548 shots: shooter error median 3.38 ft against 17.11 ft for the baseline.
- Command line: `rebuild`, `evaluate`, `build-library`, `demo`.
- Synthetic league for tests and the demo; the test suite runs on invented data.
- Kaggle scripts for the library and the evaluation, and a bundler for offline notebooks.
