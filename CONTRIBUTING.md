# Contributing

Thank you for helping. Issues and pull requests are welcome.

## Set up

```bash
git clone https://github.com/rapha18th/pbp2track
cd pbp2track
python -m venv .venv
. .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[all]"
pytest
```

## Ground rules

- Tests run on the synthetic league in `pbp2track.synth`. Add a synthetic case for every change in behaviour.
- Keep data out of the repository: tracking logs, play-by-play, libraries and rebuilt games stay local. The `.gitignore` covers the usual files.
- Report evaluation changes with the full table from `pbp2track evaluate` on the 40 held-out games (`--games 40 --seed 7 --include 0021500438 --max-clips 90000`), before and after.
- Keep the core on numpy and pandas. Optional extras carry anything heavier.
- One topic per pull request, with a short note on what changed and why.

## Good first issues

- An offensive rebound as its own possession boundary.
- A learned refiner on top of the retrieved clip.
- Libraries from other tracking sources written in the SportVU-style JSON.
