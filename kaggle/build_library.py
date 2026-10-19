"""Kaggle script: build the 2015-16 possession library.

Notebook settings: Internet on. Input: the chevronronson/nba-stats-dataset dataset.
Output in /kaggle/working: frames/ (one .npz per game), possessions.parquet,
events.parquet, rosters.parquet, games.csv.

The SportVU logs come from the public research copy at linouk23/NBA-Player-Movements.
They belong to STATS LLC: use them for research and keep them out of public repositories.
"""
import glob
import json
import os
import subprocess
import sys

try:
    import pbp2track  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/rapha18th/pbp2track"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "py7zr"], check=True)

from pbp2track.sportvu import build_library  # noqa: E402

REPO = "/tmp/movements"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/linouk23/NBA-Player-Movements", REPO], check=True)
logs = sorted(glob.glob(os.path.join(REPO, "data", "2016.NBA.Raw.SportVU.Game.Logs", "*.7z")))
logs = [p for p in logs if os.path.getsize(p) > 10_000]  # five archives in the copy are broken (under 10 KB)

pbp = glob.glob("/kaggle/input/**/play_by_play.csv", recursive=True)[0]
report = build_library(logs, pbp, "/kaggle/working", workers=4, game_ids=range(21500001, 21501231),
                       log=lambda s: print(s, flush=True))
with open("/kaggle/working/library_report.json", "w") as f:
    json.dump(report, f, indent=1, default=str)
print(json.dumps(report, default=str)[:2000])
