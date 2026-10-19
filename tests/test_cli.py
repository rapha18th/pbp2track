import json
import os

import numpy as np
import pandas as pd
import pytest

from pbp2track import __version__
from pbp2track.cli import main, teams_of
from pbp2track.export import load_rebuild, save_rebuild


def test_teams_of():
    gp = pd.DataFrame({"team_tri_code": ["AAA", "BBB", np.nan], "team_id": [1, 2, np.nan],
                       "game_matchup": ["BBB @ AAA"] * 3})
    assert teams_of(gp) == (1, 2)
    gp["game_matchup"] = "BBB vs. AAA"
    assert teams_of(gp) == (2, 1)
    assert teams_of(gp, 5, 6) == (5, 6)
    with pytest.raises(SystemExit):
        teams_of(pd.DataFrame({"team_id": [1]}))


def test_rebuild_command(library_root, held_game, held_id, tmp_path, capsys):
    csv = tmp_path / "pbp.csv"
    held_game["pbp"].to_csv(csv, index=False)
    prefix = str(tmp_path / "out" / held_id)
    main(["rebuild", "--library", library_root, "--pbp", str(csv), "--game", held_id, "--out", prefix,
          "--exclude-self", "--quiet"])
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["frames"] > 1000 and os.path.exists(report["npz"]) and os.path.exists(report["json"])
    rec = load_rebuild(prefix)
    assert rec["pos"].shape == (report["frames"], 23)
    assert rec["meta"]["game_id"] == held_id and rec["meta"]["home_id"] == held_game["home"]["id"]
    assert rec["lineups"][0]["s"] == 0


def test_save_rebuild_writes_plain_json(tmp_path):
    rec = {"pos": np.zeros((3, 23), np.float32), "gc": np.array([3.0, 2.0, 1.0]), "sc": np.zeros(3), "q": np.ones(3),
           "lineups": [{"s": 0, "e": 2, "pids": [np.int64(1)] * 10}],
           "events": [{"f": 1, "home": np.float64("nan"), "away": np.int64(4), "ok": np.bool_(True)}],
           "pinned": [[] for _ in range(10)], "possessions": []}
    paths = save_rebuild(rec, str(tmp_path / "g"))
    with open(paths["json"], encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["events"][0] == {"f": 1, "home": None, "away": 4, "ok": True}
    assert doc["frames"] == 3


def test_demo_command(tmp_path, capsys):
    main(["demo", "--out", str(tmp_path), "--games", "3", "--held", "1", "--period-s", "90", "--quiet"])
    out = capsys.readouterr().out
    assert "shooter error, median ft" in out
    with open(tmp_path / "eval.json", encoding="utf-8") as f:
        s = json.load(f)
    assert s["ours"]["shots_matched"] > 5
    assert any(f.endswith(".npz") for f in os.listdir(tmp_path / "rebuilt"))


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out
