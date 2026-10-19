import os

import numpy as np
import pandas as pd

from pbp2track import synth
from pbp2track.library import FrameStore
from pbp2track.sportvu import build_game, build_library, gid10, peek_game_id, read_pbp, timeline
from pbp2track.tables import read_table


def tiny_log():
    team = {"name": "Team", "teamid": 7, "abbreviation": "T07",
            "players": [{"lastname": f"P{i}", "firstname": "X", "playerid": 700 + i, "jersey": str(i), "position": "G"}
                        for i in range(5)]}
    other = {**team, "teamid": 8, "abbreviation": "T08",
             "players": [{**p, "playerid": p["playerid"] + 100} for p in team["players"]]}

    def moment(i):
        ents = [[-1, -1, 10.0 + i, 20.0, 5.0]]
        ents += [[7, 700 + k, 1.0 * k, 2.0, 0.0] for k in range(5)]
        ents += [[8, 800 + k, 50.0 + k, 3.0, 0.0] for k in range(5)]
        return [1, 1_000_000 + 40 * i, 720.0 - i / 25, 24.0, None, ents]
    # two events that share frames 2 and 3, as SportVU logs do
    return {"gameid": "0012300001", "events": [
        {"eventId": "1", "home": team, "visitor": other, "moments": [moment(i) for i in range(4)]},
        {"eventId": "2", "home": team, "visitor": other, "moments": [moment(i) for i in range(2, 6)]},
    ]}


def test_gid10():
    assert gid10(21500438) == "0021500438"
    assert gid10("0021500438") == "0021500438"


def test_timeline_deduplicates_moments():
    tl = timeline(tiny_log())
    assert len(tl["q"]) == 6
    assert tl["home_id"] == 7 and tl["game_id"] == "0012300001"
    assert np.allclose(tl["ball"][:, 0], 10.0 + np.arange(6))
    assert list(tl["pid"][0]) == [700, 701, 702, 703, 704, 800, 801, 802, 803, 804]
    assert np.all(np.diff(tl["gc"]) < 0)
    assert len(tl["roster"]) == 10


def test_build_game_matches_truth(held_game):
    stats, roster, poss, events, tl = build_game(held_game["sportvu"], held_game["pbp"])
    truth = held_game["truth"]
    cut = [p for p in poss if p["outcome"] != "period_end"]
    assert [p["outcome"] for p in cut] == [t["outcome"] for t in truth]
    assert stats["mapped"] == 1.0
    assert 20 <= stats["offset_frames"] <= 26
    # the side of the floor each possession attacks
    assert [p["mirror"] for p in cut] == [not t["attack_right"] for t in truth]
    # possession ends land within two frames of the simulated ones
    ends = np.array([p["end_frame"] for p in cut]) - np.array([t["end_frame"] for t in truth])
    assert np.abs(ends).max() <= 2
    assert set(events["kind"]) >= {"shot", "reb", "tov", "period"}


def test_build_library_writes_tables(built, games):
    root, report = built
    assert report["games_ok"] == len(games) and not report["failed"]
    for g in games:
        assert os.path.exists(os.path.join(root, "frames", f"{g['sportvu']['gameid']}.npz"))
    P = read_table(root, "possessions")
    assert len(P) == report["possessions"]
    assert P["valid"].all()
    E = read_table(root, "events")
    assert (pd.to_numeric(E["frame"]) >= 0).all()
    games_csv = pd.read_csv(os.path.join(root, "games.csv"), dtype={"game_id": str})
    assert sorted(games_csv["game_id"]) == sorted(g["sportvu"]["gameid"] for g in games)


def test_frames_round_trip(built, held_game):
    root, _ = built
    g = FrameStore(os.path.join(root, "frames")).get(held_game["sportvu"]["gameid"])
    tl = timeline(held_game["sportvu"])
    assert g["pxy"].shape == tl["pxy"].shape
    assert np.nanmax(np.abs(g["pxy"] - tl["pxy"])) < 0.0101  # stored as whole hundredths of a foot


def test_build_from_files(tmp_path):
    games = synth.league(1, seed=11, period_s=90.0)
    raw = synth.write_league(games, str(tmp_path / "raw"))
    assert peek_game_id(raw["logs"][0]) == games[0]["sportvu"]["gameid"]
    pbp = read_pbp(raw["pbp"], [games[0]["sportvu"]["gameid"]])
    assert len(pbp) == len(games[0]["pbp"])
    report = build_library(raw["logs"], raw["pbp"], str(tmp_path / "lib"), log=lambda s: None)
    assert report["games_ok"] == 1 and report["possessions"] > 10


def test_missing_play_by_play_is_reported(tmp_path, held_game):
    empty = held_game["pbp"].iloc[:0]
    report = build_library([held_game["sportvu"]], empty, str(tmp_path), log=lambda s: None)
    assert report["games_ok"] == 0 and len(report["failed"]) == 1
