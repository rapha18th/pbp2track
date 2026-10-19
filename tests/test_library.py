import os

import numpy as np
import pandas as pd
import pytest

from pbp2track.library import FrameStore, Library, _fill
from pbp2track.tables import read_table, write_table


def test_fill_forward_and_back():
    a = np.array([[np.nan], [1.0], [np.nan], [3.0], [np.nan]])
    assert _fill(a)[:, 0].tolist() == [1.0, 1.0, 1.0, 3.0, 3.0]
    assert _fill(np.full((3, 1), np.nan))[:, 0].tolist() == [0.0, 0.0, 0.0]


def test_clip_shapes(library):
    assert len(library.clips) > 100
    for c in library.clips:
        T = len(c["ball"])
        assert c["off"].shape == (T, 5, 2) and c["dfn"].shape == (T, 5, 2) and c["ball"].shape == (T, 3)
        assert np.isfinite(c["off"]).all() and np.isfinite(c["ball"]).all()
        assert 25 <= c["end_t"] < T
        assert len(c["off_ids"]) == 5 and len(c["def_ids"]) == 5


def test_clips_attack_plus_x(library):
    # normalised: the ball ends every made shot at the +x hoop
    made = [c for c in library.clips if c["outcome"].startswith("made")]
    ends = np.array([c["ball"][c["end_t"], :2] for c in made])
    assert np.median(np.hypot(ends[:, 0] - 88.75, ends[:, 1] - 25.0)) < 2.0


def test_roles(library):
    by = {}
    for c in library.clips:
        by.setdefault(c["outcome"], []).append(c)
    for o in ("made2", "made3", "miss_dreb"):
        located = [c for c in by[o] if c["shooter"] is not None and c["release_t"] is not None]
        assert len(located) / len(by[o]) > 0.9
        assert all(c["release_t"] <= c["shot_t"] + 10 for c in located)
    assert all(c["rebounder"] is not None for c in by["miss_dreb"])
    assert all(c["handler"] is not None for c in by["turnover"])
    assert any(c["assister"] is not None for c in by["made2"] + by["made3"])


def test_calibration(library):
    cal = library.legacy_to_court()
    assert (cal["a"], cal["b"]) == (-1, 1)
    assert cal["median_err_ft"] < 0.5


def test_query_ranks_same_outcome(library):
    spot = np.array([71.5, 8.0])
    cand, cost = library.query("made3", spot, "jump", 8.0, "made2", k=5)
    assert all(library.outcome[i] == "made3" for i in cand)
    assert np.all(np.diff(cost) >= 0)
    again, _ = library.query("made3", spot, "jump", 8.0, "made2", k=5, avoid=[int(cand[0])])
    assert int(again[0]) != int(cand[0])


def test_held_out_game_is_excluded(library, held_id):
    assert held_id not in {c["gid"] for c in library.clips}


def test_max_clips_samples(library_root):
    lib = Library(library_root, max_clips=40, seed=1)
    assert len(lib.P) == 40


def test_from_tables_matches_disk(library_root, held_id):
    P, E, R = (read_table(library_root, n) for n in ("possessions", "events", "rosters"))
    lib = Library.from_tables(P, E, R, FrameStore(os.path.join(library_root, "frames")), exclude_games=[held_id])
    lib.build(log=lambda s: None)
    ref = Library(library_root, exclude_games=[held_id]).build(log=lambda s: None)
    assert len(lib.clips) == len(ref.clips)


@pytest.mark.parametrize("fmt", ["pickle", "parquet"])
def test_tables_round_trip(tmp_path, fmt):
    if fmt == "parquet":
        pytest.importorskip("pyarrow")
    df = pd.DataFrame({"game_id": ["0000000001", "0000000002"], "event_numbers": [[1, 2], [3]], "x": [1.5, np.nan]})
    write_table(df, str(tmp_path), "t", fmt=fmt)
    back = read_table(str(tmp_path), "t")
    assert list(back["game_id"]) == ["0000000001", "0000000002"]
    assert [list(v) for v in back["event_numbers"]] == [[1, 2], [3]]


def test_missing_table_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_table(str(tmp_path), "possessions")
