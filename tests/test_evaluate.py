import numpy as np
import pandas as pd
import pytest

from pbp2track.evaluate import (heldout, hull_area, ks, overlap_rate, pick_held_out, real_directions, real_release,
                                speeds, table, to_attack_frame)
from pbp2track.tables import read_table


def test_ks():
    a = np.arange(100.0)
    assert ks(a, a) == 0.0
    assert ks(a, a + 1000) == 1.0
    assert np.isnan(ks(np.array([]), a))


def test_hull_area():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]], float)
    assert hull_area(square) == pytest.approx(100.0)
    assert hull_area(np.array([[0, 0], [1, 1], [2, 2], [3, 3]], float)) == pytest.approx(0.0)
    assert np.isnan(hull_area(np.array([[0, 0], [1, 1]], float)))
    assert hull_area(np.array([[0, 0], [4, 0], [0, 3], [np.nan, 1]], float)) == pytest.approx(6.0)


def test_overlap_rate():
    spread = np.tile(np.array([[k * 5.0, 0.0] for k in range(10)]), (20, 1, 1))
    assert overlap_rate(spread) == 0.0
    stacked = spread.copy()
    stacked[:, 1] = stacked[:, 0] + 0.5
    assert overlap_rate(stacked) == 1.0


def test_speeds():
    T = 26
    pos = np.zeros((T, 10, 2))
    pos[:, :, 0] = np.arange(T)[:, None] * 0.24  # 6 ft/s at 25 Hz
    v = speeds(pos, np.ones(T, bool))
    assert len(v) == 250 and np.allclose(v, 6.0)


def test_to_attack_frame():
    p = to_attack_frame(np.array([10.0, 20.0]), np.array([10.0, 20.0]))
    assert p.tolist() == [84.0, 30.0]
    q = to_attack_frame(np.array([80.0, 20.0]), np.array([80.0, 20.0]))
    assert q.tolist() == [80.0, 20.0]


def test_real_release():
    T = 60
    shooter = np.zeros((T, 2))
    ball = np.zeros((T, 3))
    ball[30:, 0] = np.arange(T - 30) * 1.0 + 0.5  # leaves the hands at frame 30
    assert real_release(ball, shooter, 40) == 32


def test_pick_held_out():
    ids = [f"{i:010d}" for i in range(50)]
    a = pick_held_out(ids, n=10, seed=7, include=["0000000003"])
    assert a[0] == "0000000003" and len(a) == 10 and len(set(a)) == 10
    assert a == pick_held_out(ids, n=10, seed=7, include=["0000000003"])


def test_real_directions(library_root, held_game):
    P = read_table(library_root, "possessions")
    P["game_id"] = P["game_id"].astype(str)
    d = real_directions(P, held_game["sportvu"]["gameid"])
    assert d == {1: True, 2: True, 3: False, 4: False}


@pytest.fixture(scope="module")
def summary(library_root, held_id):
    return heldout(library_root, held=[held_id], log=lambda s: None)


def test_heldout_beats_the_baseline(summary):
    ours, base = summary["ours"], summary["baseline_retrieval_only"]
    assert ours["shots_matched"] > 20
    assert ours["shooter_err_ft_median"] < 0.5
    assert base["shooter_err_ft_median"] > 5.0
    assert ours["closest_defender_ft"]["mae"] < base["closest_defender_ft"]["mae"]
    assert ours["speed_ft_s"]["ks"] < 0.3
    assert 0.0 <= ours["overlap_rate"]["rebuilt"] <= 1.0


def test_table_renders(summary):
    text = table(summary)
    assert "pbp2track" in text and "shooter error, median ft" in text
    assert str(summary["ours"]["shooter_err_ft_median"]) in text
