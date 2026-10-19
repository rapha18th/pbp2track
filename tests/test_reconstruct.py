import numpy as np
import pandas as pd
import pytest

from pbp2track.possessions import names_from_pbp, segment_pbp
from pbp2track.reconstruct import BLEND, best_assignment, reconstruct_game, resample, smoothstep, warp


def fake_clip(end_t=99, tail=25, release_t=80):
    T = end_t + 1 + tail
    t = np.arange(T, dtype=np.float32)
    off = np.stack([np.stack([t, np.full(T, k, np.float32)], axis=1) for k in range(5)], axis=1)
    return {"off": off, "dfn": off + 1, "ball": np.stack([t, t, t], axis=1), "end_t": end_t,
            "release_t": release_t, "shot_t": release_t + 5}


def test_smoothstep_and_resample():
    assert smoothstep(np.array([-1.0, 0.0, 0.5, 1.0, 2.0])).tolist() == [0.0, 0.0, 0.5, 1.0, 1.0]
    a = np.arange(11, dtype=float)
    r = resample(a, 21)
    assert r[0] == 0 and r[-1] == 10 and r[1] == pytest.approx(0.5)
    assert resample(a, 1).tolist() == [0.0]


@pytest.mark.parametrize("seconds,frames", [(4.0, 100), (10.0, 250), (2.0, 50)])
def test_warp_hits_the_logged_length(seconds, frames):
    parts, t = warp(fake_clip(), seconds)
    assert all(len(v) == frames for v in parts.values())
    assert t["end_t"] == frames - 1
    assert 0 <= t["release_t"] <= t["end_t"]
    # the end stays aligned: the last warped frame is the clip's possession end
    assert parts["ball"][-1, 0] == pytest.approx(99.0)


def test_warp_pads_long_and_crops_short():
    _, long = warp(fake_clip(), 10.0)  # ratio capped at 1.35: 135 frames, padded by 115
    assert long["release_t"] == round(80 * 134 / 99) + 115
    parts, short = warp(fake_clip(), 2.0)  # ratio floored at 0.75: 75 frames, the first 25 cropped
    assert short["release_t"] == round(80 * 74 / 99) - 25
    assert parts["ball"][0, 0] > 0


def test_best_assignment_respects_roles_and_distance():
    start = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]], float)
    prev = {101: np.array([39.0, 0.0]), 102: np.array([1.0, 0.0]), 103: None, 104: np.array([21.0, 0.0]),
            105: np.array([11.0, 0.0])}
    out = best_assignment(prev, start, {3: 103})
    assert out[3] == 103
    assert out[4] == 101 and out[0] == 102 and out[2] == 104 and out[1] == 105


@pytest.fixture(scope="module")
def rebuilt(library, held_game):
    gp = held_game["pbp"]
    home, away = held_game["home"]["id"], held_game["away"]["id"]
    return reconstruct_game(gp, home, away, library, names_from_pbp(gp))


def test_rebuilt_shapes(rebuilt, held_game):
    n = len(rebuilt["pos"])
    assert rebuilt["pos"].shape == (n, 23) and n > 1000
    assert len(rebuilt["gc"]) == len(rebuilt["sc"]) == len(rebuilt["q"]) == n
    assert np.isfinite(rebuilt["pos"]).mean() > 0.99
    xy = rebuilt["pos"][:, 3:].reshape(-1, 10, 2)
    assert np.nanmin(xy[..., 0]) > -10 and np.nanmax(xy[..., 0]) < 104
    assert set(np.unique(rebuilt["q"])) <= {1, 2, 3, 4}
    # the clock runs down inside each period
    for q in np.unique(rebuilt["q"]):
        assert np.all(np.diff(rebuilt["gc"][rebuilt["q"] == q]) <= 1e-6)


def test_rebuilt_lineups_cover_every_frame(rebuilt, held_game):
    lus = rebuilt["lineups"]
    assert lus[0]["s"] == 0 and lus[-1]["e"] == len(rebuilt["pos"]) - 1
    assert all(b["s"] == a["e"] + 1 for a, b in zip(lus, lus[1:]))
    home_ids = {p["id"] for p in held_game["home"]["players"]}
    for l in lus:
        assert len(l["pids"]) == 10 and set(l["pids"][:5]) <= home_ids


def test_shooter_is_pinned_at_the_logged_spot(rebuilt, library, held_game):
    home = held_game["home"]["id"]
    poss = segment_pbp(held_game["pbp"], (home, held_game["away"]["id"]))
    errs = []
    for pl in rebuilt["possessions"]:
        if pl["release_f"] is None:
            continue
        p = poss[pl["i"]]
        want = library.court_xy(p["shot_x"], p["shot_y"])
        if (pl["offense"] == home) != (p["period"] <= 2):
            want = np.array([94 - want[0], 50 - want[1]])
        f = pl["release_f"]
        lu = next(l for l in rebuilt["lineups"] if l["s"] <= f <= l["e"])
        k = lu["pids"].index(pl["shooter"])
        errs.append(float(np.hypot(*(rebuilt["pos"][f, 3 + 2 * k: 5 + 2 * k] - want))))
    assert len(errs) > 20
    assert np.median(errs) < 0.05
    # only a release inside the opening blend can drift from its pin
    assert np.mean(np.array(errs) < 0.5) > 0.9


def test_rebuilt_matches_the_truth(rebuilt, held_game):
    # the synthetic truth knows where each shooter really stood
    truth = [t for t in held_game["truth"] if t["shot_xy"] is not None]
    by_actor = {}
    for t in truth:
        by_actor.setdefault(t["actor"], []).append(np.array(t["shot_xy"]))
    errs = []
    for pl in rebuilt["possessions"]:
        if pl["release_f"] is None:
            continue
        f = pl["release_f"]
        lu = next(l for l in rebuilt["lineups"] if l["s"] <= f <= l["e"])
        k = lu["pids"].index(pl["shooter"])
        p = rebuilt["pos"][f, 3 + 2 * k: 5 + 2 * k]
        errs.append(min(float(np.hypot(*(p - s))) for s in by_actor[pl["shooter"]]))
    assert np.median(errs) < 0.5


def test_events_and_pins(rebuilt):
    n = len(rebuilt["pos"])
    assert rebuilt["events"] and all(0 <= e["f"] < n for e in rebuilt["events"])
    kinds = {e["kind"] for e in rebuilt["events"]}
    assert {"shot", "reb", "tov"} <= kinds
    assert sum(len(p) for p in rebuilt["pinned"]) > 20


def test_baseline_mode_runs(library, held_game):
    gp = held_game["pbp"]
    rec = reconstruct_game(gp, held_game["home"]["id"], held_game["away"]["id"], library, pin=False,
                           rng=np.random.default_rng(1))
    assert rec["pos"].shape[1] == 23 and len(rec["possessions"]) > 20


def test_empty_log(library):
    gp = pd.DataFrame([{"action_number": 1, "clock": "PT12M00.00S", "period": 1, "team_id": np.nan, "person_id": 0,
                        "player_name_i": np.nan, "action_type": "period", "subtype": "start", "description": ""}])
    rec = reconstruct_game(gp, 1, 2, library)
    assert rec["pos"].shape == (0, 23) and rec["possessions"] == []


def test_blend_constant():
    assert 0 < BLEND < 50
