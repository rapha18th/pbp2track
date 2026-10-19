"""Held-out evaluation: rebuild games that have real tracking and compare, shot by shot.

Games are held out of the library, rebuilt from their play-by-play alone, then compared
with their own tracking at every matched shot release:
  shooter error        distance between the rebuilt and real shooter at release
  closest defender     distance from the shooter to the nearest defender, real vs rebuilt
  offense spacing      convex hull area of the five offensive players at release
  speed                distribution of player speeds over live play (two-sample KS)
  overlap rate         share of sampled frames where two players stand within 1 ft
The baseline draws a random library possession with the same outcome and applies no
event constraints.
"""
import json
import os

import numpy as np
import pandas as pd

from .library import FPS, FrameStore, Library
from .possessions import names_from_pbp
from .reconstruct import reconstruct_game
from .tables import read_table


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def hull_area(pts):
    """Area of the convex hull of (n,2) points (monotone chain); NaN below 3 points."""
    pts = np.asarray(pts, float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        return np.nan
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(P):
        h = []
        for p in P:
            while len(h) >= 2 and _cross(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h
    lo, up = half(pts), half(pts[::-1])
    poly = np.array(lo[:-1] + up[:-1])
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def real_release(ball, shooter_xy, shot_f):
    """Last frame within 40 frames before the logged shot (to 10 after) with the ball in the shooter's hands."""
    lo = max(0, shot_f - 40)
    d = np.linalg.norm(ball[lo:shot_f + 10, :2] - shooter_xy[lo:shot_f + 10], axis=1)
    near = np.where(d < 3.0)[0]
    return lo + int(near[-1]) if len(near) else shot_f


def speeds(pos10, valid):
    """Player speeds in ft/s over frames flagged valid (live play), capped at 35 ft/s."""
    v = np.linalg.norm(np.diff(pos10, axis=0), axis=2) * FPS
    v = v[valid[1:]]
    return v[np.isfinite(v) & (v < 35)]


def overlap_rate(pos10):
    """Share of every fifth frame where two players stand within 1 ft of each other."""
    n = 0
    hits = 0
    for t in range(0, len(pos10), 5):
        p = pos10[t]
        p = p[np.isfinite(p).all(axis=1)]
        if len(p) < 2:
            continue
        d = np.linalg.norm(p[:, None] - p[None], axis=2)
        n += 1
        hits += int((d[np.triu_indices(len(p), 1)] < 1.0).any())
    return hits / max(1, n)


def ks(a, b):
    """Two-sample Kolmogorov-Smirnov statistic."""
    a, b = np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)])
    if not len(a) or not len(b):
        return np.nan
    xs = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, xs, "right") / len(a) - np.searchsorted(b, xs, "right") / len(b))))


def real_directions(possessions: pd.DataFrame, gid: str) -> dict:
    """{period: True if the home team attacked the +x hoop} from the real possessions."""
    P = possessions[(possessions["game_id"] == gid) & possessions["valid"] & possessions["offense_is_home"]]
    return {int(q): bool((~g["mirror"].astype(bool)).mean() >= 0.5) for q, g in P.groupby("period")}


def to_attack_frame(points, shooter):
    """Mirror so the shooter attacks the +x hoop; both sides are compared in this frame."""
    pts = np.array(points, dtype=float)
    if shooter[0] < 47:
        pts = pts.copy()
        pts[..., 0] = 94 - pts[..., 0]
        pts[..., 1] = 50 - pts[..., 1]
    return pts


def evaluate_game(g: dict, shot_events: pd.DataFrame, rec: dict):
    """Compare a rebuilt game with the real tracking of the same game.

    g: real frames (FrameStore.get); shot_events: that game's shot rows with a frame;
    rec: reconstruct_game output. Returns (per-shot DataFrame, game-level dict).
    """
    ev = shot_events[pd.to_numeric(shot_events["frame"], errors="coerce") >= 0]
    rows = []
    pos = rec["pos"]
    for pl in rec["possessions"]:
        if pl["release_f"] is None or not pl["shooter"]:
            continue
        # the real shot: same period, same shooter, nearest clock
        cand = ev[(ev["period"] == pl["period"]) & (pd.to_numeric(ev["person_id"], errors="coerce") == pl["shooter"])]
        if cand.empty:
            continue
        j = (cand["clock_s"] - pl["end_c"]).abs().idxmin()
        if abs(cand.at[j, "clock_s"] - pl["end_c"]) > 3:
            continue
        sf = int(cand.at[j, "frame"])
        slot = np.where(g["pid"][sf] == pl["shooter"])[0]
        if not len(slot):
            continue
        k = int(slot[0])
        rf = real_release(g["ball"], g["pxy"][:, k], sf)
        real_sh = g["pxy"][rf, k]
        real_off = g["pxy"][rf][g["tid"][rf] == pl["offense"]]
        real_def = g["pxy"][rf][g["tid"][rf] != pl["offense"]]
        # rebuilt side: find the shooter's slot in the rebuilt lineup at the release frame
        f = pl["release_f"]
        lu = next(l for l in rec["lineups"] if l["s"] <= f <= l["e"])
        if pl["shooter"] not in lu["pids"]:
            continue
        ks_ = lu["pids"].index(pl["shooter"])
        p10 = pos[f, 3:].reshape(10, 2)
        rec_sh = p10[ks_]
        home_first = ks_ < 5
        rec_off = p10[:5] if home_first else p10[5:]
        rec_def = p10[5:] if home_first else p10[:5]
        real_off, real_def = to_attack_frame(real_off, real_sh), to_attack_frame(real_def, real_sh)
        rec_off, rec_def = to_attack_frame(rec_off, rec_sh), to_attack_frame(rec_def, rec_sh)
        real_sh, rec_sh = to_attack_frame(real_sh, real_sh), to_attack_frame(rec_sh, rec_sh)
        rows.append({
            "shooter_err": float(np.hypot(*(rec_sh - real_sh))),
            "def_real": float(np.min(np.linalg.norm(real_def - real_sh, axis=1))),
            "def_rec": float(np.min(np.linalg.norm(rec_def - rec_sh, axis=1))),
            "hull_real": hull_area(real_off), "hull_rec": hull_area(rec_off),
        })
    df = pd.DataFrame(rows, columns=["shooter_err", "def_real", "def_rec", "hull_real", "hull_rec"])
    pos10_rec = pos[:, 3:].reshape(-1, 10, 2)
    real10 = g["pxy"]
    live_real = np.r_[False, np.diff(g["gc"]) < 0]
    live_rec = np.r_[False, np.diff(rec["gc"]) < 0]
    return df, {
        "speed_real": speeds(real10, live_real), "speed_rec": speeds(pos10_rec, live_rec),
        "overlap_real": overlap_rate(real10[live_real]), "overlap_rec": overlap_rate(pos10_rec),
    }


def summarise(shots, sp_real, sp_rec, ov_real, ov_rec) -> dict:
    S = pd.concat(shots, ignore_index=True) if shots else pd.DataFrame(columns=["shooter_err", "def_real", "def_rec", "hull_real", "hull_rec"])
    a_r = np.concatenate(sp_real) if sp_real else np.zeros(0)
    a_c = np.concatenate(sp_rec) if sp_rec else np.zeros(0)
    S = S.astype(float)
    return {
        "shots_matched": int(len(S)),
        "shooter_err_ft_median": round(float(S["shooter_err"].median()), 2),
        "shooter_err_ft_p90": round(float(S["shooter_err"].quantile(0.9)), 2),
        "closest_defender_ft": {"real_mean": round(float(S["def_real"].mean()), 2), "rebuilt_mean": round(float(S["def_rec"].mean()), 2),
                                "mae": round(float((S["def_real"] - S["def_rec"]).abs().mean()), 2),
                                "ks": round(ks(S["def_real"].values, S["def_rec"].values), 3)},
        "offense_spacing_sqft": {"real_mean": round(float(S["hull_real"].mean()), 1), "rebuilt_mean": round(float(S["hull_rec"].mean()), 1),
                                 "ks": round(ks(S["hull_real"].values, S["hull_rec"].values), 3)},
        "speed_ft_s": {"real_mean": round(float(a_r.mean()), 2) if len(a_r) else float("nan"),
                       "rebuilt_mean": round(float(a_c.mean()), 2) if len(a_c) else float("nan"),
                       "ks": round(ks(a_r, a_c), 3)},
        "overlap_rate": {"real": round(float(np.mean(ov_real)), 4) if ov_real else float("nan"),
                         "rebuilt": round(float(np.mean(ov_rec)), 4) if ov_rec else float("nan")},
    }


def pick_held_out(game_ids, n=40, seed=7, include=()):
    """The include list first, then n - len(include) games drawn at random from the rest."""
    include = [str(g) for g in include]
    rng = np.random.default_rng(seed)
    pool = sorted(set(str(g) for g in game_ids) - set(include))
    k = max(0, min(len(pool), n - len(include)))
    return include + [str(x) for x in rng.choice(pool, k, replace=False)]


def heldout(root: str, held=None, n: int = 40, seed: int = 7, include=(), max_clips=None,
            modes=("ours", "baseline_retrieval_only"), log=print) -> dict:
    """Run the held-out comparison on a library directory.

    held: explicit game ids; otherwise pick_held_out(games.csv, n, seed, include).
    The published run used n=40, seed=7, include=['0021500438'], max_clips=90000.
    """
    games = pd.read_csv(os.path.join(root, "games.csv"), dtype={"game_id": str})
    held = [str(g) for g in held] if held is not None else pick_held_out(games["game_id"], n, seed, include)
    log(f"held out: {len(held)} games")
    lib = Library(root, exclude_games=held, max_clips=max_clips).build(log=log)
    calib = lib.legacy_to_court()
    E = read_table(root, "events")
    E["game_id"] = E["game_id"].astype(str)
    R = read_table(root, "rosters")
    R["game_id"] = R["game_id"].astype(str)
    Pall = read_table(root, "possessions")
    Pall["game_id"] = Pall["game_id"].astype(str)
    store = FrameStore(os.path.join(root, "frames"), cap=4)

    summary = {"held_out": held, "library_clips": len(lib.clips), "calibration": calib}
    for mode in modes:
        pin = mode == "ours"
        shots, sp_r, sp_c, ov_r, ov_c = [], [], [], [], []
        for gi, gid in enumerate(held):
            gp = E[E["game_id"] == gid].copy()
            gp["action_number"] = pd.to_numeric(gp["action_number"], errors="coerce")
            ro = R[R["game_id"] == gid]
            home = int(ro[ro["side"] == "home"]["team_id"].iloc[0])
            away = int(ro[ro["side"] == "visitor"]["team_id"].iloc[0])
            rec = reconstruct_game(gp, home, away, lib, names_from_pbp(gp), pin=pin, rng=np.random.default_rng(gi),
                                   home_right=real_directions(Pall, gid))
            df, gl = evaluate_game(store.get(gid), gp[gp["kind"] == "shot"], rec)
            shots.append(df)
            sp_r.append(gl["speed_real"]); sp_c.append(gl["speed_rec"])
            ov_r.append(gl["overlap_real"]); ov_c.append(gl["overlap_rec"])
        summary[mode] = summarise(shots, sp_r, sp_c, ov_r, ov_c)
        log(f"{mode}: " + json.dumps(summary[mode]))
    return summary


def table(summary: dict) -> str:
    """A plain text table of a heldout() summary."""
    o, b = summary.get("ours", {}), summary.get("baseline_retrieval_only", {})

    def g(d, *ks_):
        for k in ks_:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        return d if not isinstance(d, dict) else ""
    rows = [
        ("shots matched", g(o, "shots_matched"), g(b, "shots_matched"), ""),
        ("shooter error, median ft", g(o, "shooter_err_ft_median"), g(b, "shooter_err_ft_median"), ""),
        ("shooter error, p90 ft", g(o, "shooter_err_ft_p90"), g(b, "shooter_err_ft_p90"), ""),
        ("speed, mean ft/s", g(o, "speed_ft_s", "rebuilt_mean"), g(b, "speed_ft_s", "rebuilt_mean"), g(o, "speed_ft_s", "real_mean")),
        ("speed KS vs real", g(o, "speed_ft_s", "ks"), g(b, "speed_ft_s", "ks"), ""),
        ("closest defender MAE ft", g(o, "closest_defender_ft", "mae"), g(b, "closest_defender_ft", "mae"), ""),
        ("spacing, mean sq ft", g(o, "offense_spacing_sqft", "rebuilt_mean"), g(b, "offense_spacing_sqft", "rebuilt_mean"),
         g(o, "offense_spacing_sqft", "real_mean")),
        ("overlap rate", g(o, "overlap_rate", "rebuilt"), g(b, "overlap_rate", "rebuilt"), g(o, "overlap_rate", "real")),
    ]
    w = max(len(r[0]) for r in rows)
    out = [f"{'metric':<{w}}  {'pbp2track':>10}  {'baseline':>10}  {'real':>8}"]
    for name, x, y, z in rows:
        out.append(f"{name:<{w}}  {str(x):>10}  {str(y):>10}  {str(z):>8}")
    return "\n".join(out)
