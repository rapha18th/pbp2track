"""The possession library: real tracked possessions as normalised clips.

Normalised frame: the offense attacks the hoop at (88.75, 25). Units: feet.
Each clip keeps 5 offense and 5 defense tracks, the ball, the shot moment and
the role slots (shooter, assister, rebounder, ball handler at a turnover).

A library on disk is the output of `pbp2track build-library`:
  frames/<game_id>.npz   25 Hz tracking per game (see pbp2track.sportvu)
  possessions            one row per possession with frame range and outcome
  events                 play-by-play rows with their tracking frame
  rosters                players per game
  games.csv              one row per game built
"""
import os
import re
from collections import OrderedDict

import numpy as np
import pandas as pd

from .possessions import shot_class
from .tables import read_table

HOOP = np.array([88.75, 25.0])
FPS = 25
TAIL = 25  # frames kept after the possession ends (follow-through, rebound)
MISSING = -32768


def decode_frames(z) -> dict:
    """Stored int16 hundredths of a foot -> float feet with NaN for missing samples."""
    ball = np.asarray(z["ball"]).astype(np.float32)
    ball[ball == MISSING] = np.nan
    pxy = np.asarray(z["pxy"]).astype(np.float32)
    pxy[pxy == MISSING] = np.nan
    return {"ball": ball / 100.0, "pxy": pxy / 100.0, "pid": np.asarray(z["pid"]), "tid": np.asarray(z["tid"]),
            "gc": np.asarray(z["gc"]), "q": np.asarray(z["q"])}


class FrameStore:
    """Lazy per-game frame loader with a small LRU cache."""

    def __init__(self, frames_dir, cap=8):
        self.dir = frames_dir
        self.cap = cap
        self.cache = OrderedDict()

    def get(self, gid):
        if gid in self.cache:
            self.cache.move_to_end(gid)
            return self.cache[gid]
        with np.load(os.path.join(self.dir, f"{gid}.npz")) as z:
            g = decode_frames(z)
        self.cache[gid] = g
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
        return g


def _fill(a):
    """Forward/back fill NaNs along time (axis 0)."""
    a = a.copy()
    flat = a.reshape(a.shape[0], -1)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        ok = np.isfinite(col)
        if ok.all() or not ok.any():
            if not ok.any():
                col[:] = 0
            continue
        idx = np.where(ok, np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        col[:] = col[idx]
        first = np.argmax(ok)
        col[:first] = col[first]
    return flat.reshape(a.shape)


class Library:
    """Possession clips plus the retrieval index.

    Library(root) reads a library directory; Library.from_tables(...) takes the
    tables and a frame store directly. Call build() before querying.
    """

    def __init__(self, root=None, exclude_games=(), max_clips=None, seed=0):
        self.root = root
        if root is not None:
            self._setup(read_table(root, "possessions"), read_table(root, "events"), read_table(root, "rosters"),
                        FrameStore(os.path.join(root, "frames")), exclude_games, max_clips, seed)

    @classmethod
    def from_tables(cls, possessions, events, rosters, store, exclude_games=(), max_clips=None, seed=0):
        lib = cls(None)
        lib._setup(possessions, events, rosters, store, exclude_games, max_clips, seed)
        return lib

    def _setup(self, P, E, rosters, store, exclude_games, max_clips, seed):
        P = P.copy()
        P["game_id"] = P["game_id"].astype(str)
        E = E.copy()
        E["game_id"] = E["game_id"].astype(str)
        E["frame"] = pd.to_numeric(E["frame"], errors="coerce").fillna(-1).astype(int)
        E["action_number"] = pd.to_numeric(E["action_number"], errors="coerce").fillna(-1).astype(int)
        self.rosters = rosters.copy()
        self.rosters["game_id"] = self.rosters["game_id"].astype(str)
        self.store = store
        P = P[P["valid"] & ~P["game_id"].isin(set(exclude_games)) & (P["outcome"] != "period_end")]
        if max_clips and len(P) > max_clips:
            P = P.sample(max_clips, random_state=seed)
        self.P = P.sort_values(["game_id", "start_frame"]).reset_index(drop=True)
        self.E = E.set_index(["game_id", "action_number"]).sort_index()
        self.clips = []
        self.feat = None
        self.calib = None

    # ------------------------------------------------------------ build
    def build(self, log=print):
        names = {}
        for r in self.rosters.itertuples():
            names[(r.game_id, str(r.last).lower())] = int(r.player_id)
        for gi, (gid, gP) in enumerate(self.P.groupby("game_id", sort=False)):
            g = self.store.get(gid)
            last = "start"
            for r in gP.itertuples():
                c = self._clip(gid, g, r, names, last)
                last = r.outcome
                if c is not None:
                    self.clips.append(c)
            if gi % 50 == 0:
                log(f"library: {gi} games, {len(self.clips)} clips")
        self._features()
        log(f"library: {len(self.clips)} clips")
        return self

    def _clip(self, gid, g, r, names, prev):
        s, e = int(r.start_frame), int(r.end_frame)
        n = len(g["q"])
        e2 = min(n - 1, e + TAIL)
        if e - s < 25:
            return None
        tid = g["tid"][s]
        pid = g["pid"][s]
        off_slots = [k for k in range(10) if tid[k] == r.offense_team_id]
        def_slots = [k for k in range(10) if tid[k] != r.offense_team_id]
        if len(off_slots) != 5 or len(def_slots) != 5:
            return None
        # the lineup must hold for the whole clip
        if np.any(g["pid"][e2] != pid):
            return None
        pxy = g["pxy"][s:e2 + 1]
        ball = g["ball"][s:e2 + 1]
        if np.isnan(pxy).mean() > 0.05 or np.isnan(ball[:, 0]).mean() > 0.2:
            return None
        pxy = _fill(pxy)
        ball = _fill(ball)
        if r.mirror:
            pxy = pxy.copy(); ball = ball.copy()
            pxy[..., 0] = 94 - pxy[..., 0]; pxy[..., 1] = 50 - pxy[..., 1]
            ball[:, 0] = 94 - ball[:, 0]; ball[:, 1] = 50 - ball[:, 1]
        off = pxy[:, off_slots]
        dfn = pxy[:, def_slots]
        off_ids = [int(pid[k]) for k in off_slots]
        def_ids = [int(pid[k]) for k in def_slots]

        # shot moment: frame of the last shot event in the possession
        shot_t = None
        evn = list(r.event_numbers)
        for an in reversed(evn):
            key = (gid, int(an))
            if key in self.E.index:
                ev = self.E.loc[key]
                if isinstance(ev, pd.DataFrame):
                    ev = ev.iloc[0]
                if ev["kind"] == "shot" and ev["frame"] >= 0:
                    shot_t = int(ev["frame"]) - s
                    break
        T = len(ball)
        shooter_slot = off_ids.index(int(r.shooter_id)) if int(r.shooter_id) in off_ids else None
        release_t = None
        shot_xy = None
        if shot_t is not None and 0 <= shot_t < T and shooter_slot is not None:
            # release: last frame up to 1.5 s before the logged shot time where the ball is in the shooter's hands
            lo = max(0, shot_t - 40)
            d = np.linalg.norm(ball[lo:shot_t + 10, :2] - off[lo:shot_t + 10, shooter_slot], axis=1)
            near = np.where(d < 3.0)[0]
            release_t = lo + int(near[-1]) if len(near) else shot_t
            shot_xy = off[release_t, shooter_slot].copy()
        ast_slot = None
        m = re.search(r"\((?:[A-Z]\. )?([A-Za-z'\-\. ]+?) \d+ AST\)", str(r.label))
        if m and shooter_slot is not None:
            aid = names.get((gid, m.group(1).split()[-1].lower()))
            ast_slot = off_ids.index(aid) if aid in off_ids else None
        end_t = e - s
        bd = np.linalg.norm(ball[end_t, :2][None] - np.concatenate([off[end_t], dfn[end_t]]), axis=1)
        reb_slot = int(np.argmin(bd[5:])) if r.outcome == "miss_dreb" else None
        tov_slot = int(np.argmin(bd[:5])) if r.outcome == "turnover" else None
        gc = g["gc"]
        live = float(max(0.0, gc[s] - gc[e])) if g["q"][s] == g["q"][e] else (e - s) / FPS
        return {
            "gid": gid, "s": s, "outcome": r.outcome, "prev": prev,
            "off": off.astype(np.float32), "dfn": dfn.astype(np.float32), "ball": ball.astype(np.float32),
            "off_ids": off_ids, "def_ids": def_ids, "end_t": end_t, "shot_t": shot_t, "release_t": release_t,
            "shooter": shooter_slot, "assister": ast_slot, "rebounder": reb_slot, "handler": tov_slot,
            "shot_xy": shot_xy, "shot_class": shot_class(r.subtype, r.label) if r.subtype else "",
            "shot_value": int(r.shot_value), "legacy": (float(r.shot_x), float(r.shot_y)),
            "live": live, "wall": (e - s) / FPS,
        }

    def _features(self):
        f = []
        for c in self.clips:
            sx, sy = (c["shot_xy"] if c["shot_xy"] is not None else (np.nan, np.nan))
            f.append((sx, sy, c["live"], c["wall"]))
        self.feat = np.array(f, np.float32).reshape(-1, 4)
        self.outcome = np.array([c["outcome"] for c in self.clips])
        self.sclass = np.array([c["shot_class"] for c in self.clips])
        self.prev = np.array([c["prev"] for c in self.clips])

    # ------------------------------------------------------------ calibration
    def legacy_to_court(self):
        """Fit legacy shot coordinates (tenths of a foot from the hoop) to the normalised court.

        Sets and returns the fit: x = 88.75 + a*ylegacy/10 + dx, y = 25 + b*xlegacy/10 + dy.
        """
        L, R = [], []
        for c in self.clips:
            if c["shot_xy"] is None or not np.isfinite(c["legacy"][0]):
                continue
            L.append(c["legacy"]); R.append(c["shot_xy"])
        if not L:
            raise ValueError("no clips with a located shot: build() the library first")
        L = np.array(L) / 10.0
        R = np.array(R)
        best = None
        for a in (-1, 1):
            for b in (-1, 1):
                px = HOOP[0] + a * L[:, 1]
                py = HOOP[1] + b * L[:, 0]
                err = np.median(np.hypot(px - R[:, 0], py - R[:, 1]))
                if best is None or err < best[0]:
                    best = (err, a, b, float(np.median(R[:, 0] - px)), float(np.median(R[:, 1] - py)))
        self.calib = best[1:]
        return {"median_err_ft": float(best[0]), "a": best[1], "b": best[2], "dx": best[3], "dy": best[4], "n": len(L)}

    def court_xy(self, xl, yl):
        """Legacy shot coordinates -> normalised court feet (offense attacking +x)."""
        if self.calib is None:
            self.legacy_to_court()
        a, b, dx, dy = self.calib
        return np.array([HOOP[0] + a * yl / 10.0 + dx, HOOP[1] + b * xl / 10.0 + dy])

    # ------------------------------------------------------------ retrieval
    def query(self, outcome, shot_xy=None, sclass="", live=None, prev="", k=8, avoid=()):
        """Rank clips with the same outcome by shot spot, shot family, length and previous outcome."""
        cand = np.where(self.outcome == outcome)[0]
        if len(cand) == 0:
            cand = np.arange(len(self.clips))
        cost = np.zeros(len(cand), np.float32)
        F = self.feat[cand]
        if shot_xy is not None and np.isfinite(shot_xy).all():
            d = np.hypot(F[:, 0] - shot_xy[0], F[:, 1] - shot_xy[1])
            cost += np.where(np.isfinite(d), d, 40.0) * 1.0
        if sclass:
            cost += (self.sclass[cand] != sclass) * 6.0
        if live is not None:
            cost += np.abs(F[:, 2] - live) * 0.6
        if prev:
            cost += ((self.prev[cand] == "miss_dreb") != (prev == "miss_dreb")) * 3.0
            cost += ((self.prev[cand] == "turnover") != (prev == "turnover")) * 2.0
        if avoid:
            cost += np.isin(cand, list(avoid)) * 25.0
        order = np.argsort(cost)[:k]
        return cand[order], cost[order]
