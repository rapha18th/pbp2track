"""Rebuild a full game's tracking from its play-by-play using the possession library."""
import itertools

import numpy as np
import pandas as pd

from .library import FPS, HOOP, Library
from .possessions import lineups_from_pbp, prepare, segment_pbp

BLEND = 18  # frames to blend player positions across a possession boundary
R_MIN, R_MAX = 0.75, 1.35  # allowed time-warp ratios


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def resample(a, n_out):
    """Linear time resampling of a (T, ...) array to n_out frames."""
    T = a.shape[0]
    if n_out <= 1 or T <= 1:
        return np.repeat(a[:1], max(1, n_out), axis=0)
    src = np.linspace(0, T - 1, n_out)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (src - i0).reshape((-1,) + (1,) * (a.ndim - 1))
    return a[i0] * (1 - w) + a[i1] * w


def warp(clip, target_wall_s):
    """Time-warp a clip so that its possession part lasts target_wall_s, end-aligned."""
    end = clip["end_t"]
    src_len = end + 1
    want = max(10, int(round(target_wall_s * FPS)))
    r = np.clip(want / src_len, R_MIN, R_MAX)
    n1 = max(2, int(round(src_len * r)))
    parts = {k: resample(clip[k][: end + 1], n1) for k in ("off", "dfn", "ball")}
    scale = (n1 - 1) / max(1, end)
    tmap = lambda t: None if t is None else int(round(t * scale))  # noqa: E731
    if n1 > want:  # crop the start, keep the ending aligned
        cut = n1 - want
        parts = {k: v[cut:] for k, v in parts.items()}
        shift = -cut
    elif n1 < want:  # hold the opening shape (inbound, set-up) for the extra time
        pad = want - n1
        parts = {k: np.concatenate([np.repeat(v[:1], pad, axis=0), v]) for k, v in parts.items()}
        shift = pad
    else:
        shift = 0

    def m(t):
        if t is None:
            return None
        return int(np.clip(tmap(t) + shift, 0, want - 1))
    return parts, {"release_t": m(clip["release_t"]), "shot_t": m(clip["shot_t"]), "end_t": want - 1}


def best_assignment(prev_pos, start_pos, fixed):
    """Assign free target players to free clip slots, minimising travel from where they were.

    prev_pos: {player_id: xy or None}; start_pos: (5,2) clip positions at frame 0;
    fixed: {slot: player_id} already decided by role. Returns {slot: player_id}.
    """
    free_slots = [s for s in range(5) if s not in fixed]
    free_players = [p for p in prev_pos if p not in fixed.values()]
    best, best_cost = None, None
    if len(free_players) > len(free_slots):
        free_players = free_players[: len(free_slots)]
    for perm in itertools.permutations(free_slots, len(free_players)):
        cost = 0.0
        for p, s in zip(free_players, perm):
            q = prev_pos[p]
            cost += 0.0 if q is None else float(np.hypot(*(q - start_pos[s])))
        if best_cost is None or cost < best_cost:
            best, best_cost = perm, cost
    out = dict(fixed)
    for p, s in zip(free_players, best or ()):
        out[s] = p
    return out


def reconstruct_game(gp: pd.DataFrame, home_id: int, away_id: int, lib: Library, names: dict = None,
                     pin=True, rng=None, home_right: dict = None):
    """Rebuild one game.

    gp: the game's play-by-play rows (v3 columns).
    names: {person_id: last name}, used to resolve assists written as names.
    pin: apply the event constraints (False gives the random baseline:
         a random library clip with the same outcome, no shot pinning).
    home_right: {period: True if the home team attacks the +x hoop}; default first half +x.
    Returns dict with pos (N,23) court feet, gc, sc, q, lineups, events, pinned, possessions.
    pos columns: ball x, y, z, then x, y for the home five and the away five of the
    current lineup (see lineups for the player ids of each frame range).
    """
    rng = rng or np.random.default_rng(0)
    teams = (int(home_id), int(away_id))
    gp = prepare(gp)
    poss = segment_pbp(gp, teams)
    lus = lineups_from_pbp(gp, teams)
    names = names or {}
    by_last = {}
    for pid, last in names.items():
        by_last.setdefault(str(last).lower(), pid)

    frames_pos, frames_gc, frames_sc, frames_q = [], [], [], []
    lineups, events, pinned = [], [], [[] for _ in range(10)]
    last_pos = {}  # player id -> court xy at the end of the previous block
    used = []
    f0 = 0
    plog = []
    for pi, p in enumerate(poss):
        if p["outcome"] == "period_end" or p["dur"] <= 0.2:
            continue
        an_first = int(p["events"][0]["action_number"]) if p["events"] else None
        lu = lus.get(an_first) or (lineups[-1]["raw"] if lineups else None)
        if not lu:
            continue
        off_team = p["offense"]
        def_team = home_id if off_team == away_id else away_id
        off_players = [x for x in lu.get(off_team, [])][:5]
        def_players = [x for x in lu.get(def_team, [])][:5]
        if len(off_players) < 5 or len(def_players) < 5:
            continue

        shot_xy = lib.court_xy(p["shot_x"], p["shot_y"]) if np.isfinite(p["shot_x"]) else None
        cand, cost = lib.query(p["outcome"], shot_xy, p["shot_class"], p["dur"], p["prev"], k=6, avoid=used[-12:])
        ci = int(cand[0]) if pin else int(rng.choice(np.where(lib.outcome == p["outcome"])[0]))
        used.append(ci)
        clip = lib.clips[ci]
        wall = clip["wall"] * (p["dur"] / clip["live"]) if clip["live"] > 0.5 else clip["wall"]
        parts, t = warp(clip, wall)
        off, dfn, ball = parts["off"].copy(), parts["dfn"].copy(), parts["ball"].copy()
        T = len(ball)

        # attack direction: home attacks +x in the first half, -x after
        hr = home_right.get(p["period"], p["period"] <= 2) if home_right else p["period"] <= 2
        attack_right = (off_team == home_id) == hr

        # roles
        fixed_o, fixed_d = {}, {}
        if clip["shooter"] is not None and p["shooter"] in off_players:
            fixed_o[clip["shooter"]] = p["shooter"]
        if p["assist_name"] and clip["assister"] is not None:
            aid = by_last.get(p["assist_name"].split()[-1].lower())
            if aid in off_players and clip["assister"] not in fixed_o:
                fixed_o[clip["assister"]] = aid
        if clip["handler"] is not None and p["tov_player"] in off_players and clip["handler"] not in fixed_o:
            fixed_o[clip["handler"]] = p["tov_player"]
        if clip["rebounder"] is not None and p["rebounder"] in def_players:
            fixed_d[clip["rebounder"]] = p["rebounder"]

        def norm_prev(pid):
            q = last_pos.get(pid)
            if q is None:
                return None
            return q if attack_right else np.array([94 - q[0], 50 - q[1]])
        ao = best_assignment({x: norm_prev(x) for x in off_players}, off[0], fixed_o)
        ad = best_assignment({x: norm_prev(x) for x in def_players}, dfn[0], fixed_d)

        # constraints
        pins_here = []
        rt = t["release_t"]
        if pin and shot_xy is not None and rt is not None and clip["shooter"] is not None:
            sh = clip["shooter"]
            delta = shot_xy - off[rt, sh]
            w = smoothstep((np.arange(T) - (rt - 60)) / 60.0)[:, None]
            off[:, sh] += w * delta
            dd = np.linalg.norm(dfn[rt] - off[rt, sh], axis=1)
            near = int(np.argmin(dd))
            for k in range(5):
                if k != sh:
                    off[:, k] += w * delta * 0.25
                dfn[:, k] += w * delta * (0.7 if k == near else 0.25)
            # ball: carried with the shooter until release, then eased back onto its real flight to the rim
            arrive = rt + 1 + int(np.argmax(np.linalg.norm(ball[rt + 1:, :2] - HOOP, axis=1) < 1.6)) if rt + 1 < T else T - 1
            arrive = max(arrive, rt + 8)
            wb = w[:, 0].copy()
            post = np.arange(T) > rt
            wb[post] = np.clip(1 - (np.arange(T)[post] - rt) / max(1, arrive - rt), 0, 1)
            ball[:, :2] += wb[:, None] * delta
            pins_here.append((p["shooter"], rt))
        if p["rebounder"]:
            pins_here.append((p["rebounder"], T - 1))
        if p["tov_player"]:
            pins_here.append((p["tov_player"], T - 1))

        # denormalise
        if not attack_right:
            for a in (off, dfn):
                a[..., 0] = 94 - a[..., 0]; a[..., 1] = 50 - a[..., 1]
            ball[:, 0] = 94 - ball[:, 0]; ball[:, 1] = 50 - ball[:, 1]

        # slot order: home five then away five
        pos_by_player = {}
        for s, pid in ao.items():
            pos_by_player[pid] = off[:, s]
        for s, pid in ad.items():
            pos_by_player[pid] = dfn[:, s]
        home_five = off_players if off_team == home_id else def_players
        away_five = def_players if off_team == home_id else off_players
        order = list(home_five) + list(away_five)

        # blend from where each continuing player was
        ramp = 1 - smoothstep(np.arange(T) / BLEND)
        for pid in order:
            if pid in last_pos and pid in pos_by_player:
                d0 = last_pos[pid] - pos_by_player[pid][0]
                if np.hypot(*d0) < 40:
                    pos_by_player[pid] = pos_by_player[pid] + ramp[:, None] * d0

        block = np.zeros((T, 23), np.float32)
        block[:, :3] = ball
        for k, pid in enumerate(order):
            block[:, 3 + 2 * k: 5 + 2 * k] = pos_by_player.get(pid, np.full((T, 2), np.nan))
            last_pos[pid] = block[-1, 3 + 2 * k: 5 + 2 * k].copy()
        frames_pos.append(block)
        frames_gc.append(np.linspace(p["start_c"], p["end_c"], T))
        live = np.linspace(0, p["dur"], T)
        frames_sc.append(np.clip(24 - live, 0, 24))
        frames_q.append(np.full(T, p["period"], np.uint8))
        if not lineups or lineups[-1]["pids"] != order:
            lineups.append({"s": f0, "e": f0 + T - 1, "pids": [int(x) for x in order], "raw": lu})
        else:
            lineups[-1]["e"] = f0 + T - 1
        for pid, tf in pins_here:
            if pid in order:
                pinned[order.index(pid)].append(int(f0 + tf))

        # events inside the possession get frames by clock fraction; the final shot sits at release
        span = max(1e-6, p["start_c"] - p["end_c"])
        for e in p["events"]:
            frac = (p["start_c"] - e["clock_s"]) / span
            fe = f0 + int(np.clip(frac, 0, 1) * (T - 1))
            if e["kind"] == "shot" and rt is not None and int(e.get("person_id") or 0) == p["shooter"]:
                fe = f0 + min(T - 1, rt + 10)
            events.append({"f": int(fe), "period": int(e["period"]), "clock": float(e["clock_s"]), "kind": e["kind"],
                           "team_id": int(e.get("team_id") or 0), "person_id": int(e.get("person_id") or 0),
                           "desc": str(e.get("description") or ""), "value": int(e.get("shot_value") or 0)
                           if pd.notna(e.get("shot_value")) else 0,
                           "home": e.get("score_home"), "away": e.get("score_away")})
        plog.append({"i": pi, "f0": f0, "T": T, "clip": ci, "cost": float(cost[0]), "outcome": p["outcome"],
                     "period": p["period"], "end_c": p["end_c"], "release_f": None if rt is None else f0 + rt,
                     "shooter": p["shooter"], "offense": off_team})
        f0 += T

    if not frames_pos:
        empty = np.zeros(0, np.float32)
        return {"pos": np.zeros((0, 23), np.float32), "gc": empty, "sc": empty.copy(), "q": np.zeros(0, np.uint8),
                "lineups": [], "events": [], "pinned": [[] for _ in range(10)], "possessions": []}
    pos = np.concatenate(frames_pos)
    for l in lineups:
        l.pop("raw", None)
    return {"pos": pos, "gc": np.concatenate(frames_gc), "sc": np.concatenate(frames_sc), "q": np.concatenate(frames_q),
            "lineups": lineups, "events": events, "pinned": [sorted(set(x)) for x in pinned], "possessions": plog}
