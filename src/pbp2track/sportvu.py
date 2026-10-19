"""Build a possession library from SportVU tracking logs plus play-by-play.

A SportVU game log is JSON: {"gameid", "events": [{"home", "visitor", "moments"}]}.
Each moment is [period, unix ms, game clock, shot clock, None, entities] and each
entity is [team id, player id, x, y, z] with team id -1 for the ball.
The 2015-16 logs ship as one .7z per game; reading those needs py7zr.

For every game this writes frames/<game_id>.npz:
  unix (n,) ms, q (n,) period, gc (n,) game clock, sc (n,) shot clock,
  ball (n,3) and pxy (n,10,2) as int16 hundredths of a foot (-32768 = missing),
  pid (n,10) player ids, tid (n,10) team ids.
and returns the possession, event and roster rows that build_library() saves as tables.
"""
import glob
import json
import os
import re
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .library import MISSING
from .possessions import clock_s, ft_last, kind, made
from .tables import write_table

HOOPS = np.array([[5.25, 25.0], [88.75, 25.0]])
SCALE = 100.0  # stored units: 0.01 ft
PBP_COLUMNS = ["game_id", "action_number", "clock", "period", "team_id", "team_tri_code", "person_id",
               "player_name_i", "action_type", "subtype", "description", "shot_result", "is_field_goal",
               "shot_value", "shot_distance", "xlegacy", "ylegacy", "score_home", "score_away"]


def gid10(g) -> str:
    """21500438 or '0021500438' -> '0021500438'."""
    return f"{int(g):010d}"


# ---------------------------------------------------------------- reading logs
def load_game(source) -> dict:
    """A SportVU game: an already-loaded dict, a .json path or a .7z path (needs py7zr)."""
    if isinstance(source, dict):
        return source
    path = str(source)
    if path.endswith(".7z"):
        import py7zr
        dest = tempfile.mkdtemp(prefix="pbp2track_")
        try:
            with py7zr.SevenZipFile(path, "r") as z:
                z.extractall(dest)
            jpath = glob.glob(os.path.join(dest, "**", "*.json"), recursive=True)[0]
            with open(jpath) as f:
                return json.load(f)
        finally:
            shutil.rmtree(dest, ignore_errors=True)
    with open(path) as f:
        return json.load(f)


def peek_game_id(source):
    """The 10-digit game id of a log without decoding it (the archive member is <gameid>.json)."""
    if isinstance(source, dict):
        return gid10(source["gameid"])
    path = str(source)
    if path.endswith(".7z"):
        import py7zr
        with py7zr.SevenZipFile(path, "r") as z:
            names = z.getnames()
        m = re.search(r"(\d{10})", " ".join(names))
        return m.group(1) if m else None
    m = re.search(r"(\d{10})", os.path.basename(path))
    if m:
        return m.group(1)
    return gid10(load_game(path)["gameid"])


def timeline(game: dict) -> dict:
    """One continuous 25 Hz timeline per game: moments deduplicated by wall-clock ms."""
    ev0 = game["events"][0]
    roster = []
    for side in ("home", "visitor"):
        t = ev0[side]
        for p in t["players"]:
            roster.append({"game_id": gid10(game["gameid"]), "team_id": t["teamid"], "team": t["abbreviation"],
                           "side": side, "player_id": p["playerid"], "first": p["firstname"], "last": p["lastname"],
                           "jersey": p["jersey"], "position": p["position"]})
    seen = {}
    for e in game["events"]:
        for m in e["moments"]:
            if m[1] not in seen:
                seen[m[1]] = m
    ts = sorted(seen)
    n = len(ts)
    q = np.zeros(n, np.int8)
    gc = np.zeros(n, np.float32)
    sc = np.full(n, np.nan, np.float32)
    ball = np.full((n, 3), np.nan, np.float32)
    pxy = np.full((n, 10, 2), np.nan, np.float32)
    pid = np.zeros((n, 10), np.int32)
    tid = np.zeros((n, 10), np.int32)
    for i, t in enumerate(ts):
        m = seen[t]
        q[i], gc[i] = m[0], m[2]
        if m[3] is not None:
            sc[i] = m[3]
        k = 0
        for en in m[5]:
            if en[0] == -1:
                ball[i] = en[2:5]
            elif k < 10:
                tid[i, k], pid[i, k] = en[0], en[1]
                pxy[i, k] = en[2:4]
                k += 1
    return {"game_id": gid10(game["gameid"]), "home_id": int(ev0["home"]["teamid"]), "roster": roster,
            "unix": np.array(ts, np.int64), "q": q, "gc": gc, "sc": sc, "ball": ball, "pxy": pxy, "pid": pid, "tid": tid}


def save_frames(path: str, tl: dict):
    np.savez_compressed(path, unix=tl["unix"], q=tl["q"], gc=tl["gc"].astype(np.float32), sc=tl["sc"],
                        ball=np.nan_to_num(tl["ball"] * SCALE, nan=MISSING).astype(np.int16),
                        pxy=np.nan_to_num(tl["pxy"] * SCALE, nan=MISSING).astype(np.int16),
                        pid=tl["pid"], tid=tl["tid"])


# ---------------------------------------------------------------- joining the log to frames
def align(game_pbp: pd.DataFrame, tl: dict):
    """Give each play-by-play row a frame.

    First frame in the period where the clock reaches the logged clock + 1 s, then one
    per-game offset: the median gap between that frame and the ball's first pass
    through a rim on made field goals.
    Returns (rows with clock_s, kind, frame0 and frame; offset frames; offsets used).
    """
    q, gc, ball = tl["q"], tl["gc"], tl["ball"]
    n = len(q)
    gp = game_pbp.drop_duplicates("action_number").sort_values(["period", "action_number"]).reset_index(drop=True)
    if "clock_s" not in gp:
        gp["clock_s"] = gp["clock"].map(clock_s)
    gp["kind"] = gp["action_type"].map(kind)
    frame0 = np.full(len(gp), -1, np.int64)
    for p in np.unique(q):
        idx = np.where(q == p)[0]
        sel = gp.index[gp["period"] == p]
        for j in sel:
            c = gp.at[j, "clock_s"]
            hit = idx[gc[idx] < c + 1.0]
            frame0[j] = hit[0] if len(hit) else -1
    gp["frame0"] = frame0

    deltas = []
    for j in gp.index[(gp["kind"] == "shot")]:
        if not made(gp.loc[j]) or gp.at[j, "frame0"] < 0:
            continue
        f0 = gp.at[j, "frame0"]
        lo, hi = max(0, f0 - 150), min(n, f0 + 100)
        b = ball[lo:hi]
        d = np.min(np.linalg.norm(b[:, None, :2] - HOOPS[None], axis=2), axis=1)
        ok = np.where((d < 1.5) & (b[:, 2] > 7.5) & (b[:, 2] < 11.0))[0]
        if len(ok):
            deltas.append(int(lo + ok[0] - f0))
    off = int(np.median(deltas)) if deltas else 0
    gp["frame"] = np.where(gp["frame0"] >= 0, np.clip(gp["frame0"] + off, 0, n - 1), -1)
    return gp, off, deltas


def cut_possessions(gp: pd.DataFrame, tl: dict) -> list:
    """Split the aligned log into possessions on the tracking timeline."""
    gid, home_id, ball, tid = tl["game_id"], tl["home_id"], tl["ball"], tl["tid"]
    poss = []
    cur = None
    start_f = int(gp["frame"][gp["frame"] >= 0].min()) if (gp["frame"] >= 0).any() else 0
    evs = []

    def close(end_f, outcome, period):
        nonlocal cur, start_f, evs
        if cur is not None and end_f > start_f:
            # side of the floor the offense attacks: ball position over the final 40%
            seg = ball[start_f + int(0.6 * (end_f - start_f)):end_f, 0]
            bx = np.nanmean(seg) if np.isfinite(seg).any() else 47.0
            shots = [e for e in evs if e["kind"] == "shot"]
            last_shot = shots[-1] if shots else None
            poss.append({"game_id": gid, "period": int(period), "offense_team_id": int(cur),
                         "offense_is_home": int(cur) == int(home_id),
                         "start_frame": int(start_f), "end_frame": int(end_f),
                         "duration_s": round((end_f - start_f) / 25.0, 2),
                         "outcome": outcome, "n_events": len(evs),
                         "valid": bool(1.0 <= (end_f - start_f) / 25.0 <= 60.0),
                         "mirror": bool(bx < 47.0),
                         "shooter_id": int(last_shot["person_id"]) if last_shot else 0,
                         "shot_value": int(last_shot["shot_value"]) if last_shot and pd.notna(last_shot["shot_value"]) else 0,
                         "shot_x": float(last_shot["xlegacy"]) if last_shot else np.nan,
                         "shot_y": float(last_shot["ylegacy"]) if last_shot else np.nan,
                         "shot_dist": float(last_shot["shot_distance"]) if last_shot else np.nan,
                         "subtype": str(last_shot["subtype"]) if last_shot else "",
                         "label": " | ".join(str(e["description"]) for e in evs if e["kind"] in ("shot", "ft", "tov", "reb"))[:400],
                         "event_numbers": [int(e["action_number"]) for e in evs]})
        start_f = end_f
        evs = []

    for _, r in gp.iterrows():
        k = r["kind"]
        f = int(r["frame"])
        team = int(r["team_id"]) if pd.notna(r["team_id"]) and r["team_id"] else 0
        if k == "period":
            if "end" in str(r["subtype"]).lower() and f >= 0:
                close(f, "period_end", r["period"])
                cur = None
            continue
        if f < 0:
            continue
        if cur is None and team and k in ("shot", "tov", "ft", "jump", "foul", "other"):
            cur = team if k != "foul" else None
            if cur is not None:
                start_f = max(start_f, f - 25 * 5) if not poss else start_f
        if cur is None:
            continue
        evs.append(r.to_dict())
        if k == "shot" and team == cur and made(r):
            close(f, f"made{int(r['shot_value']) if pd.notna(r['shot_value']) else 2}", r["period"])
            cur = None
        elif k == "reb" and team and team != cur:
            close(f, "miss_dreb", r["period"])
            cur = team
        elif k == "tov" and team == cur:
            close(f, "turnover", r["period"])
            cur = None
        elif k == "ft" and team == cur and ft_last(r) and made(r):
            close(f, "ft", r["period"])
            cur = None
        if cur is None:
            # next possession belongs to the other team present on the floor
            teams = [t for t in np.unique(tid[f]) if t and t != team]
            cur = int(teams[0]) if teams else None
    return poss


def build_game(source, game_pbp: pd.DataFrame, frames_dir: str = None):
    """Timeline, aligned events and possessions for one game; writes frames/<gid>.npz when frames_dir is set.

    Returns (stats, roster rows, possession rows, events DataFrame, timeline).
    """
    tl = timeline(load_game(source))
    gp, off, deltas = align(game_pbp, tl)
    poss = cut_possessions(gp, tl)
    gid = tl["game_id"]
    if frames_dir:
        save_frames(os.path.join(frames_dir, f"{gid}.npz"), tl)
    stats = {"game_id": gid, "frames": len(tl["q"]), "pbp_rows": len(gp),
             "mapped": float((gp["frame0"] >= 0).mean()), "offset_frames": off,
             "offset_n": len(deltas),
             "offset_iqr": float(np.subtract(*np.percentile(deltas, [75, 25]))) if len(deltas) > 3 else None,
             "ball_missing": float(np.isnan(tl["ball"][:, 0]).mean()), "possessions": len(poss)}
    events = gp.drop(columns=["clock"], errors="ignore").assign(game_id=gid)
    return stats, tl["roster"], poss, events, tl


def _build_one(source, game_pbp, frames_dir):
    try:
        stats, roster, poss, events, _ = build_game(source, game_pbp, frames_dir)
        return (stats, roster, poss, events), None
    except Exception as e:  # keep going; report the failure
        return None, repr(e)[:300]


def read_pbp(pbp, game_ids=None, chunksize=2_000_000) -> pd.DataFrame:
    """Play-by-play rows (a DataFrame or a CSV path), optionally only the given games."""
    want = None if game_ids is None else {int(g) for g in game_ids}
    if isinstance(pbp, pd.DataFrame):
        df = pbp
        if want is not None:
            df = df[df["game_id"].astype(int).isin(want)]
        return df.copy()
    head = pd.read_csv(pbp, nrows=0).columns
    cols = [c for c in PBP_COLUMNS + ["game_date", "game_matchup"] if c in head]
    parts = []
    for chunk in pd.read_csv(pbp, chunksize=chunksize, usecols=cols, low_memory=False):
        parts.append(chunk if want is None else chunk[chunk["game_id"].astype(int).isin(want)])
    return pd.concat(parts, ignore_index=True)


def build_library(sources, pbp, out_dir: str, workers: int = 1, game_ids=None, log=print) -> dict:
    """Build a possession library directory from SportVU logs and play-by-play.

    sources: SportVU logs (.7z or .json paths, or loaded dicts).
    pbp: play-by-play DataFrame or CSV path (NBA stats v3 columns).
    game_ids: optional filter on the play-by-play (ints or 10-digit strings).
    Writes frames/, possessions, events, rosters and games.csv into out_dir.
    """
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    df = read_pbp(pbp, game_ids)
    if "clock_s" not in df:
        df["clock_s"] = df["clock"].map(clock_s)
    by_gid = {gid10(k): v for k, v in df.groupby("game_id")}
    stats_all, rosters, possessions, events, failed = [], [], [], [], []

    def take(res, name):
        out, err = res
        if err:
            failed.append({"source": name, "error": err})
            return
        st, ro, po, ev = out
        stats_all.append(st)
        rosters.extend(ro)
        possessions.extend(po)
        events.append(ev)

    jobs = []
    for src in sources:
        name = src if not isinstance(src, dict) else gid10(src["gameid"])
        gid = peek_game_id(src)
        if gid is None or gid not in by_gid:
            failed.append({"source": str(name), "error": f"no play-by-play for {gid}"})
            continue
        jobs.append((src, by_gid[gid], str(name)))

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_build_one, s, g, frames_dir): nm for s, g, nm in jobs}
            for i, fu in enumerate(as_completed(futs)):
                take(fu.result(), futs[fu])
                if i % 50 == 0:
                    log(f"build: {i + 1}/{len(jobs)} {futs[fu]}")
    else:
        for i, (s, g, nm) in enumerate(jobs):
            take(_build_one(s, g, frames_dir), nm)
            if i % 50 == 0:
                log(f"build: {i + 1}/{len(jobs)} {nm}")

    stats = pd.DataFrame(stats_all)
    P = pd.DataFrame(possessions)
    E = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    R = pd.DataFrame(rosters).drop_duplicates(["game_id", "player_id"]) if rosters else pd.DataFrame()
    write_table(P, out_dir, "possessions")
    write_table(E.astype({c: str for c in E.columns if E[c].dtype == object}), out_dir, "events")
    write_table(R, out_dir, "rosters")
    stats.to_csv(os.path.join(out_dir, "games.csv"), index=False)
    report = {"games_ok": len(stats), "failed": failed, "possessions": len(P),
              "outcomes": P["outcome"].value_counts().to_dict() if len(P) else {},
              "valid_share": round(float(P["valid"].mean()), 4) if len(P) else 0.0,
              "mapped_mean": round(float(stats["mapped"].mean()), 4) if len(stats) else 0.0,
              "offset_frames_median": float(stats["offset_frames"].median()) if len(stats) else 0.0}
    log(f"library: {report['games_ok']} games, {report['possessions']} possessions, {len(failed)} failed")
    return report
