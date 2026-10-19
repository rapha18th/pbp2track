"""Possessions and on-court lineups from play-by-play alone (NBA stats v3 style rows).

Expected columns: action_number, clock (or clock_s), period, team_id, person_id,
player_name_i, action_type, subtype, description, shot_result, shot_value,
xlegacy, ylegacy. score_home and score_away are carried along when present.
"""
import re

import numpy as np
import pandas as pd


def clock_s(v) -> float:
    """'PT11M41.00S' -> 701.0 seconds left in the period."""
    m = re.match(r"PT(\d+)M([\d.]+)S", str(v))
    return float(m.group(1)) * 60 + float(m.group(2)) if m else np.nan


def kind(action_type) -> str:
    """Map an action_type to one of: shot, ft, reb, tov, period, sub, jump, foul, other."""
    t = str(action_type).lower()
    if t in ("made shot", "missed shot", "2pt", "3pt"):
        return "shot"
    if "free" in t:
        return "ft"
    if "rebound" in t:
        return "reb"
    if "turnover" in t:
        return "tov"
    if "period" in t:
        return "period"
    if "substitution" in t:
        return "sub"
    if "jump" in t:
        return "jump"
    if "foul" in t:
        return "foul"
    return "other"


def made(row) -> bool:
    r = str(row.get("shot_result", "")).lower()
    if r in ("made", "1", "true"):
        return True
    if r in ("missed", "0", "false"):
        return False
    return not str(row.get("description", "")).upper().startswith("MISS")


def ft_last(row) -> bool:
    m = re.search(r"(\d)\s*of\s*(\d)", f"{row.get('subtype', '')} {row.get('description', '')}")
    return (m is None) or m.group(1) == m.group(2)


SHOT_CLASS = [
    ("dunk", "rim"), ("layup", "rim"), ("tip", "rim"), ("alley", "rim"), ("finger roll", "rim"),
    ("hook", "hook"), ("fadeaway", "jump"), ("jump", "jump"), ("bank", "jump"),
]


def shot_class(subtype: str, desc: str = "") -> str:
    """Coarse shot family used in retrieval: rim, hook or jump."""
    s = f"{subtype} {desc}".lower()
    for key, cls in SHOT_CLASS:
        if key in s:
            return cls
    return "jump"


def prepare(gp: pd.DataFrame) -> pd.DataFrame:
    gp = gp.drop_duplicates("action_number").copy()
    gp["clock_s"] = gp["clock"].map(clock_s) if "clock_s" not in gp else gp["clock_s"]
    gp["kind"] = gp["action_type"].map(kind)
    return gp.sort_values(["period", "action_number"]).reset_index(drop=True)


def names_from_pbp(gp: pd.DataFrame) -> dict:
    """{person_id: last name} from player_name_i ('S. Curry' -> 'Curry')."""
    out = {}
    col = gp["player_name_i"] if "player_name_i" in gp else gp.get("player_name")
    if col is None:
        return out
    for pid, nm in zip(gp["person_id"], col):
        if pd.notna(pid) and pid and isinstance(nm, str):
            out[int(pid)] = nm.split()[-1]
    return out


def segment_pbp(gp: pd.DataFrame, team_ids: tuple) -> list:
    """Split one game's play-by-play into possessions, using the game clock only.

    Returns dicts with offense team, period, start/end clock (seconds left), outcome,
    the events inside, shooter / assister / rebounder ids and the shot location.
    Mirrors the frame-based splitter used to build the library.
    """
    gp = prepare(gp)
    out = []
    cur = None
    start_c = None
    period = None
    evs = []
    prev_outcome = "start"

    def other(t):
        return [x for x in team_ids if x != t][0] if t in team_ids else None

    def close(end_c, outcome):
        nonlocal cur, start_c, evs, prev_outcome
        if cur is not None and start_c is not None:
            shots = [e for e in evs if e["kind"] == "shot"]
            ls = shots[-1] if shots else None
            ast = None
            if ls is not None:
                m = re.search(r"\(([^()]+?) \d+ AST\)", str(ls["description"]))
                ast = m.group(1) if m else None
            reb = next((e for e in reversed(evs) if e["kind"] == "reb"), None)
            tov = next((e for e in reversed(evs) if e["kind"] == "tov"), None)
            out.append({
                "period": int(period), "offense": int(cur), "start_c": float(start_c), "end_c": float(end_c),
                "dur": float(max(0.0, start_c - end_c)), "outcome": outcome, "prev": prev_outcome,
                "events": evs,
                "shooter": int(ls["person_id"]) if ls is not None else 0,
                "assist_name": ast,
                "shot_value": int(ls["shot_value"]) if ls is not None and pd.notna(ls.get("shot_value")) else 0,
                "shot_x": float(ls["xlegacy"]) if ls is not None and pd.notna(ls.get("xlegacy")) else np.nan,
                "shot_y": float(ls["ylegacy"]) if ls is not None and pd.notna(ls.get("ylegacy")) else np.nan,
                "shot_class": shot_class(ls.get("subtype", ""), ls.get("description", "")) if ls is not None else "",
                "rebounder": int(reb["person_id"]) if reb is not None and outcome == "miss_dreb" else 0,
                "tov_player": int(tov["person_id"]) if tov is not None and outcome == "turnover" else 0,
                "n_off_reb": sum(1 for e in evs if e["kind"] == "reb" and int(e.get("team_id") or 0) == cur),
            })
            prev_outcome = outcome
        start_c = end_c
        evs = []

    for r in gp.to_dict(orient="records"):
        k = r["kind"]
        c = r["clock_s"]
        team = int(r["team_id"]) if pd.notna(r.get("team_id")) and r.get("team_id") else 0
        if k == "period":
            if "start" in str(r.get("subtype", "")).lower():
                period, cur, start_c, evs, prev_outcome = r["period"], None, c, [], "start"
            elif "end" in str(r.get("subtype", "")).lower():
                close(c, "period_end")
                cur = None
            continue
        if period is None:
            period, start_c = r["period"], 720.0 if r["period"] <= 4 else 300.0
        if cur is None and team in team_ids and k in ("shot", "tov", "ft", "jump", "other"):
            cur = team
        if cur is None:
            continue
        r["team_id"] = team
        evs.append(r)
        if k == "shot" and team == cur and made(r):
            close(c, f"made{int(r['shot_value']) if pd.notna(r.get('shot_value')) else 2}")
            cur = other(team)
        elif k == "reb" and team and team != cur:
            close(c, "miss_dreb")
            cur = team
        elif k == "tov" and team == cur:
            close(c, "turnover")
            cur = other(team)
        elif k == "ft" and team == cur and ft_last(r) and made(r):
            close(c, "ft")
            cur = other(team)
    return out


def _sub_moves(gp: pd.DataFrame, team_ids: tuple):
    """Normalise substitution rows to (row index, team, out_id, in_id).

    v2 rows: one row 'SUB: <in name> FOR <out name>', person_id = the player leaving.
    v3 rows: separate rows with subtype 'in' / 'out', person_id = that player.
    The incoming name of a v2 row is resolved through the team's players in this game.
    """
    roster = {t: {} for t in team_ids}
    for r in gp[["team_id", "person_id", "player_name_i"]].dropna().itertuples(index=False):
        t, pid, nm = int(r.team_id), int(r.person_id), str(r.player_name_i)
        if t in roster and pid:
            roster[t].setdefault(nm.split()[-1].lower(), set()).add(pid)
    moves = {}
    v3 = {}  # (period, clock, team) -> row indices of v3 in/out rows
    for i, r in gp[gp["kind"] == "sub"].iterrows():
        t = int(r["team_id"]) if pd.notna(r.get("team_id")) and r.get("team_id") else 0
        pid = int(r["person_id"]) if pd.notna(r.get("person_id")) else 0
        if t not in team_ids or not pid:
            continue
        sub = str(r.get("subtype", "")).lower()
        if sub in ("in", "out"):
            moves[i] = (t, 0, pid) if sub == "in" else (t, pid, 0)
            v3.setdefault((r["period"], r["clock_s"], t), []).append(i)
        else:
            m = re.match(r"\s*SUB:\s*(.+?)\s+FOR\s+(.+?)\s*$", str(r.get("description", "")), re.I)
            cands = roster[t].get(m.group(1).split()[-1].lower(), set()) - {pid} if m else set()
            moves[i] = (t, pid, min(cands) if len(cands) == 1 else (sorted(cands)[0] if cands else 0))
    # v3: pair each 'out' row with an 'in' row of the same team at the same clock. The swap
    # happens at the first row of the pair, so the lineup holds five players between the two.
    for (_, _, t), idx in v3.items():
        outs = [i for i in idx if moves[i][1]]
        ins = [i for i in idx if moves[i][2]]
        for io, ii in zip(outs, ins):
            first, second = min(io, ii), max(io, ii)
            moves[first], moves[second] = (t, moves[io][1], moves[ii][2]), (t, 0, 0)
    return moves


def lineups_from_pbp(gp: pd.DataFrame, team_ids: tuple) -> dict:
    """On-court players per team after each action_number: {action_number: {team_id: [5 ids]}}.

    Starters of a period are the first five players of a team who act, or leave in a
    substitution, before being subbed in. Short lineups borrow from the previous period.
    """
    gp = prepare(gp)
    moves = _sub_moves(gp, team_ids)
    result = {}
    last_lineup = {t: [] for t in team_ids}
    for p, g in gp.groupby("period", sort=True):
        starters = {t: [] for t in team_ids}
        entered = {t: set() for t in team_ids}
        for i, r in g.iterrows():
            if i in moves:
                t, out_id, in_id = moves[i]
                if out_id and out_id not in entered[t] and out_id not in starters[t] and len(starters[t]) < 5:
                    starters[t].append(out_id)
                if in_id:
                    entered[t].add(in_id)
                continue
            t = int(r["team_id"]) if pd.notna(r.get("team_id")) and r.get("team_id") else 0
            pid = int(r["person_id"]) if pd.notna(r.get("person_id")) else 0
            if t in team_ids and pid and pid not in entered[t] and pid not in starters[t] and len(starters[t]) < 5:
                starters[t].append(pid)
        for t in team_ids:
            for pid in last_lineup[t]:
                if len(starters[t]) >= 5:
                    break
                if pid not in starters[t] and pid not in entered[t]:
                    starters[t].append(pid)
        on = {t: list(starters[t]) for t in team_ids}
        pending = {t: [] for t in team_ids}  # v3: an 'in' row waiting for its 'out' row
        for i, r in g.iterrows():
            if i in moves:
                t, out_id, in_id = moves[i]
                if out_id and in_id:
                    if out_id in on[t]:
                        on[t].remove(out_id)
                    if in_id not in on[t]:
                        on[t].append(in_id)
                elif out_id:
                    if out_id in on[t]:
                        on[t].remove(out_id)
                    if pending[t]:
                        on[t].append(pending[t].pop(0))
                elif in_id and in_id not in on[t]:
                    (on[t] if len(on[t]) < 5 else pending[t]).append(in_id)
            else:
                for tt in team_ids:
                    while pending[tt]:
                        on[tt].append(pending[tt].pop(0))
                        if len(on[tt]) > 5:
                            on[tt].pop(0)
            for tt in team_ids:
                while len(on[tt]) > 5:
                    on[tt].pop(0)
            result[int(r["action_number"])] = {tt: list(on[tt]) for tt in team_ids}
        last_lineup = on
    return result
