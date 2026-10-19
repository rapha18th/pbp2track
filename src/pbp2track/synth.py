"""A small synthetic basketball world for the tests and the demo.

It plays short games with simple smooth motion and writes them in the two formats
pbp2track reads: a SportVU-style tracking log and an NBA stats v3-style
play-by-play table. Teams, players and games are invented; no real data is used.

Each possession: the offense walks to set spots and drifts around them, each
defender guards one attacker from the hoop side, the ball is dribbled by the
handler, passed to the player who finishes the possession, and then shot (made or
missed, with a defensive rebound) or turned over.
"""
import json
import math
import os

import numpy as np
import pandas as pd

FPS = 25
HOOP = (88.75, 25.0)  # the hoop the offense attacks, in the attack frame
SPOTS3 = [(88.0, 2.5), (88.0, 47.5), (71.5, 8.0), (71.5, 42.0), (64.5, 25.0)]
SPOTS2 = [(84.0, 25.0), (80.0, 18.0), (80.0, 32.0), (75.0, 25.0), (85.5, 21.0), (85.5, 29.0)]
FLOOR = [(70.0, 6.0), (70.0, 44.0), (62.0, 18.0), (62.0, 32.0), (82.0, 12.0), (82.0, 38.0), (87.0, 5.0), (87.0, 45.0)]
OUTCOMES = ["made2", "made3", "miss", "turnover"]
OUTCOME_P = [0.32, 0.16, 0.34, 0.18]
FOLLOW = 0.08  # share of the gap to the desired spot closed per frame
MAX_STEP = 0.75  # ft per frame (18.75 ft/s)

_SYL1 = ["Ka", "Lo", "Mi", "Ne", "Ru", "Sa", "Te", "Vo"]
_SYL2 = ["bara", "dene", "lito", "mura", "nobe", "ravi", "sell", "tano"]
_INITIALS = "ABCDEFGHJKLMNPRST"


def make_teams(n_teams: int = 4, players: int = 8) -> list:
    """Invented teams with unique surnames: [{id, abbr, name, players: [{id, first, last, jersey}]}]."""
    names = [a + b for b in _SYL2 for a in _SYL1]
    teams = []
    for i in range(n_teams):
        tid = 1001 + i
        roster = []
        for j in range(players):
            k = i * players + j
            roster.append({"id": tid * 100 + j + 1, "first": _INITIALS[k % len(_INITIALS)], "last": names[k % len(names)],
                           "jersey": str(j * 3 + 1), "position": "GFC"[j % 3]})
        teams.append({"id": tid, "abbr": f"T{i + 1:02d}", "name": f"Team {i + 1}", "players": roster})
    return teams


def _unit(dx, dy):
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else (0.0, 0.0)


def _lerp(a, b, s):
    return (a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s)


def _flip(v, right):
    """Court frame <-> attack frame (the mirror is its own inverse)."""
    return (v[0], v[1]) if right else (94.0 - v[0], 50.0 - v[1])


def _clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m = int(seconds // 60)
    return f"PT{m:02d}M{seconds - 60 * m:05.2f}S"


def play_game(game_id: str, home: dict, away: dict, periods: int = 4, period_s: float = 150.0, seed: int = 0,
              sub_style: str = "v2", sub_p: float = 0.08) -> dict:
    """Simulate one game.

    Returns {"sportvu": log dict, "pbp": DataFrame, "truth": [possession dicts], "home": team, "away": team}.
    sub_style: "v2" writes one 'SUB: X FOR Y' row per substitution, "v3" writes an out row and an in row.
    Attack direction: the home team attacks the +x hoop in the first half of the periods.
    """
    rng = np.random.default_rng(seed)
    H, A = int(home["id"]), int(away["id"])
    info = {p["id"]: p for p in home["players"] + away["players"]}
    abbr = {H: home["abbr"], A: away["abbr"]}
    on = {H: [p["id"] for p in home["players"][:5]], A: [p["id"] for p in away["players"][:5]]}
    bench = {H: [p["id"] for p in home["players"][5:]], A: [p["id"] for p in away["players"][5:]]}
    pos = {}
    for j, pid in enumerate(on[H]):
        pos[pid] = (40.0, 5.0 + 10.0 * j)
    for j, pid in enumerate(on[A]):
        pos[pid] = (54.0, 5.0 + 10.0 * j)
    phase = {pid: float(rng.uniform(0, 2 * math.pi)) for pid in info}

    moments, events_json = [], []
    rows = []
    truth = []
    score = {H: 0, A: 0}
    tally = {pid: {"pts": 0, "ast": 0, "reb": 0, "tov": 0} for pid in info}
    counters = {H: 0, A: 0}
    unix0 = 1_450_000_000_000
    state = {"an": 0, "frame": 0}
    matchup = f"{away['abbr']} @ {home['abbr']}"

    def row(period, seconds, team, pid, action_type, subtype="", description="", **kw):
        state["an"] += 1
        r = {"game_id": int(game_id), "game_date": "2016-01-01", "game_matchup": matchup,
             "action_number": state["an"], "clock": _clock(seconds), "period": period,
             "team_id": team if team else np.nan, "team_tri_code": abbr.get(team, np.nan) if team else np.nan,
             "person_id": pid if pid else 0,
             "player_name_i": f"{info[pid]['first']}. {info[pid]['last']}" if pid else np.nan,
             "action_type": action_type, "subtype": subtype if subtype else np.nan, "description": description,
             "shot_result": np.nan, "is_field_goal": 0, "shot_value": np.nan, "shot_distance": np.nan,
             "xlegacy": np.nan, "ylegacy": np.nan, "score_home": score[H], "score_away": score[A]}
        r.update(kw)
        rows.append(r)

    def step(pid, des):
        x, y = pos[pid]
        mx, my = FOLLOW * (des[0] - x), FOLLOW * (des[1] - y)
        n = math.hypot(mx, my)
        if n > MAX_STEP:
            mx, my = mx * MAX_STEP / n, my * MAX_STEP / n
        pos[pid] = (x + mx, y + my)

    def emit(period, f_in_period, sc, ball):
        ents = [[-1, -1, ball[0], ball[1], ball[2]]]
        for t in (H, A):
            for pid in on[t]:
                x, y = pos[pid]
                ents.append([t, pid, x, y, 0.0])
        moments.append([period, unix0 + period * 10_000_000 + f_in_period * 40, round(period_s - f_in_period / FPS, 2),
                        round(min(24.0, max(0.0, sc)), 2), None, ents])
        state["frame"] += 1

    def team_json(team):
        return {"name": team["name"], "teamid": team["id"], "abbreviation": team["abbr"],
                "players": [{"lastname": p["last"], "firstname": p["first"], "playerid": p["id"],
                             "jersey": p["jersey"], "position": p["position"]} for p in team["players"]]}

    def flush_event():
        if moments:
            events_json.append({"eventId": str(len(events_json) + 1), "visitor": team_json(away),
                                "home": team_json(home), "moments": list(moments)})
            moments.clear()

    ball = (47.0, 25.0, 5.0)
    for period in range(1, periods + 1):
        home_right = period <= max(1, periods // 2)
        total = int(round(period_s * FPS))
        f = 0
        row(period, period_s, 0, 0, "period", "start", f"Start of period {period}")
        offense = H if period % 2 == 1 else A
        while True:
            outcome = str(rng.choice(OUTCOMES, p=OUTCOME_P))
            D = int(rng.uniform(5.0, 11.0) * FPS)
            need = D + (25 if outcome == "miss" else 0)
            if f + need + 2 * FPS > total:
                break
            defense = A if offense == H else H
            right = (offense == H) == home_right
            off_ids, def_ids = list(on[offense]), list(on[defense])
            a_slot = counters[offense] % 5
            counters[offense] += 1
            actor = off_ids[a_slot]
            handler = off_ids[0]
            value = 3 if outcome == "made3" else 2 if outcome == "made2" else int(rng.choice([2, 3]))
            if outcome == "turnover":
                spot = FLOOR[int(rng.integers(len(FLOOR)))]
            else:
                pool = SPOTS3 if value == 3 else SPOTS2
                spot = pool[int(rng.integers(len(pool)))]
            others = [FLOOR[int(k)] for k in rng.choice(len(FLOOR), 5, replace=False)]
            spots = [spot if j == a_slot else others[j] for j in range(5)]
            r_t = D - 21 if outcome != "turnover" else None  # release
            a_t = D - 1  # the ball reaches the rim, or the turnover
            pass_t = (D - 51 if outcome != "turnover" else D - 30) if actor != handler else None
            ball_start = _flip(ball, right)
            rebounder, reb_spot, rel_n, shot_spot = None, None, None, None
            for t in range(need):
                # offense: walk to the spot and drift around it; the shooter settles 40 frames before release
                ang = 2 * math.pi * t / (3 * FPS)
                for j, pid in enumerate(off_ids):
                    sx, sy = spots[j]
                    if not (j == a_slot and r_t is not None and t >= r_t - 40):
                        sx += 2.0 * math.cos(ang + phase[pid])
                        sy += 2.0 * math.sin(ang + phase[pid])
                    step(pid, _flip((sx, sy), right))
                off_n = [_flip(pos[p], right) for p in off_ids]
                if outcome == "miss" and t == r_t + 1:
                    dn = [_flip(pos[p], right) for p in def_ids]
                    k = min(range(5), key=lambda i: math.hypot(dn[i][0] - HOOP[0], dn[i][1] - HOOP[1]))
                    rebounder = def_ids[k]
                    ux, uy = _unit(dn[k][0] - HOOP[0], dn[k][1] - HOOP[1])
                    reb_spot = (min(93.0, max(1.0, HOOP[0] + 8.0 * ux)), min(49.0, max(1.0, HOOP[1] + 8.0 * uy)))
                # defense: guard one attacker from the hoop side; the rebounder goes to the ball
                for j, pid in enumerate(def_ids):
                    if pid == rebounder:
                        des = reb_spot
                    else:
                        ox, oy = off_n[j]
                        ux, uy = _unit(HOOP[0] - ox, HOOP[1] - oy)
                        des = (ox + 3.5 * ux, oy + 3.5 * uy)
                    step(pid, _flip(des, right))
                # ball, in the attack frame
                holder = handler if pass_t is None or t < pass_t else actor
                hp = _flip(pos[holder], right)
                ux, uy = _unit(HOOP[0] - hp[0], HOOP[1] - hp[1])
                dribble = (hp[0] + 0.7 * ux, hp[1] + 0.7 * uy, 1.0 + 2.5 * abs(math.sin(math.pi * t / 10)))
                if t < 12:
                    bxy = _lerp(ball_start, dribble, t / 12)
                    bn = (bxy[0], bxy[1], 4.0)
                elif pass_t is not None and pass_t <= t < pass_t + 12:
                    bxy = _lerp(_flip(pos[handler], right), hp, (t - pass_t) / 12)
                    bn = (bxy[0], bxy[1], 5.0)
                elif r_t is not None and t >= r_t:
                    if t == r_t:
                        rel_n = hp
                    if t <= a_t:
                        s = (t - r_t) / (a_t - r_t)
                        bxy = _lerp(rel_n, HOOP, s)
                        bn = (bxy[0], bxy[1], 8.0 + 2.0 * s + 6.0 * math.sin(math.pi * s))
                    else:
                        s = (t - a_t) / 25
                        bxy = _lerp(HOOP, reb_spot, s)
                        bn = (bxy[0], bxy[1], 10.0 + 3.0 * math.sin(math.pi * s) - 4.0 * s)
                elif r_t is not None and t >= r_t - 10:
                    bn = (hp[0], hp[1], 5.0 + 3.0 * (t - (r_t - 10)) / 10)
                else:
                    bn = dribble
                bxy = _flip(bn, right)
                ball = (bxy[0], bxy[1], bn[2])
                emit(period, f, 24.0 - t / FPS, ball)
                f += 1
                if t == a_t and outcome != "turnover":
                    sec = period_s - (f - 1) / FPS
                    dist = math.hypot(rel_n[0] - HOOP[0], rel_n[1] - HOOP[1])
                    xl, yl = round((rel_n[1] - 25.0) * 10), round((88.75 - rel_n[0]) * 10)
                    word = "Jump Shot" if dist > 10 else "Layup Shot"
                    tri = "3PT " if value == 3 else ""
                    if outcome.startswith("made"):
                        score[offense] += value
                        tally[actor]["pts"] += value
                        desc = f"{info[actor]['last']} {round(dist)}' {tri}{word} ({tally[actor]['pts']} PTS)"
                        if actor != handler:
                            tally[handler]["ast"] += 1
                            desc += f" ({info[handler]['last']} {tally[handler]['ast']} AST)"
                        row(period, sec, offense, actor, "Made Shot", word, desc, shot_result="Made", is_field_goal=1,
                            shot_value=value, shot_distance=round(dist), xlegacy=xl, ylegacy=yl)
                    else:
                        row(period, sec, offense, actor, "Missed Shot", word,
                            f"MISS {info[actor]['last']} {round(dist)}' {tri}{word}", shot_result="Missed",
                            is_field_goal=1, shot_value=value, shot_distance=round(dist), xlegacy=xl, ylegacy=yl)
                    shot_spot = _flip(rel_n, right)
            sec_end = period_s - (f - 1) / FPS
            if outcome == "miss":
                tally[rebounder]["reb"] += 1
                row(period, sec_end, defense, rebounder, "Rebound", "Unknown",
                    f"{info[rebounder]['last']} REBOUND (Off:0 Def:{tally[rebounder]['reb']})")
            elif outcome == "turnover":
                tally[actor]["tov"] += 1
                row(period, sec_end, offense, actor, "Turnover", "Bad Pass",
                    f"{info[actor]['last']} Bad Pass Turnover (P1.T{tally[actor]['tov']})")
            truth.append({"period": period, "offense": offense, "outcome": "miss_dreb" if outcome == "miss" else outcome,
                          "start_frame": state["frame"] - need, "end_frame": state["frame"] - 1, "actor": actor,
                          "handler": handler, "rebounder": rebounder or 0, "attack_right": bool(right),
                          "release_frame": None if r_t is None else state["frame"] - need + r_t,
                          "shot_xy": None if shot_spot is None else [float(shot_spot[0]), float(shot_spot[1])]})
            flush_event()
            # substitutions at the dead ball
            for team in (H, A):
                if rng.random() < sub_p and bench[team]:
                    j = int(rng.integers(5))
                    out_id = on[team][j]
                    in_id = bench[team].pop(int(rng.integers(len(bench[team]))))
                    bench[team].append(out_id)
                    if sub_style == "v3":
                        row(period, sec_end, team, out_id, "Substitution", "out", f"SUB out: {info[out_id]['last']}")
                        row(period, sec_end, team, in_id, "Substitution", "in", f"SUB in: {info[in_id]['last']}")
                    else:
                        row(period, sec_end, team, out_id, "Substitution", "",
                            f"SUB: {info[in_id]['last']} FOR {info[out_id]['last']}")
                    on[team][j] = in_id
                    pos[in_id] = pos.pop(out_id)
            offense = A if offense == H else H
        # run out the clock: players drift, the ball rests with the next handler
        while f < total:
            hx, hy = pos[on[offense][0]]
            emit(period, f, 24.0, (hx, hy, 3.0))
            f += 1
        flush_event()
        row(period, 0.0, 0, 0, "period", "end", f"End of period {period}")

    pbp = pd.DataFrame(rows)
    sportvu = {"gameid": game_id, "gamedate": "2016-01-01", "events": events_json}
    return {"sportvu": sportvu, "pbp": pbp, "truth": truth, "home": home, "away": away}


def league(n_games: int = 8, seed: int = 0, periods: int = 4, period_s: float = 150.0) -> list:
    """n_games games among four invented teams; game ids 0099900001, 0099900002, ..."""
    teams = make_teams(4)
    games = []
    for i in range(n_games):
        home, away = teams[i % 4], teams[(i + 1 + i // 4) % 4]
        if home["id"] == away["id"]:
            away = teams[(i + 2) % 4]
        games.append(play_game(f"00999{i + 1:05d}", home, away, periods=periods, period_s=period_s,
                               seed=seed * 1000 + i, sub_style="v3" if i % 2 else "v2"))
    return games


def write_league(games: list, out_dir: str) -> dict:
    """Write each game's tracking log as JSON and all play-by-play as one CSV."""
    os.makedirs(out_dir, exist_ok=True)
    logs = []
    for g in games:
        path = os.path.join(out_dir, f"{g['sportvu']['gameid']}.json")
        with open(path, "w") as f:
            json.dump(g["sportvu"], f)
        logs.append(path)
    pbp_path = os.path.join(out_dir, "play_by_play.csv")
    pd.concat([g["pbp"] for g in games], ignore_index=True).to_csv(pbp_path, index=False)
    return {"logs": logs, "pbp": pbp_path}
