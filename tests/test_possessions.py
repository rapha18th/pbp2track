import numpy as np
import pandas as pd
import pytest

from pbp2track.possessions import (clock_s, ft_last, kind, lineups_from_pbp, made, names_from_pbp, segment_pbp,
                                   shot_class)

T1, T2 = 1, 2
ROSTER = {11: "A. Aone", 12: "B. Atwo", 13: "C. Athree", 14: "D. Afour", 15: "E. Afive", 16: "F. Asix",
          21: "G. Bone", 22: "H. Btwo", 23: "J. Bthree", 24: "K. Bfour", 25: "L. Bfive", 26: "M. Bsix"}


class Log:
    """Builds NBA stats v3-style rows for hand-written test games."""

    def __init__(self):
        self.rows = []

    def add(self, clock, team, pid, action_type, subtype="", description="", period=1, **kw):
        r = {"action_number": len(self.rows) + 1, "clock": clock, "period": period,
             "team_id": team if team else np.nan, "person_id": pid,
             "player_name_i": ROSTER.get(pid, np.nan), "action_type": action_type,
             "subtype": subtype or np.nan, "description": description, "shot_result": np.nan,
             "shot_value": np.nan, "xlegacy": np.nan, "ylegacy": np.nan}
        r.update(kw)
        self.rows.append(r)
        return self

    def df(self):
        return pd.DataFrame(self.rows)


def starters_log():
    """Every player acts once at the start so each team's five is known."""
    log = Log().add("PT12M00.00S", 0, 0, "period", "start", "Start of 1st Period")
    for pid in (11, 12, 13, 14, 15):
        log.add("PT12M00.00S", T1, pid, "Foul", "Personal", f"{ROSTER[pid].split()[-1]} P.FOUL")
    for pid in (21, 22, 23, 24, 25):
        log.add("PT12M00.00S", T2, pid, "Foul", "Personal", f"{ROSTER[pid].split()[-1]} P.FOUL")
    return log


def test_clock_s():
    assert clock_s("PT11M41.00S") == 701.0
    assert clock_s("PT00M04.50S") == 4.5
    assert np.isnan(clock_s("11:41"))


def test_kind():
    assert kind("Made Shot") == "shot"
    assert kind("Missed Shot") == "shot"
    assert kind("Free Throw") == "ft"
    assert kind("Rebound") == "reb"
    assert kind("Turnover") == "tov"
    assert kind("period") == "period"
    assert kind("Substitution") == "sub"
    assert kind("Jump Ball") == "jump"
    assert kind("Foul") == "foul"
    assert kind("Timeout") == "other"


def test_made_and_ft_last():
    assert made({"shot_result": "Made"})
    assert not made({"shot_result": "Missed"})
    assert not made({"shot_result": np.nan, "description": "MISS Curryish 26' 3PT Jump Shot"})
    assert made({"shot_result": np.nan, "description": "Curryish 26' 3PT Jump Shot (3 PTS)"})
    assert not ft_last({"subtype": "Free Throw 1 of 2", "description": ""})
    assert ft_last({"subtype": "Free Throw 2 of 2", "description": ""})
    assert ft_last({"subtype": "Technical", "description": ""})


def test_shot_class():
    assert shot_class("Driving Layup Shot") == "rim"
    assert shot_class("Dunk Shot") == "rim"
    assert shot_class("Hook Shot") == "hook"
    assert shot_class("Step Back Jump shot") == "jump"
    assert shot_class("Unknown") == "jump"


def test_names_from_pbp():
    df = starters_log().df()
    names = names_from_pbp(df)
    assert names[11] == "Aone" and names[25] == "Bfive"
    assert 0 not in names


def test_segment_small_log():
    log = starters_log()
    log.add("PT11M50.00S", T1, 11, "Made Shot", "Jump Shot", "Aone 18' Jump Shot (2 PTS) (Atwo 1 AST)",
            shot_result="Made", shot_value=2, xlegacy=-60, ylegacy=170)
    log.add("PT11M35.00S", T2, 21, "Missed Shot", "Jump Shot", "MISS Bone 25' 3PT Jump Shot",
            shot_result="Missed", shot_value=3, xlegacy=200, ylegacy=150)
    log.add("PT11M33.00S", T1, 13, "Rebound", "Unknown", "Athree REBOUND (Off:0 Def:1)")
    log.add("PT11M20.00S", T1, 14, "Turnover", "Bad Pass", "Afour Bad Pass Turnover (P1.T1)")
    log.add("PT11M10.00S", T2, 22, "Free Throw", "Free Throw 1 of 2", "Btwo Free Throw 1 of 2 (1 PTS)", shot_result="Made")
    log.add("PT11M10.00S", T2, 22, "Free Throw", "Free Throw 2 of 2", "Btwo Free Throw 2 of 2 (2 PTS)", shot_result="Made")
    log.add("PT11M00.00S", 0, 0, "period", "end", "End of 1st Period")
    poss = segment_pbp(log.df(), (T1, T2))
    assert [p["outcome"] for p in poss] == ["made2", "miss_dreb", "turnover", "ft", "period_end"]
    assert [p["offense"] for p in poss] == [T1, T2, T1, T2, T1]
    first, miss, tov = poss[0], poss[1], poss[2]
    assert first["shooter"] == 11 and first["assist_name"] == "Atwo" and first["shot_value"] == 2
    assert (first["shot_x"], first["shot_y"]) == (-60.0, 170.0)
    assert first["dur"] == pytest.approx(10.0)
    assert miss["rebounder"] == 13 and miss["prev"] == "made2" and miss["shot_class"] == "jump"
    assert tov["tov_player"] == 14 and tov["prev"] == "miss_dreb"


@pytest.mark.parametrize("style", ["v2", "v3"])
def test_lineups_follow_substitutions(style):
    log = starters_log()
    if style == "v2":
        log.add("PT11M00.00S", T1, 12, "Substitution", "", "SUB: Asix FOR Atwo")
        log.add("PT10M50.00S", T1, 16, "Made Shot", "Layup Shot", "Asix 2' Layup Shot (2 PTS)", shot_result="Made",
                shot_value=2)
    else:
        log.add("PT11M00.00S", T1, 12, "Substitution", "out", "SUB out: Atwo")
        log.add("PT11M00.00S", T1, 16, "Substitution", "in", "SUB in: Asix")
    df = log.df()
    lus = lineups_from_pbp(df, (T1, T2))
    before = lus[int(df["action_number"].iloc[10])]
    after = lus[int(df["action_number"].iloc[-1])]
    assert sorted(before[T1]) == [11, 12, 13, 14, 15]
    assert sorted(after[T1]) == [11, 13, 14, 15, 16]
    assert sorted(after[T2]) == [21, 22, 23, 24, 25]


def test_v3_rows_pair_out_and_in():
    log = starters_log()
    log.add("PT11M00.00S", T1, 12, "Substitution", "out", "SUB out: Atwo")
    log.add("PT11M00.00S", T1, 13, "Substitution", "out", "SUB out: Athree")
    log.add("PT11M00.00S", T1, 16, "Substitution", "in", "SUB in: Asix")
    log.add("PT11M00.00S", T1, 17, "Substitution", "in", "SUB in: Aseven")
    df = log.df()
    lus = lineups_from_pbp(df, (T1, T2))
    ans = df["action_number"].tolist()[-4:]
    assert all(len(lus[a][T1]) == 5 for a in ans)
    assert sorted(lus[ans[0]][T1]) == [11, 13, 14, 15, 16]
    assert sorted(lus[ans[-1]][T1]) == [11, 14, 15, 16, 17]


def test_segment_matches_synthetic_truth(held_game):
    home, away = held_game["home"]["id"], held_game["away"]["id"]
    poss = [p for p in segment_pbp(held_game["pbp"], (home, away)) if p["outcome"] != "period_end"]
    truth = held_game["truth"]
    assert [p["outcome"] for p in poss] == [t["outcome"] for t in truth]
    assert [p["offense"] for p in poss] == [t["offense"] for t in truth]
    shooters = [p["shooter"] for p in poss if p["outcome"] != "turnover"]
    assert shooters == [t["actor"] for t in truth if t["outcome"] != "turnover"]


def test_synthetic_lineups_have_five(games):
    # v3 rows name both players of a substitution, so every lineup is complete
    g = games[3]
    home, away = g["home"]["id"], g["away"]["id"]
    assert (g["pbp"]["subtype"] == "in").any()
    lus = lineups_from_pbp(g["pbp"], (home, away))
    assert all(len(v[t]) == 5 for v in lus.values() for t in (home, away))
