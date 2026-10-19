"""Command line: pbp2track {rebuild, evaluate, build-library, demo}."""
import argparse
import glob
import json
import os
import re
import sys
import tempfile
import time

import pandas as pd

from . import __version__
from .evaluate import heldout, table
from .export import save_rebuild
from .library import Library
from .possessions import names_from_pbp
from .reconstruct import reconstruct_game
from .sportvu import build_library, gid10, read_pbp


def _log(quiet):
    t0 = time.time()

    def log(msg):
        if not quiet:
            print(f"[{time.time() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)
    return log


def teams_of(gp: pd.DataFrame, home=None, away=None):
    """(home team id, away team id) from --home/--away or from game_matchup ('AWY @ HOM' or 'HOM vs. AWY')."""
    if home and away:
        return int(home), int(away)
    if "game_matchup" not in gp or "team_tri_code" not in gp:
        raise SystemExit("this play-by-play has no game_matchup: pass --home and --away team ids")
    tm = (gp.dropna(subset=["team_tri_code", "team_id"]).drop_duplicates("team_tri_code")
          .set_index("team_tri_code")["team_id"].astype(int).to_dict())
    for m in gp["game_matchup"].dropna().astype(str).unique():
        if "@" in m:
            a, h = [s.strip() for s in m.split("@")][:2]
        elif re.search(r"vs\.?", m):
            h, a = [s.strip() for s in re.split(r"vs\.?", m)][:2]
        else:
            continue
        if h in tm and a in tm:
            return tm[h], tm[a]
    raise SystemExit(f"could not read home and away from game_matchup {gp['game_matchup'].dropna().unique()[:2]}: "
                     "pass --home and --away team ids")


def cmd_rebuild(a):
    log = _log(a.quiet)
    gp = read_pbp(a.pbp, [a.game])
    if gp.empty:
        raise SystemExit(f"game {a.game} has no rows in {a.pbp}")
    home, away = teams_of(gp, a.home, a.away)
    log(f"game {gid10(a.game)}: {len(gp)} rows, home {home}, away {away}")
    lib = Library(a.library, exclude_games=[gid10(a.game)] if a.exclude_self else (), max_clips=a.max_clips,
                  seed=a.seed).build(log=log)
    cal = lib.legacy_to_court()
    log(f"calibration: median {cal['median_err_ft']:.2f} ft on {cal['n']} shots")
    rec = reconstruct_game(gp, home, away, lib, names_from_pbp(gp))
    out = a.out or gid10(a.game)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    paths = save_rebuild(rec, out, meta={"game_id": gid10(a.game), "home_id": home, "away_id": away,
                                         "library": os.path.abspath(a.library), "clips": len(lib.clips)})
    n = len(rec["pos"])
    print(json.dumps({"frames": n, "minutes": round(n / 25 / 60, 1), "possessions": len(rec["possessions"]),
                      "events": len(rec["events"]), **paths}))


def cmd_evaluate(a):
    log = _log(a.quiet)
    s = heldout(a.library, held=a.held or None, n=a.games, seed=a.seed, include=a.include or (),
                max_clips=a.max_clips, log=log)
    print(table(s))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(s, f, indent=1, default=str)


def cmd_build(a):
    log = _log(a.quiet)
    files = sorted(glob.glob(os.path.join(a.sportvu, "*.7z")) + glob.glob(os.path.join(a.sportvu, "*.json")))
    files = [f for f in files if os.path.getsize(f) > 10_000 or f.endswith(".json")]
    if a.limit:
        files = files[: a.limit]
    ids = range(a.game_range[0], a.game_range[1] + 1) if a.game_range else None
    report = build_library(files, a.pbp, a.out, workers=a.workers, game_ids=ids, log=log)
    print(json.dumps(report, default=str))


def cmd_demo(a):
    from . import synth
    log = _log(a.quiet)
    out = a.out or tempfile.mkdtemp(prefix="pbp2track_demo_")
    log(f"demo folder: {out}")
    games = synth.league(a.games, seed=a.seed, period_s=a.period_s)
    raw = synth.write_league(games, os.path.join(out, "raw"))
    root = os.path.join(out, "library")
    build_library(raw["logs"], raw["pbp"], root, log=log)
    held = [g["sportvu"]["gameid"] for g in games[-a.held:]]
    s = heldout(root, held=held, log=log)
    print(table(s))
    with open(os.path.join(out, "eval.json"), "w") as f:
        json.dump(s, f, indent=1, default=str)
    gp = read_pbp(raw["pbp"], [held[0]])
    lib = Library(root, exclude_games=held).build(log=log)
    home, away = teams_of(gp)
    rec = reconstruct_game(gp, home, away, lib, names_from_pbp(gp))
    paths = save_rebuild(rec, os.path.join(out, "rebuilt", held[0]), meta={"game_id": held[0], "synthetic": True})
    print(f"rebuilt {held[0]}: {len(rec['pos'])} frames -> {paths['npz']}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="pbp2track", description="Rebuild player and ball tracking from play-by-play.")
    p.add_argument("--version", action="version", version=f"pbp2track {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rebuild", help="rebuild one game from its play-by-play")
    r.add_argument("--library", required=True, help="library directory (build-library output)")
    r.add_argument("--pbp", required=True, help="play-by-play CSV (NBA stats v3 columns)")
    r.add_argument("--game", required=True, help="game id, e.g. 0020500591")
    r.add_argument("--home", type=int, help="home team id (read from game_matchup when omitted)")
    r.add_argument("--away", type=int, help="away team id")
    r.add_argument("--out", help="output prefix (default: the game id)")
    r.add_argument("--max-clips", type=int, help="sample this many library possessions")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--exclude-self", action="store_true", help="keep this game's own possessions out of the library")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(fn=cmd_rebuild)

    e = sub.add_parser("evaluate", help="held-out comparison against real tracking")
    e.add_argument("--library", required=True)
    e.add_argument("--games", type=int, default=40, help="number of held-out games")
    e.add_argument("--seed", type=int, default=7)
    e.add_argument("--include", nargs="*", help="game ids always held out (drawn first)")
    e.add_argument("--held", nargs="*", help="exact held-out game ids (overrides --games)")
    e.add_argument("--max-clips", type=int, default=90_000)
    e.add_argument("--out", help="write the summary JSON here")
    e.add_argument("--quiet", action="store_true")
    e.set_defaults(fn=cmd_evaluate)

    b = sub.add_parser("build-library", help="build a possession library from SportVU logs and play-by-play")
    b.add_argument("--sportvu", required=True, help="folder of SportVU game logs (.7z needs py7zr, or .json)")
    b.add_argument("--pbp", required=True, help="play-by-play CSV")
    b.add_argument("--out", required=True, help="library directory to write")
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--limit", type=int, help="first N logs only (smoke runs)")
    b.add_argument("--game-range", type=int, nargs=2, metavar=("FIRST", "LAST"),
                   help="play-by-play game ids to keep, e.g. 21500001 21501230 for 2015-16")
    b.add_argument("--quiet", action="store_true")
    b.set_defaults(fn=cmd_build)

    d = sub.add_parser("demo", help="synthetic league end to end: build, rebuild, evaluate (no data needed)")
    d.add_argument("--out", help="folder for the demo files (default: a temp folder)")
    d.add_argument("--games", type=int, default=8)
    d.add_argument("--held", type=int, default=2)
    d.add_argument("--period-s", type=float, default=150.0)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--quiet", action="store_true")
    d.set_defaults(fn=cmd_demo)

    a = p.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
