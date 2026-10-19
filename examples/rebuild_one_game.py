"""Rebuild one game from its play-by-play with the Python API.

usage:
  python examples/rebuild_one_game.py LIBRARY_DIR PLAY_BY_PLAY.csv GAME_ID [OUT_PREFIX]
  python examples/rebuild_one_game.py            (a synthetic league in a temp folder)
"""
import os
import sys
import tempfile

import numpy as np

from pbp2track import Library, build_library, names_from_pbp, reconstruct_game, save_rebuild
from pbp2track.cli import teams_of
from pbp2track.sportvu import gid10, read_pbp


def synthetic():
    from pbp2track import synth
    out = tempfile.mkdtemp(prefix="pbp2track_example_")
    games = synth.league(4, seed=5)
    raw = synth.write_league(games, os.path.join(out, "raw"))
    build_library(raw["logs"], raw["pbp"], os.path.join(out, "library"))
    held = games[-1]["sportvu"]["gameid"]
    return os.path.join(out, "library"), raw["pbp"], held, os.path.join(out, "rebuilt", held), [held]


def main(argv):
    if len(argv) >= 3:
        root, pbp_path, game = argv[:3]
        out = argv[3] if len(argv) > 3 else gid10(game)
        exclude = []
    else:
        root, pbp_path, game, out, exclude = synthetic()

    game_rows = read_pbp(pbp_path, [game])
    home, away = teams_of(game_rows)

    lib = Library(root, exclude_games=exclude).build()
    print("calibration", lib.legacy_to_court())

    rec = reconstruct_game(game_rows, home, away, lib, names_from_pbp(game_rows))
    paths = save_rebuild(rec, out, meta={"game_id": gid10(game)})

    pos = rec["pos"]
    speed = np.linalg.norm(np.diff(pos[:, 3:].reshape(-1, 10, 2), axis=0), axis=2) * 25
    print(f"{len(pos)} frames ({len(pos) / 25 / 60:.1f} min of live play), {len(rec['possessions'])} possessions")
    print(f"mean player speed {np.nanmean(speed[speed < 35]):.2f} ft/s")
    print("wrote", paths["npz"], "and", paths["json"])


if __name__ == "__main__":
    main(sys.argv[1:])
