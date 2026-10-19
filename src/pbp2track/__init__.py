"""pbp2track: rebuild basketball player and ball tracking from play-by-play.

Motion matching over a library of real tracked possessions: each logged possession is
matched to a real one with the same outcome, shot location, shot type and duration,
retargeted onto the actual players, time-warped to the logged clock and bent so the
logged facts hold (the shooter at the shot spot at release, the ball through the rim
on a make, the rebounder at the ball).
"""
__version__ = "0.1.0"

from .possessions import clock_s, lineups_from_pbp, names_from_pbp, segment_pbp
from .library import Library
from .reconstruct import reconstruct_game
from .evaluate import heldout
from .export import load_rebuild, save_rebuild
from .sportvu import build_library

__all__ = ["segment_pbp", "lineups_from_pbp", "names_from_pbp", "clock_s", "Library", "reconstruct_game",
           "heldout", "build_library", "save_rebuild", "load_rebuild", "__version__"]
