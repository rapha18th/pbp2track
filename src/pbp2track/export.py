"""Save and load a rebuilt game.

<prefix>.npz holds the tracks:
  pos (n,23) float32 feet: ball x, y, z, then x, y for ten players (home five, away five)
  gc (n,) game clock seconds left, sc (n,) shot clock, q (n,) period
<prefix>.json holds what the tracks mean:
  fps, frames, lineups [{s, e, pids}], events [{f, period, clock, kind, team_id, person_id, desc, ...}],
  pinned (frames where a logged fact holds a player in place, per slot), possessions, meta.
Court frame: x 0..94 ft along the floor, y 0..50 ft, hoops at (5.25, 25) and (88.75, 25).
"""
import json
import os

import numpy as np
import pandas as pd

from .library import FPS


def _plain(v):
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, np.ndarray):
        return v.tolist()
    if v is pd.NA or v is pd.NaT:
        return None
    return str(v)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (str, int, bool)) or o is None or isinstance(o, float):
        return o
    return _plain(o)


def save_rebuild(rec: dict, prefix: str, meta: dict = None) -> dict:
    """Write <prefix>.npz and <prefix>.json; returns both paths."""
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    np.savez_compressed(f"{prefix}.npz", pos=rec["pos"].astype(np.float32), gc=np.asarray(rec["gc"], np.float32),
                        sc=np.asarray(rec["sc"], np.float32), q=np.asarray(rec["q"], np.uint8))
    doc = {"fps": FPS, "frames": int(len(rec["pos"])), "lineups": rec["lineups"], "events": rec["events"],
           "pinned": rec["pinned"], "possessions": rec["possessions"], "meta": meta or {}}
    with open(f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(_clean(doc), f, separators=(",", ":"))
    return {"npz": f"{prefix}.npz", "json": f"{prefix}.json"}


def load_rebuild(prefix: str) -> dict:
    """Read a game written by save_rebuild back into the reconstruct_game layout."""
    with np.load(f"{prefix}.npz") as z:
        out = {k: z[k] for k in ("pos", "gc", "sc", "q")}
    with open(f"{prefix}.json", encoding="utf-8") as f:
        doc = json.load(f)
    out.update({k: doc[k] for k in ("lineups", "events", "pinned", "possessions", "meta")})
    return out
