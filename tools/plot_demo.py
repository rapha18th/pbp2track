"""Draw one possession of a `pbp2track demo` run: its tracking beside its rebuild.

usage: pbp2track demo --out DEMO && python tools/plot_demo.py DEMO docs/demo.png
Needs matplotlib.
"""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc, Circle, Rectangle  # noqa: E402

from pbp2track.export import load_rebuild  # noqa: E402
from pbp2track.library import FrameStore  # noqa: E402

INK, HOME, AWAY, BALL = "#2b2b2b", "#2f5d9e", "#c0443a", "#e08a1e"


def court(ax):
    ax.add_patch(Rectangle((0, 0), 94, 50, fill=False, lw=1.2, ec=INK))
    ax.plot([47, 47], [0, 50], color=INK, lw=0.8)
    ax.add_patch(Circle((47, 25), 6, fill=False, lw=0.8, ec=INK))
    for hx, s in ((5.25, 1), (88.75, -1)):
        ax.add_patch(Circle((hx, 25), 0.75, fill=False, lw=1.5, ec=BALL))
        ax.add_patch(Arc((hx, 25), 47.5, 47.5, theta1=-68 if s == 1 else 112, theta2=68 if s == 1 else 248, lw=0.8, ec=INK))
        ax.add_patch(Rectangle((0 if s == 1 else 75, 17), 19, 16, fill=False, lw=0.8, ec=INK))
    ax.set_xlim(-2, 96)
    ax.set_ylim(-2, 52)
    ax.set_aspect("equal")
    ax.axis("off")


def tracks(ax, p10, ball, home_mask, title, release=None):
    court(ax)
    for k in range(10):
        c = HOME if home_mask[k] else AWAY
        ax.plot(p10[:, k, 0], p10[:, k, 1], color=c, lw=1.3, alpha=0.85)
        ax.scatter(p10[-1, k, 0], p10[-1, k, 1], s=36, color=c, zorder=3)
    ax.plot(ball[:, 0], ball[:, 1], color=BALL, lw=1.6, ls="--")
    if release is not None:
        ax.scatter(*release, s=140, facecolors="none", edgecolors=INK, lw=1.4, zorder=4)
    ax.set_title(title, fontsize=11, color=INK)


def main(demo, out):
    with open(os.path.join(demo, "eval.json"), encoding="utf-8") as f:
        gid = json.load(f)["held_out"][0]
    rec = load_rebuild(os.path.join(demo, "rebuilt", gid))
    g = FrameStore(os.path.join(demo, "library", "frames")).get(gid)
    pls = [p for p in rec["possessions"] if p["outcome"] == "made3" and p["release_f"] is not None and p["T"] > 150]
    pl = pls[len(pls) // 2]
    s, e = pl["f0"], pl["f0"] + pl["T"]
    lu = next(l for l in rec["lineups"] if l["s"] <= s <= l["e"])
    home_ids = set(lu["pids"][:5])

    # the same stretch of game clock in the tracking
    q, gc = g["q"], g["gc"]
    idx = np.where((q == pl["period"]) & (gc <= rec["gc"][s] + 1e-3) & (gc >= pl["end_c"] - 1e-3))[0]
    real_home = np.array([pid in home_ids for pid in g["pid"][idx[0]]])
    k_real = int(np.where(g["pid"][idx[0]] == pl["shooter"])[0][0])
    k_rec = lu["pids"].index(pl["shooter"])
    rf = pl["release_f"]
    real_release = idx[0] + int(round((rf - s) / max(1, e - 1 - s) * (len(idx) - 1)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), dpi=110, facecolor="white")
    tracks(axes[0], g["pxy"][idx], g["ball"][idx], real_home, "Tracking", g["pxy"][real_release, k_real])
    tracks(axes[1], rec["pos"][s:e, 3:].reshape(-1, 10, 2), rec["pos"][s:e, :3], [True] * 5 + [False] * 5,
           "Rebuilt from the play-by-play", rec["pos"][rf, 3 + 2 * k_rec: 5 + 2 * k_rec])
    fig.text(0.5, 0.02, "A synthetic made three from pbp2track demo. Rings mark the shooter at release.",
             ha="center", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
