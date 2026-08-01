"""Render the scenario figure for the paper, straight from scenario_v2.apply_layout.

Generating the figure from the same code that runs the experiments means it cannot
drift from the layout actually measured.

    python make_layout_figure.py --out ../paper/fig_layout.pdf

Outputs vector PDF (sharp at any zoom, which is what IEEE wants). Sized for a
single IEEE column (3.5 in wide).
"""

from __future__ import annotations

import argparse
import os
import sys

STATIC_DIR = os.path.abspath(os.path.dirname(__file__))
if STATIC_DIR not in sys.path:
    sys.path.insert(0, STATIC_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

import GA.GA as GA  # noqa: E402
import scenario_v2 as sc  # noqa: E402

# Greyscale-safe palette: the figure must stay readable if printed in black
# and white, so category is carried by hatch/edge as well as by fill.
C_DOOR = "#B5D4F4"
C_STATION = "#F5C4B3"
C_BAY = "#CECBF6"
C_QUEUE = "#EDEDED"


def draw(ax, num_amrs: int, show_queues: bool, show_flow: bool) -> None:
    sc.apply_layout(num_amrs=num_amrs, depot_x=4)

    w, h = GA.GRID_MAX_X, GA.GRID_MAX_Y
    docks = dict(GA.INBOUND_DOCK_LOCATIONS)
    stations = dict(GA.STATIONS)
    bays = list(GA.AMR_STARTS.values())

    # grid
    for x in range(w + 2):
        ax.plot([x - 0.5, x - 0.5], [0.5, h + 0.5], lw=0.3, color="#D8D8D8", zorder=0)
    for y in range(1, h + 2):
        ax.plot([-0.5, w + 0.5], [y - 0.5, y - 0.5], lw=0.3, color="#D8D8D8", zorder=0)

    # waiting cells
    if show_queues:
        seen = set()
        for d in list(docks.values()) + list(stations.values()):
            for c in GA.dock_waiting_slots(d):
                if c in seen or c in docks.values() or c in stations.values():
                    continue
                seen.add(c)
                ax.add_patch(Rectangle((c[0] - 0.5, c[1] - 0.5), 1, 1,
                                       facecolor=C_QUEUE, edgecolor="none", zorder=1))

    # service points
    for name, (x, y) in docks.items():
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=C_DOOR,
                               edgecolor="#185FA5", lw=0.8, zorder=3))
    for name, (x, y) in stations.items():
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=C_STATION,
                               edgecolor="#993C1D", lw=0.8, zorder=3))

    # charging bays
    for (x, y) in bays:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=C_BAY,
                               edgecolor="#534AB7", lw=0.8, zorder=3))

    # one illustrative pickup -> delivery flow
    if show_flow:
        src = docks["dock3"]
        dst = stations["station3"]
        ax.add_patch(FancyArrowPatch((src[0] + 0.6, src[1]), (dst[0] - 0.6, dst[1]),
                                     arrowstyle="-|>", mutation_scale=9,
                                     lw=1.0, color="#444441", linestyle=(0, (4, 2)),
                                     zorder=4))
        ax.text((src[0] + dst[0]) / 2, src[1] + 0.75, r"$\rho_j \rightarrow \delta_j$",
                ha="center", va="bottom", fontsize=7, color="#444441", zorder=5)

    ax.set_xlim(-1.2, w + 1.2)
    ax.set_ylim(0.2, h + 1.6)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(-0.9, h / 2, "inbound doors", rotation=90, va="center", ha="center",
            fontsize=7.5, color="#185FA5")
    ax.text(w + 0.9, h / 2, "outbound stations", rotation=-90, va="center",
            ha="center", fontsize=7.5, color="#993C1D")

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=C_DOOR, edgecolor="#185FA5", lw=0.8),
        Rectangle((0, 0), 1, 1, facecolor=C_STATION, edgecolor="#993C1D", lw=0.8),
        Rectangle((0, 0), 1, 1, facecolor=C_BAY, edgecolor="#534AB7", lw=0.8),
        Rectangle((0, 0), 1, 1, facecolor=C_QUEUE, edgecolor="#BBBBBB", lw=0.5),
    ]
    labels = ["inbound door", "outbound station", "charging bay", "waiting cells"]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.10),
              ncol=4, frameon=False, fontsize=6.6, handlelength=1.0,
              handleheight=1.0, columnspacing=1.0, handletextpad=0.4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../paper/fig_layout.pdf")
    ap.add_argument("--amrs", type=int, default=sc.NUM_AMRS)
    ap.add_argument("--width", type=float, default=3.5, help="inches; 3.5 = 1 IEEE column")
    ap.add_argument("--no_queues", action="store_true")
    ap.add_argument("--no_flow", action="store_true")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(args.width, args.width * 0.92))
    draw(ax, args.amrs, not args.no_queues, not args.no_flow)
    fig.tight_layout(pad=0.15)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {args.out} and {png}")
    print(f"  {GA.GRID_MAX_X + 1}x{GA.GRID_MAX_Y} grid, {len(GA.INBOUND_DOCK_LOCATIONS)} doors/side, "
          f"{len(GA.AMR_KEYS)} AMRs, eta = {sc.contention_ratio(len(GA.AMR_KEYS)):.1f}")


if __name__ == "__main__":
    main()
