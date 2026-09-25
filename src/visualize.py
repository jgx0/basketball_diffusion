"""Matplotlib rendering of half-court possessions to MP4 (requires ffmpeg).

Frame convention (must match src/normalize.py): the attacked rim sits at the
ORIGIN, the offense occupies x' <= 0 and advances toward the rim as
x' -> 0-, y' is the signed lateral offset from the rim. All court geometry is
drawn in THIS frame.

Usage:
    from src.visualize import render_gif
    render_gif(traj, "out.mp4")   # traj: (T, 11, 4) array or torch tensor
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation, patches  # noqa: E402

from src.constants import (
    COURT_WIDTH,
    PAINT_LENGTH,
    PAINT_WIDTH,
    RIM_RADIUS,
    THREE_PT_CORNER_Y,
    THREE_PT_RADIUS,
)

# Derived court geometry in the rim frame (meters). The rim is 5.25 ft from
# the baseline and the half-court line is 47 ft from the baseline.
BASELINE_X = -(5.25 * 0.3048)          # -1.60 m
MIDCOURT_X = -(47.0 * 0.3048 - 5.25 * 0.3048)  # -12.73 m
BACKBOARD_X = -1.22                    # backboard face (4 ft from baseline)

OFF_COLORS = ["#c8102e", "#e0592a", "#e88b3a", "#f0b35b", "#f7d98c"]
DEF_COLORS = ["#1d428a", "#2a5cb0", "#3a76c4", "#5a92d4", "#7aaee4"]


def draw_half_court(ax: plt.Axes) -> None:
    """Half court in the rim frame: rim at origin, baseline toward -x."""
    # court rectangle: baseline -> midcourt
    ax.add_patch(patches.Rectangle((MIDCOURT_X, -COURT_WIDTH / 2),
                                   BASELINE_X - MIDCOURT_X, COURT_WIDTH,
                                   fill=False, edgecolor="black", lw=1.5))
    # rim + backboard
    ax.add_patch(patches.Circle((0, 0), RIM_RADIUS, fill=False, edgecolor="orange", lw=2))
    ax.add_patch(patches.Rectangle((BACKBOARD_X - 0.06, -RIM_RADIUS), 0.06, 2 * RIM_RADIUS,
                                   fill=True, edgecolor="black", facecolor="black"))
    # the paint: baseline to the free-throw line
    ax.add_patch(patches.Rectangle((BASELINE_X, -PAINT_WIDTH / 2), PAINT_LENGTH, PAINT_WIDTH,
                                   fill=False, edgecolor="black", lw=1))
    ax.add_patch(patches.Arc((BASELINE_X + PAINT_LENGTH, 0), 2 * 1.8, 2 * 1.8,
                             theta1=-90, theta2=90, edgecolor="black", lw=1))
    # 3-point line: arc around the rim + corner straight lines to the baseline
    y_c = THREE_PT_CORNER_Y
    r = THREE_PT_RADIUS
    x_apex = np.sqrt(max(r**2 - y_c**2, 0.0))  # arc meets corner lines at |y| = y_c
    theta = np.degrees(np.arctan2(y_c, x_apex))
    ax.add_patch(patches.Arc((0, 0), 2 * r, 2 * r, theta1=180 - theta, theta2=180 + theta,
                             edgecolor="black", lw=1))
    ax.plot([BASELINE_X, -x_apex], [y_c, y_c], color="black", lw=1)
    ax.plot([BASELINE_X, -x_apex], [-y_c, -y_c], color="black", lw=1)
    # restricted-area arc under the rim
    ax.add_patch(patches.Arc((0, 0), 2 * 1.25, 2 * 1.25, theta1=90, theta2=270,
                             edgecolor="gray", lw=0.8))


def render_gif(traj, path: str | Path, fps: int = 25) -> Path:
    """Render a (T, 11, 4) trajectory tensor to an MP4 of the half court.

    Accepts numpy arrays or torch tensors on any device (CUDA tensors are
    copied to host memory first).
    """
    if "torch" in str(type(traj)):
        traj = traj.detach().cpu().numpy()
    traj = np.asarray(traj)
    T = traj.shape[0]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    draw_half_court(ax)
    ax.set_xlim(MIDCOURT_X - 0.7, 1.2)
    ax.set_ylim(-COURT_WIDTH / 2 - 0.5, COURT_WIDTH / 2 + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    dots, trails = [], []
    for i in range(11):
        color = "black" if i == 10 else (OFF_COLORS[i] if i < 5 else DEF_COLORS[i - 5])
        marker = "h" if i == 10 else "o"
        size = 90 if i == 10 else 140
        (d,) = ax.plot([], [], marker, color=color, ms=size / 10, zorder=5)
        (tr,) = ax.plot([], [], "-", color=color, lw=1.0, alpha=0.45, zorder=2)
        dots.append(d)
        trails.append(tr)

    def update(frame: int):
        xy = traj[frame, :, :2]
        for i in range(11):
            dots[i].set_data([xy[i, 0]], [xy[i, 1]])
            trails[i].set_data(traj[: frame + 1, i, 0], traj[: frame + 1, i, 1])
        return dots + trails

    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(str(out), writer=writer)
    plt.close(fig)
    return out
