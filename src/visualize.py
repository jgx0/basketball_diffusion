"""Matplotlib rendering of half-court possessions to MP4 (requires ffmpeg).

Usage:
    from src.visualize import render_gif
    render_gif(traj, "out.mp4")   # traj: (T, 11, 4) numpy array
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation, patches  # noqa: E402

from src.constants import (
    COURT_LENGTH_HALF,
    COURT_WIDTH,
    PAINT_LENGTH,
    PAINT_WIDTH,
    RIM_RADIUS,
    THREE_PT_CORNER_Y,
    THREE_PT_RADIUS,
)

OFF_COLORS = ["#c8102e", "#e0592a", "#e88b3a", "#f0b35b", "#f7d98c"]
DEF_COLORS = ["#1d428a", "#2a5cb0", "#3a76c4", "#5a92d4", "#7aaee4"]


def draw_half_court(ax: plt.Axes) -> None:
    """Rim-centric half court: rim at origin, offense attacks toward +x."""
    ax.add_patch(patches.Rectangle((-1.2, -COURT_WIDTH / 2), COURT_LENGTH_HALF + 1.2, COURT_WIDTH,
                                   fill=False, edgecolor="black", lw=1.5))
    ax.add_patch(patches.Circle((0, 0), RIM_RADIUS, fill=False, edgecolor="orange", lw=2))
    ax.add_patch(patches.Rectangle((-0.95, -RIM_RADIUS), 0.575, 2 * RIM_RADIUS, fill=True,
                                   edgecolor="black", facecolor="black"))
    ax.add_patch(patches.Rectangle((-1.2, -PAINT_WIDTH / 2), PAINT_LENGTH, PAINT_WIDTH,
                                   fill=False, edgecolor="black", lw=1))
    ax.add_patch(patches.Arc((-1.2, 0), 2.4, 2.4, theta1=-90, theta2=90, edgecolor="black", lw=1))
    # 3pt arc: arc + corner straight lines
    theta_corner = np.degrees(np.arccos(
        np.clip((THREE_PT_RADIUS ** 2 - THREE_PT_CORNER_Y ** 2) ** 0.5 / THREE_PT_RADIUS, -1, 1)))
    ax.add_patch(patches.Arc((-1.2, 0), 2 * THREE_PT_RADIUS, 2 * THREE_PT_RADIUS,
                             theta1=-theta_corner, theta2=theta_corner, edgecolor="black", lw=1))
    corner_y = THREE_PT_CORNER_Y
    ax.plot([-1.2, corner_y * 0 + 0], [corner_y, corner_y], color="black", lw=1)
    ax.plot([-1.2, 0], [-corner_y, -corner_y], color="black", lw=1)


def render_gif(traj: np.ndarray, path: str | Path, fps: int = 25) -> Path:
    """Render a (T, 11, 4) trajectory tensor to an MP4 of the half court."""
    traj = np.asarray(traj)
    T = traj.shape[0]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    draw_half_court(ax)
    ax.set_xlim(-1.4, COURT_LENGTH_HALF)
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
