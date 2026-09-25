"""Court-frame normalization, team inference, and PnR labeling utilities.

Full-court SportVU frame: meters, origin top-left, x along the length, hoops
at x = HOOP_X_FULL (left) and x = COURT_LENGTH - HOOP_X_FULL (right), y = 25 ft.
We map every possession into a **rim-centric half-court frame**:

    - the basket being attacked sits at the origin (0, 0),
    - the offense always attacks toward +x (away from the rim),
    - agents are reordered [offense 1-5, defense 1-5, ball].

Team assignment comes from SportVU ``team_id`` fields; the attacking hoop and
the offense/defense split are inferred from ball drift + centroid proximity.
"""

from __future__ import annotations

import numpy as np

from src.sportvu import FEET_TO_METERS

COURT_LENGTH = 94.0 * FEET_TO_METERS            # 28.65 m
COURT_WIDTH_FULL = 50.0 * FEET_TO_METERS        # 15.24 m
HOOP_X_FULL = 5.25 * FEET_TO_METERS             # rim center from baseline (m)
HOOP_Y_FULL = 25.0 * FEET_TO_METERS             # mid-lane (m)
HOOP_LEFT = np.array([HOOP_X_FULL, HOOP_Y_FULL])
HOOP_RIGHT = np.array([COURT_LENGTH - HOOP_X_FULL, HOOP_Y_FULL])


def velocities_from_positions(xy: np.ndarray, fps: float = 25.0) -> np.ndarray:
    """(T, 11, 2) positions -> (T, 11, 2) central-difference velocities."""
    v = np.gradient(xy, 1.0 / fps, axis=0)
    return v.astype(np.float32)


def denoise_run(xy: np.ndarray, max_speed: float = 13.0) -> np.ndarray | None:
    """Savitzky-Golay smoothing + tracker-spike rejection for one run.

    Single-frame position spikes (tracker glitches) are replaced by a median
    filter; the run is then smoothed (.savzol, 7-frame window) and rejected
    entirely if any player still exceeds `max_speed` (biomechanically
    impossible at 25 fps). Returns None if the run should be discarded.
    """
    from scipy.signal import savgol_filter, medfilt

    if len(xy) < 9:
        return None
    xy = xy.copy()
    # median filter kills isolated spikes (kernel 5 along time)
    for i in range(xy.shape[1]):
        for c in range(2):
            xy[:, i, c] = medfilt(xy[:, i, c], kernel_size=5)
    # light smoothing
    xy = savgol_filter(xy, window_length=7, polyorder=2, axis=0).astype(np.float32)
    # reject runs with residual impossible motion
    v = np.linalg.norm(np.diff(xy, axis=0), axis=-1) / (1.0 / 25.0)
    if v.max() > max_speed:
        return None
    return xy


def tensorize_window(xy: np.ndarray, fps: float = 25.0) -> np.ndarray:
    """(T, 11, 2) xy window -> (T, 11, 4) [x, y, vx, vy] float32."""
    vel = velocities_from_positions(xy, fps)
    return np.concatenate([xy, vel], axis=-1).astype(np.float32)


def estimate_attack_direction(run_xy: np.ndarray) -> bool:
    """True if the offense attacks the +x hoop, from the ball's net drift."""
    ball_x = run_xy[:, 10, 0]
    drift = np.polyfit(np.arange(len(ball_x)), ball_x, 1)[0]
    if abs(drift) < 1e-3:  # degenerate: fall back on start position
        return bool(abs(ball_x[0] - HOOP_RIGHT[0]) < abs(ball_x[0] - HOOP_LEFT[0]))
    return drift > 0


def reorder_to_offense_defense(
    run_xy: np.ndarray,
    entity_meta: np.ndarray,
    attacking_right: bool,
) -> np.ndarray:
    """Reorder an (T, 11, 2) run to [offense 1-5, defense 5, ball] slots.

    Offense = the 5-player team whose centroid is nearer the attacked hoop
    (valid for established half-court possessions, which is all we keep).
    """
    hoop_x = (COURT_LENGTH - HOOP_X_FULL) if attacking_right else HOOP_X_FULL
    team_ids = entity_meta[:-1, 0]  # (10,)
    uniq = np.unique(team_ids)
    if len(uniq) != 2:
        return run_xy  # malformed: leave untouched (filtered later)
    players_xy = run_xy[:, :10]  # (T, 10, 2), ball excluded
    centroids = {t: players_xy[:, team_ids == t].reshape(-1, 2).mean(axis=0) for t in uniq}
    d_att = {t: abs(c[0] - hoop_x) for t, c in centroids.items()}
    off_team = min(d_att, key=d_att.get)
    off_idx = np.where(team_ids == off_team)[0]
    def_idx = np.where(team_ids != off_team)[0]
    order = np.concatenate([off_idx, def_idx, [10]])
    return run_xy[:, order, :]


def normalize_to_rim_frame(reordered_xy: np.ndarray, attacking_right: bool) -> np.ndarray:
    """Map an offense/defense-ordered full-court run to the rim-centric frame.

    x' = distance from the attacked rim (attack toward +x'),
    y' = signed lateral offset from the rim's y (25 ft court centerline),
    mirrored together for left-attacking possessions to preserve handedness.
    """
    hoop_x = (COURT_LENGTH - HOOP_X_FULL) if attacking_right else HOOP_X_FULL
    out = reordered_xy.copy()
    out[..., 0] = reordered_xy[..., 0] - hoop_x
    out[..., 1] = reordered_xy[..., 1] - HOOP_Y_FULL
    if not attacking_right:
        out[..., 0] = -out[..., 0]
        out[..., 1] = -out[..., 1]
    return out


def detect_ball_screen(
    xy: np.ndarray,
    handler: int = 0,
    screener: int = 1,
    ball_defender: int = 5,
    screener_defender: int = 6,
    contact_dist: float = 1.2,
    min_contact_frames: int = 4,
) -> int | None:
    """Frame where the screener first sustains contact with the on-ball defender."""
    d = np.linalg.norm(xy[:, screener] - xy[:, ball_defender], axis=-1)
    close = d < contact_dist
    run = 0
    for t in range(len(close)):
        run = run + 1 if close[t] else 0
        if run >= min_contact_frames:
            return t - min_contact_frames + 1
    return None


def detect_pass(xy: np.ndarray, handler: int = 0, thresh: float = 2.8, hold: int = 6) -> int | None:
    """First frame where the ball detaches from the handler (throw in flight)."""
    d = np.linalg.norm(xy[:, 10] - xy[:, handler], axis=-1)
    run = 0
    for t in range(len(d)):
        run = run + 1 if d[t] > thresh else 0
        if run >= hold:
            return t - hold + 1
    return None


def label_scheme(
    xy: np.ndarray,
    screen_t: int,
    handler: int = 0,
    screener: int = 1,
    ball_defender: int = 5,
    screener_defender: int = 6,
) -> int:
    """Heuristic PnR coverage labeling on a rim-frame window.

    Decision is read ~0.5 s after screen contact:
      switch  - the screener's defender now guards the handler (closest man),
      blitz   - both ball-side defenders converge on the handler,
      drop    - the screener's defender stays deep toward the rim.

    Returns 0=drop, 1=switch, 2=blitz.
    """
    T = xy.shape[0]
    t_eval = min(screen_t + 12, T - 1)  # ~0.5 s after contact

    d6_handler = np.linalg.norm(xy[t_eval, screener_defender] - xy[t_eval, handler])
    d6_screener = np.linalg.norm(xy[t_eval, screener_defender] - xy[t_eval, screener])
    d5_handler = np.linalg.norm(xy[t_eval, ball_defender] - xy[t_eval, handler])

    # blitz: both ball-side defenders converged on the handler
    if max(d5_handler, d6_handler) < 2.2:
        return 2
    # switch: the big has taken the handler (close to him AND closer than to
    # his original man)
    if d6_handler < 2.5 and d6_handler < d6_screener - 0.3:
        return 1
    # drop: big sagged rim-ward (rim at x'=0, so depth means larger x' than
    # the screener) and well away from the handler
    if d6_handler > 3.0 and xy[t_eval, screener_defender, 0] > xy[t_eval, screener, 0] + 0.8:
        return 0
    # fallback: nearest of the three by margin
    return int(
        np.argmax(
            [
                float(d6_handler > 3.0 and xy[t_eval, screener_defender, 0] > xy[t_eval, screener, 0] + 0.8),
                float(d6_handler < 2.5 and d6_handler < d6_screener - 0.3),
                float(max(d5_handler, d6_handler) < 2.2),
            ]
        )
    )


def resample_to_hz(xy: np.ndarray, fps_in: float = 25.0, fps_out: float = 25.0) -> np.ndarray:
    """Linear time resampling."""
    if fps_in == fps_out:
        return xy
    n_in = xy.shape[0]
    n_out = int(round(n_in * fps_out / fps_in))
    idx = np.linspace(0, n_in - 1, n_out)
    out = np.zeros((n_out, xy.shape[1], 2), dtype=xy.dtype)
    for i in range(xy.shape[1]):
        for c in range(2):
            out[:, i, c] = np.interp(idx, np.arange(n_in), xy[:, i, c])
    return out
