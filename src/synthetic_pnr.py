"""Synthetic behavioral simulator of half-court pick-and-roll possessions.

This module stands in for real optical tracking when SportVU data is unavailable
and doubles as a *transparent baseline* for evaluation: it encodes hand-written
basketball tactics (drop / switch / blitz coverages, star-vs-role handler
behavior) so the trained diffusion model can be compared against a fully
interpretable generator, and so counterfactual experiments have ground truth.

Frame conventions
-----------------
- Frame 0 is the anchor instant: the screen is considered "set" a few frames in.
- The basket is at the origin (0, 0); offense attacks toward the rim.
- Coordinates are meters; 25 fps -> dt = 0.04 s.

Tactical sketches encoded here (deliberately simple, parameterized by the
conditioning labels so counterfactuals change behavior):
- Drop:     big sags toward the paint, cushioning the roller.
- Switch:   defender of the screener picks up the handler; the original
            on-ball defender peels off onto the roller (size mismatch).
- Blitz:    both defenders converge on the handler; a pass out frees the roll.
- Off-ball: weakside players drift/space; defenders sag one pass away.
- Star handler: faster pull-up decision, sharper change of pace, attracts the
            weakside helper slightly earlier (gravity).
"""

from __future__ import annotations

import numpy as np

from src.constants import (
    A_MAX,
    BALL_IDX,
    DEFENSE_SLICE,
    FEAT_DIM,
    N_AGENTS,
    PAINT_LENGTH,
    R_MIN,
    SEQ_LEN,
    V_MAX_PLAYER,
    V_MAX_BALL_PASS,
)

DT = 1.0 / 25.0
SCHEMES = ("drop", "switch", "blitz")


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _approach(cur: np.ndarray, target: np.ndarray, speed: float) -> np.ndarray:
    """Velocity that moves from cur toward target with capped speed and ease-out."""
    delta = target - cur
    dist = float(np.linalg.norm(delta))
    v = speed * delta / max(dist, 1e-6) * min(1.0, dist / 0.75)
    return v


def _integrate(pos: np.ndarray, vel: np.ndarray, acc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Semi-implicit Euler step with acceleration cap."""
    speed = np.linalg.norm(acc)
    if speed > A_MAX:
        acc = acc * (A_MAX / speed)
    vel = vel + acc * DT
    v = np.linalg.norm(vel)
    if v > V_MAX_PLAYER:
        vel = vel * (V_MAX_PLAYER / v)
    return pos + vel * DT, vel


def simulate_pnr(
    rng: np.random.Generator,
    scheme: str = "drop",
    star_handler: bool = False,
    seq_len: int = SEQ_LEN,
) -> dict:
    """Simulate one pick-and-roll possession.

    Returns dict with:
        traj : (T, N_AGENTS, FEAT_DIM) float32 — positions & velocities
        meta : dict with scheme / star_handler / handler role index
    """
    assert scheme in SCHEMES, f"unknown scheme {scheme!r}"
    T = seq_len

    # ------------------------------------------------------------------
    # Initial formation (frame 0, meters, rim at origin)
    # ------------------------------------------------------------------
    # offense: 0=handler, 1=screener, 2..4 = weakside spacing
    off = np.array(
        [
            [-8.4, -3.2],    # handler above the break, left slot
            [-6.0, -0.9],    # screener (big), slightly inside the slot
            [-7.6, 4.6],     # wing right
            [-5.2, 6.2],     # corner right
            [-9.6, 0.0],     # weakside top spacer
        ],
        dtype=np.float64,
    )
    deff = off + np.array(
        [
            [-1.0, 0.35],    # on-ball, plays a cushion
            [0.35, -0.4],    # screener's defender, ball-side (screen side)
            [-0.9, -0.5],
            [-0.8, -0.7],
            [-1.1, 0.3],
        ],
        dtype=np.float64,
    )
    ball = off[0].copy() + np.array([0.12, -0.05])

    # small per-play jitter so the dataset isn't degenerate
    off += rng.normal(0.0, 0.18, size=off.shape)
    deff += rng.normal(0.0, 0.18, size=deff.shape)

    pos = np.concatenate([off, deff, ball[None, :]], axis=0)  # (11, 2)
    vel = np.zeros((N_AGENTS, 2), dtype=np.float64)
    vel[DEFENSE_SLICE] = rng.normal(0.0, 0.35, size=(5, 2))

    traj = np.zeros((T, N_AGENTS, FEAT_DIM), dtype=np.float32)

    # ------------------------------------------------------------------
    # Tactical timeline (in frames)
    # ------------------------------------------------------------------
    handler_gravity = 1.25 if star_handler else 1.0
    t_screen = int(0.16 * T)                 # screen contact
    t_react = t_screen + int(0.10 * T)       # defense commits to the scheme
    t_pass = t_react + int(0.16 * T)         # handler releases the throw (blitz earlier)
    if scheme == "blitz":
        t_pass = t_react + int(0.09 * T)
    t_pull = int(0.62 * T) if star_handler else int(0.74 * T)  # pull-up / floater release

    passed = False
    released = False
    receiver = 2  # pass target when blitzed (weakside wing)

    for t in range(T):
        # ---------------- offense: handler ----------------
        h_vel = np.zeros(2)
        if not passed:
            if t < t_screen:
                # drive off the screen: toward a target beyond the screen spot
                drive_target = pos[1] + np.array([2.6 * handler_gravity, 1.9])
                h_vel = _approach(pos[0], drive_target, 4.6)
            elif t < t_pass:
                # slow down into the middle of the scheme
                h_vel = _approach(pos[0], np.array([-4.6, 0.4]), 2.4)
            else:
                passed = True
        else:
            h_vel = _approach(pos[0], pos[0] + np.array([-0.5, 0.6]), 1.0)

        # ---------------- offense: screener/roller ----------------
        if t < t_screen:
            r_vel = _approach(pos[1], pos[0] + np.array([0.55, -0.55]), 2.6)
        else:
            roll_target = np.array([-1.35, 0.55])  # rim area, slightly ball-side
            roll_speed = 3.4
            r_vel = _approach(pos[1], roll_target, roll_speed)

        # ---------------- offense: spacers ----------------
        spacer_targets = np.array(
            [
                [-7.4, 5.4],
                [-5.0, 6.7],
                [-9.9, 0.4],
            ],
            dtype=np.float64,
        )
        s_vel = [ _approach(pos[2 + i], spacer_targets[i], 1.7) for i in range(3) ]

        # ---------------- ball ----------------
        if not passed:
            b_vel = _approach(pos[BALL_IDX], pos[0] + np.array([0.1, -0.1]), 5.0)
        else:
            b_vel = _approach(pos[BALL_IDX], pos[receiver], V_MAX_BALL_PASS)

        # ---------------- defense ----------------
        d_vel = [None] * 5
        if t < t_react:
            # shadow man-to-man until the screen forces a decision
            d_vel[0] = _approach(pos[5], pos[0] + np.array([-0.85, 0.3]), 3.6)
            d_vel[1] = _approach(pos[6], pos[1] + np.array([0.3, -0.3]), 3.4)
        else:
            if scheme == "drop":
                d_vel[0] = _approach(pos[5], pos[0] + np.array([-0.9, 0.25]), 3.2)
                # screener's defender sags to the level of the screen / paint
                drop_target = np.array([-PAINT_LENGTH * 0.75, 0.3])
                d_vel[1] = _approach(pos[6], drop_target, 2.6)
            elif scheme == "switch":
                # big takes the handler, on-ball defender peels to the roller
                d_vel[0] = _approach(pos[6], pos[0] + np.array([-0.7, 0.2]), 3.0)
                d_vel[1] = _approach(pos[5], pos[1] + np.array([0.35, -0.1]), 3.2)
            else:  # blitz
                b1 = _approach(pos[5], pos[0] + np.array([-0.55, -0.45]), 4.2)
                b2 = _approach(pos[6], pos[0] + np.array([-0.55, 0.45]), 4.2)
                d_vel[0], d_vel[1] = b1, b2

        # weakside defenders: sag toward the gap, help on the roller late
        help_target = np.array([-2.8, 1.6])
        d_vel[2] = _approach(pos[7], help_target if t > t_react else pos[7] + np.array([0.4, 0.0]), 2.4)
        d_vel[3] = _approach(pos[8], pos[3] + np.array([-0.55, -0.5]), 2.4)
        d_vel[4] = _approach(pos[9], pos[4] + np.array([-0.6, 0.2]), 2.4)

        # star gravity: extra weakside shrink
        if star_handler and t > t_react:
            d_vel[2] = d_vel[2] - np.array([0.5, 0.0])

        # ---------------- integrate everyone ----------------
        accs = np.zeros((N_AGENTS, 2))
        accs[0] = (h_vel - vel[0]) / DT
        accs[1] = (r_vel - vel[1]) / DT
        accs[2:5] = np.stack([(s_vel[i] - vel[2 + i]) / DT for i in range(3)])
        accs[5:10] = np.stack([(d_vel[i] - vel[5 + i]) / DT for i in range(5)])
        accs[BALL_IDX] = (b_vel - vel[BALL_IDX]) / DT

        # slight damped-noise jerk for organic motion
        accs += rng.normal(0.0, 1.1, size=accs.shape)

        for i in range(N_AGENTS):
            pos[i], vel[i] = _integrate(pos[i], vel[i], accs[i])

        # ball stickiness: never overlap the handler hard
        d_ball = np.linalg.norm(pos[BALL_IDX] - pos[0])
        if d_ball < R_MIN:
            push = (pos[BALL_IDX] - pos[0]) / max(d_ball, 1e-6) * (R_MIN - d_ball)
            pos[BALL_IDX] += push

        traj[t, :, :2] = pos
        traj[t, :, 2:] = vel

    # post-hoc: after `t_pull` freeze the handler shot motion (simple stop)
    meta = {"scheme": scheme, "star_handler": bool(star_handler), "t_pass": int(t_pass)}
    return {"traj": traj, "meta": meta}


def build_dataset(
    n_samples: int,
    seed: int = 0,
    seq_len: int = SEQ_LEN,
) -> dict:
    """Simulate `n_samples` possessions with balanced scheme/star labels.

    Returns dict of tensors:
        traj  (n, T, 11, 4) float32
        scheme (n,) int64  in {0: drop, 1: switch, 2: blitz}
        star   (n,) int64  in {0, 1}
    """
    rng = np.random.default_rng(seed)
    trajs = np.zeros((n_samples, seq_len, N_AGENTS, FEAT_DIM), dtype=np.float32)
    schemes = np.zeros(n_samples, dtype=np.int64)
    stars = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        scheme = SCHEMES[i % len(SCHEMES)]
        star = bool(rng.integers(0, 2))
        out = simulate_pnr(rng, scheme=scheme, star_handler=star, seq_len=seq_len)
        trajs[i] = out["traj"]
        schemes[i] = SCHEMES.index(out["meta"]["scheme"])
        stars[i] = int(out["meta"]["star_handler"])
    return {"traj": trajs, "scheme": schemes, "star": stars}
