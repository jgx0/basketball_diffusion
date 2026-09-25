"""Physics-informed loss shaping for generated or predicted trajectories.

All terms are differentiable and expect tensors of shape (B, T, 11, 4) with
features ordered [x, y, vx, vy]. Velocities are re-derived from positions by
central differences (more reliable than trusting the model's own velocity
channels during sampling); the velocity channels are still supervised through
the diffusion MSE on x0-prediction.
"""

from __future__ import annotations

import torch

from src.constants import (
    A_MAX,
    BALL_IDX,
    COURT_LENGTH_HALF,
    COURT_WIDTH,
    N_PLAYERS,
    R_MIN,
    V_MAX_PLAYER,
)


def central_diff_velocity(x: torch.Tensor, dt: float = 0.04) -> torch.Tensor:
    """Positions (B, T, 11, 2) -> velocities (B, T, 11, 2) via central differences."""
    v = (x[:, 2:] - x[:, :-2]) / (2 * dt)
    v0 = (x[:, 1:2] - x[:, 0:1]) / dt
    vT = (x[:, -1:] - x[:, -2:-1]) / dt
    return torch.cat([v0, v, vT], dim=1)


def _reduce(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Scalar mean or per-sample (B,) mean over all non-batch dims."""
    if mode == "none":
        return x.flatten(1).mean(dim=1) if x.dim() > 1 else x
    return x.mean()


def velocity_penalty(
    traj: torch.Tensor, dt: float = 0.04, v_max: float = V_MAX_PLAYER, reduce: str = "mean"
) -> torch.Tensor:
    """Squared hinge on relative speed violation (players only), dimensionless."""
    xy = traj[..., :N_PLAYERS, :2]
    v = central_diff_velocity(xy, dt)
    speed = v.norm(dim=-1)
    return _reduce(torch.relu(speed / v_max - 1.0).pow(2), reduce)


def acceleration_penalty(
    traj: torch.Tensor, dt: float = 0.04, a_max: float = A_MAX, reduce: str = "mean"
) -> torch.Tensor:
    """Squared hinge on relative acceleration violation (players only).

    Normalized by a_max: finite-difference acceleration scales with 1/dt^2,
    so absolute hinges would numerically dominate the diffusion loss.
    """
    xy = traj[..., :N_PLAYERS, :2]
    v = central_diff_velocity(xy, dt)
    a = (v[:, 2:] - v[:, :-2]) / dt
    return _reduce(torch.relu(a.norm(dim=-1) / a_max - 1.0).pow(2), reduce)


def overlap_penalty(traj: torch.Tensor, r_min: float = R_MIN, reduce: str = "mean") -> torch.Tensor:
    """Hinge on relative pair-overlap violation (players only)."""
    xy = traj[..., :N_PLAYERS, :2]                       # (B, T, 10, 2)
    d = xy.unsqueeze(-2) - xy.unsqueeze(-3)              # (B, T, 10, 10, 2)
    dist = d.norm(dim=-1) + torch.eye(N_PLAYERS, device=traj.device) * 1e3
    return _reduce(torch.relu((r_min - dist) / r_min).pow(2), reduce)


def court_bounds_penalty(traj: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
    """Hinge on positions outside the half-court rectangle (normalized by tol).

    Rim-centric frame: x in (-1.0, COURT_LENGTH_HALF), y in (-W/2, W/2).
    Tolerance of 1.0 m so out-of-bounds plays aren't over-penalized.
    """
    xy = traj[..., :2]
    tol = 1.0
    lo_x = torch.relu(-(xy[..., 0]) - tol)   # x < -1 m (behind rim) penalized
    hi_x = torch.relu(xy[..., 0] - COURT_LENGTH_HALF - tol)
    lo_y = torch.relu(-(xy[..., 1] + COURT_WIDTH / 2) - tol)
    hi_y = torch.relu(xy[..., 1] - COURT_WIDTH / 2 - tol)
    return _reduce((lo_x.pow(2) + hi_x.pow(2) + lo_y.pow(2) + hi_y.pow(2)) / tol**2, reduce)


def physics_loss(
    traj: torch.Tensor,
    dt: float = 0.04,
    lam_vel: float = 1.0,
    lam_acc: float = 1.0,
    lam_col: float = 1.0,
    lam_bound: float = 0.5,
    reduce: str = "mean",
):
    """Weighted sum of physics terms.

    Returns (total, dict of components). With reduce="mean" each entry is a
    scalar; with reduce="none" each entry keeps the batch dim (B,).
    """
    r = (lambda t: t.mean()) if reduce == "mean" else (lambda t: t)
    terms = {
        "vel": r(velocity_penalty(traj, dt)),
        "acc": r(acceleration_penalty(traj, dt)),
        "col": r(overlap_penalty(traj)),
        "bound": r(court_bounds_penalty(traj)),
    }
    total = lam_vel * terms["vel"] + lam_acc * terms["acc"] + lam_col * terms["col"] + lam_bound * terms["bound"]
    return total, terms


# ---------------------------------------------------------------------------
# Counterfactual / vulnerability analytics
# ---------------------------------------------------------------------------
def roller_openness(traj: torch.Tensor, dt: float = 0.04, radius: float = 2.0) -> torch.Tensor:
    """Fraction of late frames where the roller (agent 1) has no defender within `radius`.

    traj: (B, T, 11, 4). Offense roles: 0 handler, 1 roller/screener.
    Defense occupies agents 5..9. Returns (B,) in [0, 1].
    """
    xy = traj[..., :2]
    roll = xy[:, :, 1]                                   # (B, T, 2)
    defs = xy[:, :, 5:10]                                # (B, T, 5, 2)
    d = (roll.unsqueeze(-2) - defs).norm(dim=-1)         # (B, T, 5)
    near = (d < radius).any(dim=-1)                      # (B, T)
    late = near[:, int(0.6 * traj.shape[1]):]
    return late.float().mean(dim=1)


def paint_pressure(traj: torch.Tensor, rim_radius: float = 1.0) -> torch.Tensor:
    """Mean min-distance of any defender to the roller while the roller is near the rim."""
    xy = traj[..., :2]
    roll = xy[:, :, 1]
    defs = xy[:, :, 5:10]
    near_rim = roll.norm(dim=-1) < rim_radius            # (B, T)
    d = (roll.unsqueeze(-2) - defs).norm(dim=-1).min(dim=-1).values  # (B, T)
    d = torch.where(near_rim, d, torch.full_like(d, torch.finfo(d.dtype).max))
    return d.clamp(max=10.0).mean(dim=1)


def epv_proxy(traj: torch.Tensor) -> torch.Tensor:
    """Cheap spatial value proxy: exponential in rim distance of the openest offensive player.

    Returns (B,) — higher means the offense generated more expected value.
    This is a placeholder for a learned EPV / shot-probability surface.
    """
    xy = traj[..., :2]
    off = xy[:, :, 0:5]
    defs = xy[:, :, 5:10]
    # separation of each offensive player to nearest defender
    d = (off.unsqueeze(-2) - defs.unsqueeze(-3)).norm(dim=-1)  # (B, T, 5, 5)
    sep = d.min(dim=-1).values                                  # (B, T, 5)
    rim_d = off.norm(dim=-1)                                    # (B, T, 5)
    # value: closer to rim & more separation is better
    val = torch.exp(-rim_d / 4.0) * torch.sigmoid(sep - 1.5)
    return val.mean(dim=(1, 2))


def scheme_drop_rate(pred: torch.Tensor, cond: dict) -> dict:
    """Counterfactual contrast: mean roller openness per scheme branch."""
    out = {}
    for s, name in ((0, "drop"), (1, "switch"), (2, "blitz")):
        mask = cond["scheme"] == s
        if mask.any():
            out[name] = roller_openness(pred[mask]).mean().item()
    return out
