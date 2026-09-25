"""Evaluate a trained diffusion model against held-out real possessions.

Metrics
-------
Realism:
  - FTD  : Fréchet distance between Gaussians fitted to per-possession
            summary features (rim-distance histograms + spacing stats).
  - ADE  : average displacement error of generated vs. matched real
            trajectories under identical conditioning.
  - Kinematic compliance: fraction of player frames violating velocity /
            acceleration caps or pair-overlap limits.

Counterfactual vulnerability:
  - roller openness, paint pressure and EPV-proxy contrasts across schemes
    (drop/switch/blitz) and handler type (role/star).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import A_MAX, R_MIN, V_MAX_PLAYER  # noqa: E402
from src.dataset import PnRTrajectoryDataset, collate, load_norm_stats  # noqa: E402
from src.loss import (  # noqa: E402
    acceleration_penalty,
    epv_proxy,
    overlap_penalty,
    paint_pressure,
    roller_openness,
    velocity_penalty,
)
from src.model import GaussianDiffusion, TemporalUNet  # noqa: E402


# ---------------------------------------------------------------------------
# Feature extraction for FTD
# ---------------------------------------------------------------------------
def summary_features(traj: torch.Tensor) -> np.ndarray:
    """Per-possession summary statistics: (N, D).

    Features: mean/min rim distance per offensive role (5x2), mean pairwise
    offensive spacing, mean nearest-defender distance per offensive role,
    mean |v| per role. 10 + 1 + 5 + 5 = 21 dims.
    """
    xy = traj[..., :2]
    rim_d = xy[:, :, 0:5].norm(dim=-1)                    # (N, T, 5)
    f_rim_mean = rim_d.mean(dim=1)                        # (N, 5)
    f_rim_min = rim_d.min(dim=1).values                   # (N, 5)
    off = xy[:, :, 0:5]
    pd = (off.unsqueeze(-2) - off.unsqueeze(-3)).norm(dim=-1)
    eye = torch.eye(5, device=traj.device).bool()
    pd = pd.masked_fill(eye, torch.finfo(pd.dtype).max)
    f_spacing = pd.min(dim=-1).values.min(dim=-1).values.mean(dim=1)  # (N,)
    defs = xy[:, :, 5:10]
    nd = (off.unsqueeze(-2) - defs.unsqueeze(-3)).norm(dim=-1).min(dim=-1).values
    f_sep = nd.mean(dim=1)                                # (N, 5)
    v = traj[..., :5, 2:4].norm(dim=-1)
    f_speed = v.mean(dim=1)                               # (N, 5)
    feats = torch.cat(
        [f_rim_mean, f_rim_min, f_spacing[:, None], f_sep, f_speed], dim=-1
    )
    return feats.cpu().numpy()


def frechet_distance(mu1: np.ndarray, s1: np.ndarray, mu2: np.ndarray, s2: np.ndarray) -> float:
    """Fréchet distance between two Gaussians (trace formula)."""
    from scipy import linalg

    cov1, cov2 = np.cov(s1, rowvar=False), np.cov(s2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(cov1 @ cov2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov1) + np.trace(cov2) - 2 * np.trace(covmean))


def ade(real: torch.Tensor, gen: torch.Tensor) -> float:
    """Average displacement error over positions of all agents (meters)."""
    return float((real[..., :2] - gen[..., :2]).norm(dim=-1).mean())


def kinematic_compliance(traj: torch.Tensor) -> dict:
    """Fraction of player frames / pairs violating physical caps (lower is better)."""
    from src.constants import A_MAX, R_MIN, V_MAX_PLAYER

    xy = traj[..., :10, :2]
    dt = 0.04
    v = (xy[:, 2:] - xy[:, :-2]) / (2 * dt)
    speed = v.norm(dim=-1)
    vel_rate = float((speed > V_MAX_PLAYER).float().mean())
    a = (v[:, 1:] - v[:, :-1]) / dt
    acc_rate = float((a.norm(dim=-1) > A_MAX).float().mean())
    d = (xy.unsqueeze(-2) - xy.unsqueeze(-3)).norm(dim=-1)
    eye = torch.eye(10, device=traj.device).bool()
    dist = d.masked_fill(eye, float("inf"))
    overlap_rate = float((dist < R_MIN).float().mean())
    return {
        "vel_violation_rate": vel_rate,
        "acc_violation_rate": acc_rate,
        "overlap_violation_rate": overlap_rate,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/last.pt")
    ap.add_argument("--data-dir", default="outputs/processed_tensors")
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    norm_stats = load_norm_stats(args.data_dir) or None

    unet = TemporalUNet(base=cfg["base_channels"], time_dim=cfg["time_dim"], cond_dim=cfg["cond_dim"])
    diffusion = GaussianDiffusion(unet, n_steps=cfg["n_steps"]).to(device)
    diffusion.load_state_dict(ckpt["ema"] if "ema" in ckpt else ckpt["model"])
    diffusion.eval()

    val_ds = PnRTrajectoryDataset(args.data_dir, split="val")
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate)
    batches = list(val_dl)[: max(1, args.n_samples // 64)]
    real = torch.cat([b["traj"] for b in batches])[: args.n_samples].to(device)
    cond = {
        "scheme": torch.cat([b["scheme"] for b in batches])[: args.n_samples].to(device),
        "star": torch.cat([b["star"] for b in batches])[: args.n_samples].to(device),
        "anchor": torch.cat([b["anchor"] for b in batches])[: args.n_samples].to(device),
    }
    # dataset returns standardized tensors; all metrics are computed in meters
    if norm_stats is not None:
        mean = torch.as_tensor(norm_stats["mean"], device=device)
        std = torch.as_tensor(norm_stats["std"], device=device)
        real = real * std + mean

    # conditional generation anchored to the same initial states
    gen = diffusion.ddim_sample(
        {k: v.clone() for k, v in cond.items()},
        steps=cfg["sample_steps"],
        guidance=1.0,
        norm_stats=norm_stats or None,
        device=device,
    )

    # ---------------- realism ----------------
    f_real, f_gen = summary_features(real), summary_features(gen)
    ftd = frechet_distance(f_real.mean(0), f_real, f_gen.mean(0), f_gen)
    m_ade = ade(real, gen)
    compliance = kinematic_compliance(gen)

    # ---------------- counterfactual vulnerability ----------------
    # same anchor, vary scheme
    cf_base = {k: cond[k][:16].clone() for k in cond}
    cf_results = {}
    for s, name in ((0, "drop"), (1, "switch"), (2, "blitz")):
        c = {k: v.clone() for k, v in cf_base.items()}
        c["scheme"] = torch.full_like(cf_base["scheme"], s)
        out = diffusion.ddim_sample(c, steps=cfg["sample_steps"], norm_stats=norm_stats or None, device=device)
        cf_results[name] = {
            "roller_openness": float(roller_openness(out).mean()),
            "paint_pressure_m": float(paint_pressure(out).mean()),
            "epv_proxy": float(epv_proxy(out).mean()),
        }
    # star-vs-role handler contrast (same metrics on both branches)
    c_star = {k: v.clone() for k, v in cf_base.items()}
    c_star["star"] = torch.ones_like(cf_base["star"])
    c_role = {k: v.clone() for k, v in cf_base.items()}
    c_role["star"] = torch.zeros_like(cf_base["star"])
    g_star = diffusion.ddim_sample(c_star, steps=cfg["sample_steps"], norm_stats=norm_stats or None, device=device)
    g_role = diffusion.ddim_sample(c_role, steps=cfg["sample_steps"], norm_stats=norm_stats or None, device=device)
    cf_results["star_handler"] = {
        "roller_openness": float(roller_openness(g_star).mean()),
        "epv_proxy": float(epv_proxy(g_star).mean()),
    }
    cf_results["role_handler"] = {
        "roller_openness": float(roller_openness(g_role).mean()),
        "epv_proxy": float(epv_proxy(g_role).mean()),
    }

    report = {
        "ftd": ftd,
        "ade_m": m_ade,
        "compliance": compliance,
        "counterfactual": cf_results,
        "n_real": int(real.shape[0]),
        "n_gen": int(gen.shape[0]),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    np.save(out_dir / "generated_sample.npy", gen.cpu().numpy())
    print(json.dumps(report, indent=2))
    print(f"saved report to {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
