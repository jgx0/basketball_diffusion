"""Generate counterfactual trajectory sets from a trained checkpoint.

Examples
--------
Same offensive anchor, mutate the defensive scheme:
    python scripts/generate.py --ckpt outputs/checkpoints/last.pt --schemes drop,switch,blitz

Star-handler swap:
    python scripts/generate.py --ckpt outputs/checkpoints/last.pt --star-swap

Statistically resolvable branch study (paired design, cluster bootstrap):
    python scripts/generate.py --ckpt outputs/checkpoints/last.pt \
        --branch-study 256 --anchors 8 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import SEQ_LEN  # noqa: E402
from src.dataset import PnRTrajectoryDataset, collate, load_norm_stats  # noqa: E402
from src.loss import epv_proxy, paint_pressure, roller_openness  # noqa: E402
from src.model import GaussianDiffusion, TemporalUNet  # noqa: E402


def load_diffusion(ckpt_path: str, device: str) -> tuple[GaussianDiffusion, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    unet = TemporalUNet(base=cfg["base_channels"], time_dim=cfg["time_dim"], cond_dim=cfg["cond_dim"])
    diffusion = GaussianDiffusion(unet, n_steps=cfg["n_steps"]).to(device)
    diffusion.load_state_dict(ckpt["ema"] if "ema" in ckpt else ckpt["model"])
    diffusion.eval()
    return diffusion, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/last.pt")
    ap.add_argument("--data-dir", default="outputs/processed_tensors")
    ap.add_argument("--out", default="outputs/counterfactuals")
    ap.add_argument("--schemes", default="drop,switch,blitz")
    ap.add_argument("--n-per-scheme", type=int, default=16)
    ap.add_argument("--branch-study", type=int, default=0,
                    help="if > 0, run the paired branch study with this M per branch "
                         "(overrides --n-per-scheme; writes branch_study.json)")
    ap.add_argument("--anchors", type=int, default=8,
                    help="number of validation anchors in the branch study")
    ap.add_argument("--chunk", type=int, default=64,
                    help="sampling batch size per call (memory bound)")
    ap.add_argument("--save-trajs", action="store_true",
                    help="also save all branch-study trajectories (large)")
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--physics-guidance", type=float, default=0.5,
                    help="sampling-time physics gradient strength (0 disables)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--star-swap", action="store_true")
    ap.add_argument("--render", action="store_true", help="render mp4s via src.visualize")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    diffusion, cfg = load_diffusion(args.ckpt, device)
    steps = args.steps or cfg["sample_steps"]
    norm_stats = load_norm_stats(args.data_dir)

    if args.branch_study > 0:
        run_branch_study(diffusion, cfg, args, device, norm_stats or None)
        return

    ds = PnRTrajectoryDataset(args.data_dir, split="val")
    b = collate([ds[i] for i in range(min(args.n_per_scheme, len(ds)))])
    anchor = b["anchor"].to(device)
    n = anchor.shape[0]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scheme_names = args.schemes.split(",")
    for s_idx, name in enumerate(scheme_names):
        cond = {
            "scheme": torch.full((n,), s_idx, dtype=torch.long, device=device),
            "star": torch.zeros(n, dtype=torch.long, device=device),
            "anchor": anchor,
        }
        gen = diffusion.ddim_sample(
            cond, steps=steps, guidance=args.guidance,
            physics_guidance=args.physics_guidance, norm_stats=norm_stats or None,
            device=device,
        )
        np.save(out_dir / f"scheme_{name}.npy", gen.cpu().numpy())
        print(
            f"{name:7s} | roller openness {roller_openness(gen).mean():.3f} "
            f"| paint pressure {paint_pressure(gen).mean():.2f} m "
            f"| EPV proxy {epv_proxy(gen).mean():.3f}"
        )
        if args.render:
            try:
                from src.visualize import render_gif  # local import

                render_gif(gen[0].cpu().numpy(), str(out_dir / f"scheme_{name}.mp4"))
            except Exception as e:  # rendering must never kill the metrics run
                print(f"  [warn] render failed for {name}: {e}")

    if args.star_swap:
        c0 = {
            "scheme": torch.zeros(n, dtype=torch.long, device=device),
            "star": torch.zeros(n, dtype=torch.long, device=device),
            "anchor": anchor,
        }
        c1 = {**c0, "star": torch.ones(n, dtype=torch.long, device=device)}
        g0 = diffusion.ddim_sample(c0, steps=steps, guidance=args.guidance,
                                   physics_guidance=args.physics_guidance,
                                   norm_stats=norm_stats or None, device=device)
        g1 = diffusion.ddim_sample(c1, steps=steps, guidance=args.guidance,
                                   physics_guidance=args.physics_guidance,
                                   norm_stats=norm_stats or None, device=device)
        np.save(out_dir / "handler_role.npy", g0.cpu().numpy())
        np.save(out_dir / "handler_star.npy", g1.cpu().numpy())
        print(f"star vs role handler saved; EPV delta {epv_proxy(g1).mean() - epv_proxy(g0).mean():+.4f}")


def run_branch_study(
    diffusion: GaussianDiffusion,
    cfg: dict,
    args: argparse.Namespace,
    device: str,
    norm_stats: dict | None,
) -> None:
    """Paired counterfactual branch study (paper \u00a7 protocol).

    For each of `args.anchors` validation anchors and each branch (first
    scheme in --schemes is the reference), sample `args.branch_study` futures
    in chunks, reduce each future to vulnerability sufficient statistics,
    and run the paired cluster-bootstrap analysis of src.branch_study.
    """
    from src.branch_study import BranchStudy

    M = args.branch_study
    A = max(1, args.anchors)
    ds = PnRTrajectoryDataset(args.data_dir, split="val")
    if len(ds) < A:
        raise SystemExit(f"branch study needs {A} validation anchors; cache has {len(ds)}")
    anchor_batch = collate([ds[i] for i in range(A)])
    anchors = anchor_batch["anchor"].to(device)
    base_star = anchor_batch["star"].to(device)

    scheme_names = args.schemes.split(",")
    Z = len(scheme_names)
    metrics = ("roller_openness", "paint_pressure", "epv_proxy")
    phi = {m: np.full((A, Z, M), np.nan, dtype=np.float64) for m in metrics}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = args.steps or cfg["sample_steps"]
    chunk = max(1, args.chunk)

    for z, name in enumerate(scheme_names):
        for a in range(A):
            for lo in range(0, M, chunk):
                hi = min(lo + chunk, M)
                n = hi - lo
                cond = {
                    "scheme": torch.full((n,), z, dtype=torch.long, device=device),
                    "star": base_star[a : a + 1].expand(n),
                    "anchor": anchors[a : a + 1].expand(n, -1, -1),
                }
                gen = diffusion.ddim_sample(
                    cond, steps=steps, guidance=args.guidance,
                    physics_guidance=args.physics_guidance,
                    norm_stats=norm_stats, device=device,
                )
                phi["roller_openness"][a, z, lo:hi] = roller_openness(gen).cpu().numpy()
                phi["paint_pressure"][a, z, lo:hi] = paint_pressure(gen).cpu().numpy()
                phi["epv_proxy"][a, z, lo:hi] = epv_proxy(gen).cpu().numpy()
                if args.save_trajs:
                    np.save(
                        out_dir / f"bstudy_{name}_a{a:03d}_{lo:04d}-{hi:04d}.npy",
                        gen.cpu().numpy(),
                    )
        print(f"  branch {name}: sampled {A} anchors x {M} futures")

    # first scheme in the list is the reference branch
    order = [0] + [z for z in range(1, Z)]
    report = {}
    for metric in metrics:
        study = BranchStudy(phi=phi[metric][:, order, :], branch_names=[scheme_names[z] for z in order])
        rep = study.summary(n_boot=2000, seed=0)
        rep["required_m"] = study.required_m(target_se=0.02, n_boot=500, seed=1)
        report[metric] = rep

    np.savez_compressed(
        out_dir / "branch_study_phi.npz",
        **{m: phi[m] for m in metrics},
        branches=np.array(scheme_names),
    )
    (out_dir / "branch_study.json").write_text(json.dumps(report, indent=2))

    print(f"\nbranch study ({A} anchors x {M} futures x {Z} branches) -> {out_dir}")
    for metric in metrics:
        print(f"\n{metric}: " + ", ".join(
            f"{n}={v:.3f}" for n, v in report[metric]["branch_means"].items()
        ))
        for pair in report[metric]["bootstrap"]["pairs"]:
            print(
                f"  {pair['branch']} vs {pair['reference']}: "
                f"delta {pair['delta']:+.3f} [{pair['ci_lo']:+.3f}, {pair['ci_hi']:+.3f}] "
                f"p(perm) {pair['p_permutation']:.3f}"
            )


if __name__ == "__main__":
    main()
