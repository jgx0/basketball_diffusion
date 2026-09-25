"""Generate counterfactual trajectory sets from a trained checkpoint.

Examples
--------
Same offensive anchor, mutate the defensive scheme:
    python scripts/generate.py --ckpt outputs/checkpoints/last.pt --schemes drop,switch,blitz

Star-handler swap:
    python scripts/generate.py --ckpt outputs/checkpoints/last.pt --star-swap
"""

from __future__ import annotations

import argparse
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


if __name__ == "__main__":
    main()
