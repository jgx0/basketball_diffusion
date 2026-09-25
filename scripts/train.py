"""Train the conditional spatiotemporal diffusion model.

Usage:
    python scripts/train.py --config configs/base.yaml [--epochs N] [--device cuda]

The config file is a flat `key: value` YAML subset parsed without PyYAML.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.dataset import PnRTrajectoryDataset, collate  # noqa: E402
from src.loss import physics_loss  # noqa: E402
from src.model import GaussianDiffusion, TemporalUNet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    torch.manual_seed(cfg["seed"])

    train_ds = PnRTrajectoryDataset(cfg["data_dir"], split="train")
    val_ds = PnRTrajectoryDataset(cfg["data_dir"], "val")
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate)
    print(f"train={len(train_ds)} val={len(val_ds)} device={device}")

    unet = TemporalUNet(
        base=cfg["base_channels"], time_dim=cfg["time_dim"], cond_dim=cfg["cond_dim"]
    )
    diffusion = GaussianDiffusion(unet, n_steps=cfg["n_steps"]).to(device)
    opt = torch.optim.AdamW(diffusion.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ema = copy.deepcopy(diffusion).eval().requires_grad_(False)

    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def ema_update() -> None:
        with torch.no_grad():
            for p_ema, p in zip(ema.state_dict().values(), diffusion.state_dict().values()):
                if p_ema.dtype.is_floating_point:
                    p_ema.lerp_(p, 1.0 - cfg["ema_decay"])
                else:
                    p_ema.copy_(p)

    step = 0
    best_val = float("inf")
    for epoch in range(cfg["epochs"]):
        diffusion.train()
        t0 = time.time()
        for batch in train_dl:
            x0 = batch["traj"].to(device)
            cond = {k: batch[k].to(device) for k in ("scheme", "star", "anchor")}
            noise_loss = diffusion.p_losses(x0, cond, cond_drop=cfg["cfg_dropout"])

            # Physics regularization on the x0 implied by a low-noise denoise probe.
            # At low noise the x0 reconstruction is well-conditioned, so physics
            # terms measure genuine violations the model would imprint; at high
            # noise they would only measure reconstruction noise (1/dt^2 amplified),
            # destabilizing training. As the model converges, eps error -> 0 and
            # any true physics violation shows up and gets penalized.
            low = max(1, cfg["n_steps"] // 8)
            t_probe = torch.randint(0, low, (x0.shape[0],), device=device)
            n = torch.randn_like(x0)
            x_t = diffusion.q_sample(x0, t_probe, n)
            eps = diffusion.model(x_t, t_probe, cond["scheme"], cond["star"], cond["anchor"])
            alpha = diffusion.alphas_cumprod[t_probe]                     # (B,)
            x0_hat = ((x_t - (1 - alpha[:, None, None, None]).sqrt() * eps) / alpha[:, None, None, None].sqrt()).clamp(-15.0, 15.0)
            phys_b, terms_b = physics_loss(
                x0_hat,
                lam_vel=cfg["lam_vel"],
                lam_acc=cfg["lam_acc"],
                lam_col=cfg["lam_col"],
                lam_bound=cfg["lam_bound"],
                reduce="none",
            )
            loss = noise_loss + cfg["phys_weight"] * (alpha * phys_b).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            opt.step()
            ema_update()
            step += 1
            if step % cfg["log_every"] == 0:
                print(
                    f"ep {epoch} step {step} | noise {noise_loss.item():.4f} "
                    f"| phys {(alpha * phys_b).mean().item():.4f}"
                )

        # validation (fixed seed -> comparable across epochs)
        diffusion.eval()
        with torch.no_grad():
            torch.manual_seed(1000 + epoch)
            vl = 0.0
            for batch in val_dl:
                x0 = batch["traj"].to(device)
                cond = {k: batch[k].to(device) for k in ("scheme", "star", "anchor")}
                vl += diffusion.p_losses(x0, cond).item() * len(x0)
            vl /= max(1, len(val_ds))
        print(f"epoch {epoch} done in {time.time()-t0:.1f}s | val noise-MSE {vl:.5f}")
        torch.save(
            {"model": diffusion.state_dict(), "ema": ema.state_dict(), "cfg": cfg},
            ckpt_dir / "last.pt",
        )
        if vl < best_val:
            best_val = vl
            torch.save(
                {"model": diffusion.state_dict(), "ema": ema.state_dict(), "cfg": cfg},
                ckpt_dir / "best.pt",
            )
    print(f"training complete. best val {best_val:.5f}. checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
