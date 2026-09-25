"""PyTorch dataset over preprocessed (T, 11, 4) possession tensors.

Each sample carries:
    traj    : float32 (T, 11, 4)  — x, y, vx, vy for 10 players + ball
    scheme  : int64               — 0 drop / 1 switch / 2 blitz
    star    : int64               — 0 role handler / 1 star handler
    anchor  : float32 (11, 4)     — frame-0 snapshot used as conditioning
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _cache_paths(cache_dir: Path) -> tuple[Path, Path, Path]:
    return (
        cache_dir / "traj.npy",
        cache_dir / "scheme.npy",
        cache_dir / "star.npy",
    )


def save_cache(cache_dir: str | Path, data: dict) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "traj.npy", data["traj"])
    np.save(cache_dir / "scheme.npy", data["scheme"])
    np.save(cache_dir / "star.npy", data["star"])


def load_cache(cache_dir: str | Path) -> dict:
    t, s, k = _cache_paths(Path(cache_dir))
    if not t.exists():
        raise FileNotFoundError(
            f"No tensor cache at {cache_dir}. Run scripts/pipeline.py --stage data first."
        )
    return {"traj": np.load(t), "scheme": np.load(s), "star": np.load(k)}


def save_norm_stats(cache_dir: str | Path, traj: np.ndarray) -> None:
    """Save per-feature mean/std over the whole cache (traj: N,T,11,4)."""
    cache_dir = Path(cache_dir)
    np.savez(cache_dir / "norm_stats.npz", mean=traj.mean((0, 1)), std=traj.std((0, 1)) + 1e-6)


def load_norm_stats(cache_dir: str | Path) -> dict:
    p = Path(cache_dir) / "norm_stats.npz"
    if not p.exists():
        return {}
    z = np.load(p)
    return {"mean": z["mean"], "std": z["std"]}


class PnRTrajectoryDataset(Dataset):
    """Tensors from the pipeline cache; optionally splits train/val by index parity."""

    def __init__(self, cache_dir: str | Path, split: str = "train", val_frac: float = 0.1, seed: int = 0):
        data = load_cache(cache_dir)
        # Feature standardization (per-feature over N,T): trajectories live in a
        # narrow court window (x ~ [-10, 0]); zero-mean unit-variance inputs make
        # the diffusion objective far better conditioned.
        stats = load_norm_stats(cache_dir)
        if stats:
            data = {
                **data,
                "traj": (data["traj"] - stats["mean"]) / stats["std"],
            }
        self.stats = stats
        n = len(data["traj"])
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = max(1, int(round(n * val_frac)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        self.index = train_idx if split == "train" else val_idx
        self.traj = torch.from_numpy(data["traj"].astype(np.float32))  # (n, T, 11, 4)
        self.scheme = torch.from_numpy(data["scheme"])      # (n,)
        self.star = torch.from_numpy(data["star"])          # (n,)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        j = int(self.index[i])
        return {
            "traj": self.traj[j],            # (T, 11, 4)
            "scheme": self.scheme[j],        # scalar
            "star": self.star[j],            # scalar
            "anchor": self.traj[j][0].clone()  # (11, 4)
        }


def collate(batch: list[dict]) -> dict:
    out = {}
    for k in ("traj", "scheme", "star", "anchor"):
        out[k] = torch.stack([b[k] for b in batch])
    return out
