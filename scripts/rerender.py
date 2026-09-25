"""Re-render counterfactual MP4s from saved artifact arrays.

Use this after pulling a colab_artifacts folder: the .npy trajectory arrays
are the source of truth; videos can always be regenerated locally with the
current renderer.

Usage:
    python scripts/rerender.py --artifacts colab_artifacts-3
    python scripts/rerender.py --traj outputs_real/processed_tensors/traj.npy --index 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualize import render_gif  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=None, help="colab_artifacts dir with outputs/counterfactuals/*.npy")
    ap.add_argument("--traj", default=None, help="single (T, 11, 4) trajectory .npy (e.g. a real window)")
    ap.add_argument("--index", type=int, default=0, help="sample index when using --artifacts")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.traj:
        traj = np.load(args.traj)
        if traj.ndim == 4:  # (N, T, 11, 4) -> single sample
            traj = traj[args.index]
        out = args.out or str(Path(args.traj).with_suffix(".mp4"))
        print(render_gif(traj, out))
        return

    if not args.artifacts:
        raise SystemExit("Pass --artifacts DIR or --traj FILE")

    cf_dir = Path(args.artifacts) / "outputs" / "counterfactuals"
    for npy in sorted(cf_dir.glob("scheme_*.npy")):
        gen = np.load(npy)
        out = args.out or str(npy.with_suffix(".mp4"))
        print(render_gif(gen[args.index], out))


if __name__ == "__main__":
    main()
