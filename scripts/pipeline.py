"""Data pipeline: build the (N, T, 11, 4) tensor cache for training.

Modes:
  1. If data/raw contains SportVU game files (*.json or *.json.gz), they are
     parsed, rim-frame normalized, PnR windows are cut around detected ball
     screens, and windows are tensorized.
  2. Otherwise, falls back to the synthetic behavioral simulator so the repo
     is runnable end-to-end without proprietary data.

Usage:
    python scripts/pipeline.py --output-dir outputs [--synthetic N] [--seq-len 100]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import save_cache, save_norm_stats  # noqa: E402
from src.normalize import (  # noqa: E402
    COURT_LENGTH,
    HOOP_X_FULL,
    detect_ball_screen,
    estimate_attack_direction,
    label_scheme,
    normalize_to_rim_frame,
    reorder_to_offense_defense,
    tensorize_window,
    denoise_run,
)
from src.sportvu import (  # noqa: E402
    ensure_extracted,
    event_to_arrays,
    find_sportvu_files,
    load_game,
    split_contiguous_runs,
)
from src.synthetic_pnr import build_dataset  # noqa: E402
from src.constants import SEQ_LEN  # noqa: E402


def process_sportvu(
    data_dir: Path,
    seq_len: int,
    max_windows_per_game: int = 200,
) -> dict:
    """Extract half-court PnR windows from real SportVU game files.

    Per contiguous tracking run: infer teams & attacking hoop, reorder agents
    to [offense 1-5, defense 1-5, ball], map to the rim-centric frame, detect
    ball screens, and label the coverage with a geometry heuristic.
    """
    files = ensure_extracted(data_dir)
    if not files:
        raise FileNotFoundError(f"No SportVU files under {data_dir}")

    trajs: list[np.ndarray] = []
    schemes = np.zeros(0, dtype=np.int64)
    stars = np.zeros(0, dtype=np.int64)

    for f in files:
        print(f"  parsing {f.name} ...")
        gt = load_game(f)
        n_from_game = 0
        for eid in gt.event_ids:
            if n_from_game >= max_windows_per_game:
                break
            arrs = event_to_arrays(gt.frames[eid])
            if not arrs or len(arrs["xy"]) < 50:
                continue
            for run in split_contiguous_runs(arrs):
                if len(run["xy"]) < seq_len + 5:
                    continue
                xy = run["xy"]
                right = estimate_attack_direction(xy)
                # half-court only: ball must stay deep in one half
                hoop_x = (COURT_LENGTH - HOOP_X_FULL) if right else HOOP_X_FULL
                ball_depth = (hoop_x - xy[:, 10, 0]) if right else (xy[:, 10, 0] - hoop_x)
                if ball_depth.max() > COURT_LENGTH / 2 * 0.75:
                    continue
                reordered = reorder_to_offense_defense(xy, run["entity_meta"], right)
                rim = normalize_to_rim_frame(reordered, right)
                rim = denoise_run(rim)
                if rim is None:
                    continue
                screen_t = detect_ball_screen(rim)
                if screen_t is None or screen_t < 12:
                    continue
                end = min(screen_t + seq_len, len(rim))
                start = end - seq_len
                win = rim[start:end]
                scheme = label_scheme(win, screen_t - start)
                schemes = np.append(schemes, scheme)
                stars = np.append(stars, 0)  # star flag needs roster metadata
                trajs.append(tensorize_window(win))
                n_from_game += 1
                if n_from_game >= max_windows_per_game:
                    break

    if not trajs:
        raise RuntimeError("SportVU files found but no PnR windows extracted.")
    print(
        f"  extracted {len(trajs)} PnR windows; scheme counts: "
        f"drop={int((schemes == 0).sum())} switch={int((schemes == 1).sum())} blitz={int((schemes == 2).sum())}"
    )
    return {"traj": np.stack(trajs), "scheme": schemes, "star": stars}

    if not trajs:
        raise RuntimeError("SportVU files found but no PnR windows extracted.")
    return {"traj": np.stack(trajs), "scheme": schemes, "star": stars}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--synthetic", type=int, default=2000, help="synthetic samples if no SportVU found")
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = ap.parse_args()

    cache_dir = Path(args.output_dir) / "processed_tensors"
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(args.raw_dir)
    has_raw = raw_dir.exists() and (find_sportvu_files(raw_dir) or list(raw_dir.rglob("*.7z")))
    if not has_raw and args.synthetic <= 0:
        raise SystemExit(
            f"No SportVU files or .7z archives under {raw_dir} and --synthetic <= 0. "
            "Download game archives (see COLAB.md) or pass --synthetic N."
        )
    if has_raw:
        print(f"SportVU files found in {raw_dir}; processing...")
        data = process_sportvu(raw_dir, args.seq_len)
        source = "sportvu"
    else:
        print(f"No SportVU files in {raw_dir}; generating {args.synthetic} synthetic PnR possessions...")
        data = build_dataset(args.synthetic, seed=0, seq_len=args.seq_len)
        source = "synthetic"

    save_cache(cache_dir, data)
    save_norm_stats(cache_dir, data["traj"])
    n = len(data["traj"])
    print(f"saved {n} samples ({source}) to {cache_dir}")
    print(f"traj tensor: {data['traj'].shape} {data['traj'].dtype}")


if __name__ == "__main__":
    main()
