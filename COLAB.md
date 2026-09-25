# Running the Full Pipeline on Google Colab

This guide walks through a complete research run — real SportVU ingestion,
diffusion training on GPU, counterfactual generation, evaluation, and paper
figures — entirely on a Google Colab session.

Two ways to run it:

| Option | Steps |
| --- | --- |
| **A. Import the notebook (recommended)** | Upload `colab_run.ipynb` to Colab → *Runtime → Change runtime type → GPU* → *Run all*. |
| **B. Manual cells** | Follow the cell-by-cell commands below. |

---

## 0. What you need

- A Google account (free tier works; a GPU runtime is strongly recommended).
- **GitHub repo access**: either your fork's URL, or upload a zip of this
  project. The notebook clones/pulls the repo so the code is identical to
  what is tested here.
- No API keys are required — the tracking data is public.

## 1. Runtime settings

1. `Runtime` → `Change runtime type` → **T4 GPU** (or A100 on Pro).
2. The notebook verifies `torch.cuda.is_available()` and falls back to CPU
   with a reduced-epoch warning if no GPU is attached.

### Colab Pro recommendations

| Setting | Free (T4) | Pro (A100) |
| --- | --- | --- |
| `N_GAMES` | 20 | **60** |
| `SYNTHETIC_SUPPLEMENT` | 1500 | 1000–1500 |
| `EPOCHS` | 60 | **100–150** |
| Real corpus share | ~15–20% | ~40–50% |
| Session length | ~3.5–5 h | ~3–4 h |

With `N_GAMES > 20`, notebook cell 0 auto-extends the game list from the
archive index (636 games available, Oct '15 – Jan '16) with a deterministic
season-spread sample — no manual list editing needed. Expect ~12–20 PnR
windows per game. At 60 games the corpus is roughly 700–1{,}200 real windows
against 1{,}000–1{,}500 synthetic, which shifts the mixture decisively toward
real data; that is the regime where the paper's realism tables are
meaningful.

## 2. What the run does

| Stage | Script | Typical time (T4) |
| --- | --- | --- |
| Download N SportVU games (7z, ~6 MB each) → extract | `scripts/pipeline.py` | 1–3 min |
| Parse → runs → rim frame → denoise → screens → labels → tensors | `scripts/pipeline.py` | ~1 min/game |
| Train conditional diffusion (GPU) | `scripts/train.py` | 15–60 min |
| Counterfactual generation + MP4s | `scripts/generate.py` | 5–15 min |
| FTD / ADE / compliance report | `scripts/evaluate.py` | 5–15 min |

## 3. Cell-by-cell (Option B)

```python
# --- Cell 1: clone the repo ---
import os
if not os.path.exists('basketball_diffusion'):
    # replace with your fork if you have one
    !git clone https://github.com/<your-org>/basketball_diffusion.git
%cd basketball_diffusion
```

```python
# --- Cell 2: deps (torch is preinstalled on Colab — do NOT pip install torch) ---
!pip install -q -r requirements-colab.txt
```

```python
# --- Cell 3: runtime check ---
import torch, sys
print('python', sys.version.split()[0], '| torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

```python
# --- Cell 4: real SportVU ingestion ---
# Downloads public game archives (2015-16 season mirrors) and extracts them.
!python scripts/pipeline.py --output-dir outputs --raw-dir data/raw --synthetic 0
# -> writes outputs/processed_tensors/{traj,scheme,star}.npy + norm_stats.npz
```

> `scripts/pipeline.py` auto-extracts `.7z` archives (via `py7zr`), parses the
> real moment schema `[quarter, epoch_ms, game_clock, shot_clock, null,
> positions]`, converts feet→meters, infers teams and the attacked hoop,
> reorders agents to `[offense 1–5, defense 1–5, ball]`, maps everything into
> the rim-centric frame, Savitzky-Golay denoises, detects ball screens, and
> labels coverages (drop/switch/blitz) geometrically.

```python
# --- Cell 5: (recommended) supplement with synthetic labels for balance ---
# Real heuristics under-count blitz; synthetic data balances the conditioning
# signal while real data anchors realism. The notebook mixes both caches.
```

```python
# --- Cell 6: train on GPU ---
!python scripts/train.py --config configs/base.yaml --device {DEVICE}
# checkpoints -> outputs/checkpoints/{last,best}.pt
```

```python
# --- Cell 7: counterfactual stress test + MP4 renders ---
!python scripts/generate.py --ckpt outputs/checkpoints/last.pt \
    --data-dir outputs/processed_tensors --render --device {DEVICE}
# -> outputs/counterfactuals/scheme_{drop,switch,blitz}.npy + .mp4
```

```python
# --- Cell 8: evaluation report ---
!python scripts/evaluate.py --ckpt outputs/checkpoints/last.pt \
    --data-dir outputs/processed_tensors --device {DEVICE}
# -> outputs/eval/report.json  (FTD, ADE, kinematic compliance, EPV contrasts)
```

```python
# --- Cell 9: unit tests (sanity) ---
!python -m pytest tests/ -q
```

```python
# --- Cell 10: download artifacts ---
from google.colab import files
files.download('outputs/eval/report.json')
# MP4s and .npy counterfactuals are in outputs/counterfactuals/
```

## 4. Scaling up the corpus

The pipeline is per-game linear; to use more of the season:

```python
import subprocess, pathlib
GAMES = [
    "01.01.2016.CHA.at.TOR",
    "01.01.2016.DAL.at.MIA",
    "01.01.2016.NYK.at.CHI",
    # ... see the repo listing at github.com/sealneaward/nba-movement-data
]
for g in GAMES:
    url = f"https://raw.githubusercontent.com/sealneaward/nba-movement-data/master/data/{g}.7z"
    subprocess.run(["curl", "-sL", "-o", f"data/raw/{g}.7z", url], check=True)
!python scripts/pipeline.py --output-dir outputs --raw-dir data/raw --synthetic 0
```

Rule of thumb: ~10–20 PnR windows per game with the strict screen detector.
A few hundred windows is enough for the framework to show label sensitivity;
paper-grade FTD wants thousands (or a synthetic pretrain → real finetune
recipe, which the notebook demonstrates).

## 5. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `py7zr` install fails | Colab pip cache | Restart runtime, rerun Cell 2. |
| `No SportVU files under data/raw` | download blocked | Check `ls data/raw`; rerun Cell 4. |
| `extracted 0 PnR windows` | detector found none | Loosen `detect_ball_screen` (`contact_dist` 1.2→1.4) in `src/normalize.py`. |
| OOM on GPU | batch 64 too big | Halve `batch_size` in `configs/base.yaml`. |
| Session dies mid-training | Colab idle timeout | Keep the tab open; checkpoints save every epoch (`last.pt`), so training resumes by rerunning Cell 6. |
| `torch` version conflict | torch was reinstalled | Never `pip install torch` on Colab; use `requirements-colab.txt` only. |

## 6. Getting results back for the paper

Run **cell 10.5** at the end of the notebook: it zips the paper-bound
artifacts and triggers your browser's download:

```
colab_artifacts.zip
├── outputs/eval/report.json              # FTD / ADE / compliance / EPV contrasts
├── outputs/counterfactuals/              # scheme_*.npy + scheme_*.mp4 renders
├── outputs/processed_tensors/norm_stats.npz
├── ingest_log.txt                        # funnel counts (events→runs→windows→labels)
└── train_log.txt                         # per-epoch loss curve
```

Unzip it **into the local project folder** (the `nba-proj` directory),
preserving the `outputs/...` structure:

```
nba-proj/
└── outputs/                 # <- merge this with the existing outputs/
    ├── eval/report.json
    ├── counterfactuals/
    └── ...
```

Then ask your agent to "update the paper with the Colab run" — the paper's
results tables (`paper/main.tex`, Tables 1–2) are regenerated from
`report.json` + the branch statistics, and the ingestion funnel numbers in
Appendix B come from `ingest_log.txt`.

## 7. Expected outputs

```
outputs/
├── processed_tensors/     # traj.npy (N,100,11,4), scheme.npy, star.npy, norm_stats.npz
├── checkpoints/           # last.pt, best.pt
├── counterfactuals/       # scheme_*.npy, scheme_*.mp4, handler_{role,star}.npy
└── eval/report.json       # FTD, ADE, compliance, counterfactual contrasts
```

`report.json` is the artifact the paper's experiments section quotes; the
counterfactual MP4s are the qualitative figures.
