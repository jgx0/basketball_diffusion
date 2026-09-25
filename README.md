# Counterfactual Spatiotemporal Diffusion for Basketball

Stress-testing defensive schemes via multi-agent trajectory generation.

This repository implements a **conditional score-based diffusion model** over
half-court pick-and-roll (PnR) tracking tensors `(T=100, 11 agents, 4 features)`
with physics-informed losses, plus a counterfactual evaluation framework
(defensive-scheme mutation, handler "star swap", vulnerability metrics).

> **Data note.** The canonical pipeline expects legacy NBA SportVU tracking JSONs
> (2015-16 public archives, e.g. `nba-movement-data` mirrors) in `data/raw/` —
> `.7z` archives are extracted automatically. The ingestion path is validated
> end-to-end on a real downloaded game (see `paper/main.tex` §Ingestion
> validation). The repo also runs **end-to-end on synthetic data** when no
> SportVU files are found: `src/synthetic_pnr.py` encodes hand-written
> drop/switch/blitz tactics and doubles as the interpretable baseline.
>
> **Google Colab:** see `COLAB.md` and `colab_run.ipynb` for a full GPU run —
> archive download, real-data ingestion, training, counterfactuals, evaluation.

## Repository layout

```
basketball_diffusion/
├── configs/base.yaml        # training hyperparameters (no PyYAML needed)
├── data/raw/                # put SportVU *.json[.gz] or *.7z here (git-ignored)
├── colab_run.ipynb          # importable Colab notebook (full GPU run)
├── COLAB.md                 # Colab guide & troubleshooting
├── src/
│   ├── constants.py         # court geometry, agent layout, physics caps
│   ├── sportvu.py           # raw JSON parsing -> aligned arrays
│   ├── normalize.py         # rim-centric frame, screen detection, tensorization
│   ├── synthetic_pnr.py     # behavioral PnR simulator (data fallback / baseline)
│   ├── dataset.py           # PyTorch Dataset over the tensor cache
│   ├── model.py             # Temporal U-Net + DDPM/DDIM + CFG
│   ├── loss.py              # physics penalties + vulnerability metrics
│   ├── config.py            # flat-YAML config loader
│   └── visualize.py         # half-court MP4 rendering
├── scripts/
│   ├── pipeline.py          # build tensor cache (SportVU or synthetic)
│   ├── train.py             # train conditional diffusion (EMA, physics loss)
│   ├── generate.py          # counterfactual sampling + optional MP4s
│   └── evaluate.py          # FTD / ADE / compliance / vulnerability report
├── tests/test_all.py        # unit tests (CPU, ~1 min)
├── paper/                   # LaTeX manuscript
├── requirements.txt
└── Makefile
```

## Quickstart

```bash
make setup                 # venv + deps
source .venv/bin/activate

make data                  # builds outputs/processed_tensors (synthetic if no SportVU)
make train                 # writes outputs/checkpoints/{last,best}.pt
make evaluate              # writes outputs/eval/report.json
python scripts/generate.py --render   # counterfactual MP4s into outputs/counterfactuals/
pytest -q                  # unit tests
```

## Method summary

- **Tensor**: `X ∈ R^{T×11×4}` — x, y, vx, vy for 10 players + ball, rim-centric frame.
- **Denoiser**: 1D temporal U-Net (agents×features folded into channels) with
  cross-attention conditioning on `(scheme, star, anchor frame)`.
- **Diffusion**: ε-prediction DDPM with cosine schedule; DDIM sampling;
  classifier-free guidance via learned null tokens (`--guidance`).
- **Physics losses**: hinge penalties on velocity > 7.5 m/s, acceleration
  > 5 m/s², player overlap (< 0.6 m), and out-of-bounds drift.
- **Counterfactuals**: fix the anchor (frame-0 state), mutate `scheme` or
  `star`, sample K futures, and compare roller openness / paint pressure /
  EPV proxy distributions.

## Evaluation metrics

| Metric | Meaning |
| --- | --- |
| FTD | Fréchet distance on possession-level summary features (realism) |
| ADE | Average displacement error vs. condition-matched real futures (m) |
| Compliance | Rates of velocity/acceleration/overlap violations |
| Roller openness | Fraction of late frames with no defender within 2 m of the roller |
| EPV proxy | Spatial expected-value placeholder (rim proximity × separation) |

## Honest limitations

- SportVU scheme/star labels default to 0 unless you supply PBP/roster metadata;
  the synthetic simulator is the primary source of *labeled* counterfactuals.
- The EPV proxy is a hand-crafted placeholder, not a learned EPV/chance model.
- Sampling 100-frame, 11-agent futures with a diffusion model is compute-hungry;
  DDIM (50 steps) is the default compromise.
