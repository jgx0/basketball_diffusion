"""Unit tests for data, model, and physics modules (CPU-only, fast)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import A_MAX, V_MAX_PLAYER  # noqa: E402
from src.loss import (  # noqa: E402
    acceleration_penalty,
    court_bounds_penalty,
    epv_proxy,
    overlap_penalty,
    physics_loss,
    roller_openness,
    velocity_penalty,
)
from src.model import GaussianDiffusion, TemporalUNet  # noqa: E402
from src.normalize import (  # noqa: E402
    COURT_LENGTH,
    HOOP_X_FULL,
    HOOP_Y_FULL,
    detect_ball_screen,
    detect_pass,
    estimate_attack_direction,
    label_scheme,
    normalize_to_rim_frame,
    reorder_to_offense_defense,
)
from src.synthetic_pnr import build_dataset, simulate_pnr  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic simulator
# ---------------------------------------------------------------------------
def test_simulate_shapes_and_finite():
    rng = np.random.default_rng(0)
    out = simulate_pnr(rng, scheme="drop", star_handler=False, seq_len=100)
    traj = out["traj"]
    assert traj.shape == (100, 11, 4)
    assert np.isfinite(traj).all()
    # ball starts near handler
    d0 = np.linalg.norm(traj[0, 10, :2] - traj[0, 0, :2])
    assert d0 < 2.0


def test_schemes_diverge_statistically():
    rng = np.random.default_rng(1)
    trajs = []
    for scheme in ("drop", "switch", "blitz"):
        out = simulate_pnr(rng, scheme=scheme, seq_len=100)
        trajs.append(out["traj"])
    drop, blitz = trajs[0], trajs[2]
    # screener's defender (agent 6) mean depth differs between drop and blitz
    drop_big_depth = np.linalg.norm(drop[:, 6, :2], axis=-1).mean()
    blitz_big_depth = np.linalg.norm(blitz[:, 6, :2], axis=-1).mean()
    assert abs(drop_big_depth - blitz_big_depth) > 0.5


def test_build_dataset_balanced():
    data = build_dataset(12, seed=3, seq_len=40)
    assert data["traj"].shape == (12, 40, 11, 4)
    assert set(np.unique(data["scheme"])) == {0, 1, 2}
    assert set(np.unique(data["star"])) <= {0, 1}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def test_rim_frame_normalization_right_attack():
    T = 20
    xy = np.zeros((T, 11, 2), dtype=np.float32)
    xy[:, :, 0] = (COURT_LENGTH - HOOP_X_FULL) + np.linspace(0, 4, T)[:, None]
    xy[:, :, 1] = HOOP_Y_FULL + 7.0
    out = normalize_to_rim_frame(xy, attacking_right=True)
    assert abs(float(out[0, 0, 0])) < 1e-4          # rim at origin at t=0
    assert out[-1, 10, 0] > out[0, 10, 0]           # ball drifts away from rim
    assert abs(float(out[..., 1].max()) - 7.0) < 1e-4


def test_rim_frame_normalization_left_attack_mirrors():
    T = 20
    xy = np.zeros((T, 11, 2), dtype=np.float32)
    xy[:, :, 0] = HOOP_X_FULL - np.linspace(0, 4, T)[:, None]  # moving -x
    xy[:, :, 1] = HOOP_Y_FULL - 3.0
    out = normalize_to_rim_frame(xy, attacking_right=False)
    assert abs(float(out[0, 0, 0])) < 1e-4
    assert out[-1, 10, 0] > out[0, 10, 0]           # mirrored: still attacks +x'
    assert abs(float(out[..., 1].max()) - 3.0) < 1e-4


def test_reorder_offense_defense():
    T = 10
    xy = np.zeros((T, 11, 2), dtype=np.float32)
    # team A attacks right hoop (near it), team B away
    meta = np.array([[0, i] for i in range(5)] + [[1, 100 + i] for i in range(5)] + [[-1, -1]])
    xy[:, :5, 0] = COURT_LENGTH - HOOP_X_FULL - 5.0
    xy[:, 5:10, 0] = HOOP_X_FULL + 2.0
    out = reorder_to_offense_defense(xy, meta, attacking_right=True)
    # offense (team 0) should occupy slots 0..4
    assert np.allclose(out[:, 0:5, 0], COURT_LENGTH - HOOP_X_FULL - 5.0)


def test_detect_ball_screen_triggers():
    T = 60
    xy = np.zeros((T, 11, 2), dtype=np.float32)
    xy[:, 0, 0] = np.linspace(0, 6, T)   # handler moves away
    xy[:, 5, 0] = np.linspace(1.5, 6.0, T)  # on-ball defender chases
    xy[:, 1, :] = xy[:, 5, :]  # screener sits on the defender
    t = detect_ball_screen(xy)
    assert t is not None and t >= 0


def test_label_scheme_blitz_vs_drop():
    T = 40
    screen_t = 8
    xy = np.zeros((T, 11, 2), dtype=np.float32)
    xy[:, 0, 0] = -6.0                      # handler
    xy[:, 1, 0] = -7.0                      # screener
    # drop: big deep toward rim
    xy[:, 6, 0] = -2.0
    assert label_scheme(xy, screen_t) == 0
    # blitz: both defenders converge on handler
    xy2 = xy.copy()
    xy2[:, 6, 0] = -6.4
    xy2[:, 5, 0] = -5.7
    assert label_scheme(xy2, screen_t) == 2


# ---------------------------------------------------------------------------
# Physics losses
# ---------------------------------------------------------------------------
def _traj_constant_velocity(n=2, T=50, speed=1.0):
    xy = torch.zeros(n, T, 11, 2)
    xy[:, :, :, 0] = speed * 0.04 * torch.arange(T)[None, :, None]
    return torch.cat([xy, torch.zeros_like(xy)], dim=-1)  # add zero-v channels


def test_velocity_penalty_zero_below_cap():
    traj = _traj_constant_velocity(speed=1.0)  # 1 m/s << cap
    assert velocity_penalty(traj).item() < 1e-6


def test_velocity_penalty_positive_above_cap():
    traj = _traj_constant_velocity(speed=V_MAX_PLAYER + 2.0)
    assert velocity_penalty(traj).item() > 0.01


def test_acceleration_penalty():
    T = 60
    xy = torch.zeros(1, T, 11, 2)
    xy[:, :, 0, 0] = 0.5 * (A_MAX * 3) * 0.04 ** 2 * torch.arange(T) ** 2
    traj = torch.cat([xy, torch.zeros_like(xy)], dim=-1)
    assert acceleration_penalty(traj).item() > 0


def test_overlap_penalty_fires_on_coincident_players():
    traj = torch.zeros(1, 10, 11, 4)
    assert overlap_penalty(traj).item() > 0.1
    # deterministic spread: 10 players on a large circle -> ~0
    traj2 = torch.zeros(1, 10, 11, 4)
    ang = torch.linspace(0, 2 * torch.pi, 11)[:-1]
    traj2[:, :, :10, 0] = 5.0 * torch.cos(ang)[None, None]
    traj2[:, :, :10, 1] = 5.0 * torch.sin(ang)[None, None]
    assert overlap_penalty(traj2).item() < 1e-4


def test_bounds_and_epv_and_openness_run():
    traj = torch.randn(2, 30, 11, 4) * 2
    total, terms = physics_loss(traj)
    assert torch.isfinite(total)
    assert set(terms) == {"vel", "acc", "col", "bound"}
    assert epv_proxy(traj).shape == (2,)
    assert roller_openness(traj).shape == (2,)
    assert (roller_openness(traj) <= 1.0).all()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _cfg():
    return {"base_channels": 16, "time_dim": 32, "cond_dim": 32, "n_steps": 50}


def test_unet_output_shape():
    cfg = _cfg()
    unet = TemporalUNet(base=cfg["base_channels"], time_dim=cfg["time_dim"], cond_dim=cfg["cond_dim"])
    x = torch.randn(2, 40, 11, 4)
    t = torch.tensor([5, 10])
    scheme = torch.tensor([0, 1])
    star = torch.tensor([0, 1])
    anchor = torch.randn(2, 11, 4)
    out = unet(x, t, scheme, star, anchor)
    assert out.shape == x.shape


def test_diffusion_training_step_and_sampling():
    cfg = _cfg()
    unet = TemporalUNet(base=cfg["base_channels"], time_dim=cfg["time_dim"], cond_dim=cfg["cond_dim"])
    diff = GaussianDiffusion(unet, n_steps=cfg["n_steps"])
    x0 = torch.randn(2, 40, 11, 4) * 0.5
    cond = {
        "scheme": torch.tensor([0, 1]),
        "star": torch.tensor([0, 1]),
        "anchor": x0[:, 0].clone(),
    }
    loss = diff.p_losses(x0, cond)
    assert torch.isfinite(loss)
    gen = diff.ddim_sample({k: v[:1].clone() for k, v in cond.items()}, seq_len=40, steps=5, device="cpu")
    assert gen.shape == (1, 40, 11, 4)
    assert torch.isfinite(gen).all()
    # anchor blending pins frame 0 to the observed anchor
    assert torch.allclose(gen[:, 0], cond["anchor"][:1], atol=1e-5)


def test_cfg_guidance_changes_output():
    cfg = _cfg()
    unet = TemporalUNet(base=cfg["base_channels"], time_dim=cfg["time_dim"], cond_dim=cfg["cond_dim"])
    diff = GaussianDiffusion(unet, n_steps=cfg["n_steps"])
    torch.manual_seed(0)
    cond = {
        "scheme": torch.tensor([2]),
        "star": torch.tensor([1]),
        "anchor": torch.randn(1, 11, 4),
    }
    torch.manual_seed(7)
    g1 = diff.ddim_sample({k: v.clone() for k, v in cond.items()}, seq_len=20, steps=4, guidance=1.0, device="cpu")
    torch.manual_seed(7)
    g3 = diff.ddim_sample({k: v.clone() for k, v in cond.items()}, seq_len=20, steps=4, guidance=3.0, device="cpu")
    assert not torch.allclose(g1, g3)


def test_null_conditioning_path_runs():
    cfg = _cfg()
    unet = TemporalUNet(base=cfg["base_channels"], time_dim=cfg["time_dim"], cond_dim=cfg["cond_dim"])
    x = torch.randn(2, 20, 11, 4)
    t = torch.tensor([1, 2])
    scheme = torch.tensor([0, 1])
    star = torch.tensor([0, 1])
    anchor = torch.randn(2, 11, 4)
    out = unet(x, t, scheme, star, anchor, force_null_cond=True)
    assert out.shape == x.shape


def test_detect_pass_fires_on_ball_release():
    T = 30
    xy = np.zeros((T, 11, 2), dtype=np.float32)
    xy[:, 10, 0] = np.concatenate([np.zeros(10), np.linspace(4, 12, 20)])
    t = detect_pass(xy)
    assert t is not None and 8 <= t <= 12


# ---------------------------------------------------------------------------
# Branch-study statistics
# ---------------------------------------------------------------------------
def _study_with_effect(A=30, M=40, effect=0.3, noise=0.25, seed=0):
    from src.branch_study import BranchStudy

    rng = np.random.default_rng(seed)
    anchor_shift = rng.normal(0, 0.4, size=(A, 1, 1))  # anchor-level heterogeneity
    phi_b0 = 0.5 + anchor_shift + rng.normal(0, noise, size=(A, 1, M))
    phi_b1 = 0.5 + anchor_shift + effect + rng.normal(0, noise, size=(A, 1, M))
    return BranchStudy(phi=np.concatenate([phi_b0, phi_b1], axis=1),
                       branch_names=["ref", "treat"])


def test_branch_study_detects_planted_effect():
    study = _study_with_effect(effect=0.3)
    rep = study.summary(n_boot=1000, seed=0)
    pair = rep["bootstrap"]["pairs"][0]
    assert abs(pair["delta"] - 0.3) < 0.08
    assert pair["p_permutation"] < 0.05
    pt = rep["paired_tests"][0]
    assert pt["p_ttest"] < 0.01  # paired t-test has real power here


def test_branch_study_flags_no_effect_when_null():
    study = _study_with_effect(effect=0.0)
    rep = study.summary(n_boot=1000, seed=1)
    pair = rep["bootstrap"]["pairs"][0]
    assert pair["p_permutation"] > 0.05
    assert abs(pair["delta"]) < 0.1


def test_branch_study_anchor_paired_beats_unpaired_noise():
    # anchor heterogeneity is large; the paired contrast must cancel it, so
    # the paired p-value is far more significant than an unpaired z would be
    study = _study_with_effect(A=30, M=40, effect=0.15, noise=0.25, seed=3)
    rep = study.summary(n_boot=1000, seed=0)
    pair = rep["bootstrap"]["pairs"][0]
    assert pair["p_permutation"] < 0.2  # paired: anchored variance cancels
    # marginal (unpaired) means differ by much less precisely
    bm = rep["branch_means"]
    assert abs(bm["treat"] - bm["ref"]) > 0.05


def test_required_m_monotone_and_plausible():
    # low-noise config: SE target is already met at the current M
    study = _study_with_effect(A=30, M=40, effect=0.3, noise=0.25)
    r = study.required_m(target_se=0.02, n_boot=200, seed=1)[0]
    assert r["se_at_current_m"] < r["se_at_half_m"]          # SE shrinks with M
    assert r["target_met_at_current_m"] is True
    assert r["required_m_for_se"]["0.02"] == 40


def test_required_m_extrapolates_when_target_unmet():
    # heavy noise: SE at M=40 exceeds the target, so required M must exceed 40
    # and must match the sqrt-scaling law: M_req = M * (SE/target)^2
    study = _study_with_effect(A=8, M=40, effect=0.3, noise=0.8)
    r = study.required_m(target_se=0.02, n_boot=200, seed=1)[0]
    assert r["target_met_at_current_m"] is False
    m_need = r["required_m_for_se"]["0.02"]
    expected = int(np.ceil(40 * (r["se_at_current_m"] / 0.02) ** 2))
    assert m_need == expected > 40


def test_branch_study_shape_validation():
    import pytest
    from src.branch_study import BranchStudy

    with pytest.raises(ValueError):
        BranchStudy(phi=np.zeros((5, 3)))


# ---------------------------------------------------------------------------
# SportVU archive extraction
# ---------------------------------------------------------------------------
def test_ensure_extracted_multi_archive(tmp_path):
    """Regression: each .7z must extract to its own subdirectory.

    The earlier skip logic treated ANY existing *.json in the raw dir as
    'already extracted', so in multi-game batches only the first archive was
    ever unpacked (observed on real Colab runs).
    """
    import zipfile

    py7zr = pytest.importorskip("py7zr")
    from src.sportvu import ensure_extracted

    # build two minimal valid .7z archives, each containing a game JSON
    games = {}
    for i, name in enumerate(("01.01.2016.A.at.B", "01.02.2016.C.at.D")):
        payload = json.dumps(
            {"gameid": f"002150000{i}", "gamedate": "2016-01-0%d" % (i + 1), "events": []}
        ).encode()
        inner = tmp_path / f"{name}.json"
        inner.write_bytes(payload)
        arc = tmp_path / f"{name}.7z"
        with py7zr.SevenZipFile(arc, "w") as z:
            z.write(inner, arcname=f"{name}.json")
        inner.unlink()
        games[name] = payload

    files = ensure_extracted(tmp_path)
    stems = {f.stem for f in files}
    assert stems == set(games), f"expected both games extracted, got {stems}"
    # per-archive isolation: each JSON lives under its own subdirectory
    for name in games:
        assert (tmp_path / name / f"{name}.json").exists()
    # idempotent re-run: no crash, same file set
    again = ensure_extracted(tmp_path)
    assert {f.stem for f in again} == stems
    assert not (tmp_path / "01.01.2016.A.at.B.json").exists()  # never flattened


def test_find_sportvu_files_dedupes_copies(tmp_path):
    """Loose JSON + identical-size copy in a subdir -> one entry (shallowest)."""
    from src.sportvu import find_sportvu_files

    payload = b'{"gameid": "0021500492", "events": []}'
    loose = tmp_path / "game.json"
    loose.write_bytes(payload)
    copy = tmp_path / "game-dir" / "game.json"
    copy.parent.mkdir()
    copy.write_bytes(payload)
    files = find_sportvu_files(tmp_path)
    assert len(files) == 1 and files[0] == loose
