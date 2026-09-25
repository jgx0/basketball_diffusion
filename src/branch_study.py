"""Statistics for the paired counterfactual branch study.

Design (fixed before looking at outcomes, matching the paper's protocol):

- **Anchors**: A distinct validation opening states. For each anchor a and
  branch z (e.g. scheme in {drop, switch, blitz}) we sample M futures
  conditioned on (anchor=a, branch=z). This is a PAIRED design: branch
  contrasts are computed within-anchor, so anchor-to-anchor variation ---
  the dominant variance source --- cancels in the contrast.

- **Estimand**: for vulnerability functional phi (e.g. roller openness),
  Delta_z,z' = mean_a [ mean_m phi(x_{a,m,z}) - mean_m phi(x_{a,m,z'}) ].

- **Inference**: a paired cluster bootstrap over anchors (resample anchors
  with replacement, B times) yields CIs and p-values for each contrast.
  Permutation of branch labels *within anchor* (exact under the null that
  branch labels are exchangeable within anchor) provides a second,
  assumption-light p-value.

- **Power**: from the observed within-anchor contrast variance we estimate
  the M required to resolve an effect of a given size at target power,
  via the paired design's SE = sqrt(sigma_d^2 / (A * M_eff)) where sigma_d^2
  is the variance of the per-anchor contrast and the bootstrap already
  accounts for A. We report required M for SE targets 0.01 / 0.02 / 0.05.

All estimators operate on per-sample per-anchor sufficient statistics, not
raw trajectories, so the whole study summary is small and portable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class BranchStudy:
    """Sufficient statistics for a paired branch study.

    phi[a, z, m] = value of the vulnerability functional on future m of
    branch z from anchor a. Shape (A, Z, M); NaN marks a dropped sample
    (e.g. failed generation).
    """

    phi: np.ndarray                      # (A, Z, M)
    branch_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.phi = np.asarray(self.phi, dtype=np.float64)
        if self.phi.ndim != 3:
            raise ValueError(f"phi must be (A, Z, M); got {self.phi.shape}")
        if not self.branch_names:
            self.branch_names = [f"b{i}" for i in range(self.phi.shape[1])]

    @property
    def n_anchors(self) -> int:
        return self.phi.shape[0]

    @property
    def n_branches(self) -> int:
        return self.phi.shape[1]

    @property
    def m_per_branch(self) -> int:
        return self.phi.shape[2]

    def branch_means(self) -> np.ndarray:
        """Marginal mean per branch, averaging anchors then draws: (Z,)."""
        with np.errstate(invalid="ignore"):
            per_anchor = np.nanmean(self.phi, axis=2)      # (A, Z)
        return np.nanmean(per_anchor, axis=0)              # (Z,)

    def paired_contrasts(self) -> tuple[np.ndarray, np.ndarray]:
        """Within-anchor branch contrasts.

        Returns (deltas, per_anchor_deltas):
        deltas[k] = mean over anchors of (branch k vs branch 0) contrast,
        per_anchor_deltas[k] = the A per-anchor contrasts for pair k.
        Pairs are (z, 0) for z = 1..Z-1 (branch 0 is the reference).
        """
        with np.errstate(invalid="ignore"):
            pa = np.nanmean(self.phi, axis=2)              # (A, Z) anchor means
        ref = pa[:, [0]]
        d = pa[:, 1:] - ref                                # (A, Z-1)
        return np.nanmean(d, axis=0), d

    # ------------------------------------------------------------------
    # Paired cluster bootstrap over anchors
    # ------------------------------------------------------------------
    def bootstrap(
        self,
        n_boot: int = 2000,
        seed: int = 0,
        ci: float = 0.95,
    ) -> dict:
        """Bootstrap CIs and two-sided p-values for each (z, 0) contrast.

        Resamples anchors with replacement (the cluster), recomputes the
        within-anchor contrast mean each time. p-value: bootstrap SE-based
        z-test, cross-checked against a within-anchor sign-flip permutation.
        """
        rng = np.random.default_rng(seed)
        deltas, per_anchor = self.paired_contrasts()       # (Z-1,), (A, Z-1)
        A = per_anchor.shape[0]
        idx = rng.integers(0, A, size=(n_boot, A))
        boot = per_anchor[idx].mean(axis=1)                # (n_boot, Z-1)
        alpha = (1.0 - ci) / 2.0
        lo, hi = np.quantile(boot, [alpha, 1 - alpha], axis=0)

        # sign-flip permutation within anchor (exact under exchangeability)
        n_perm = 2000
        signs = rng.choice([-1.0, 1.0], size=(n_perm, A, 1))
        perm_means = (per_anchor[None] * signs).mean(axis=1)   # (n_perm, Z-1)
        p_perm = (np.abs(perm_means) >= np.abs(deltas)[None, :]).mean(axis=0)

        se = boot.std(axis=0, ddof=1)
        z_stat = np.divide(deltas, se, out=np.zeros_like(deltas), where=se > 0)
        p_z = 2.0 * stats.norm.sf(np.abs(z_stat))
        return {
            "pairs": [
                {
                    "branch": self.branch_names[k + 1],
                    "reference": self.branch_names[0],
                    "delta": float(deltas[k]),
                    "ci_lo": float(lo[k]),
                    "ci_hi": float(hi[k]),
                    "boot_se": float(se[k]),
                    "p_bootstrap": float(p_z[k]),
                    "p_permutation": float(p_perm[k]),
                }
                for k in range(len(deltas))
            ],
            "n_boot": n_boot,
            "n_perm": n_perm,
            "anchors": int(A),
        }

    # ------------------------------------------------------------------
    # Frequentist cross-check: paired t / Wilcoxon over anchors
    # ------------------------------------------------------------------
    def paired_tests(self) -> list[dict]:
        _, per_anchor = self.paired_contrasts()
        out = []
        for k in range(per_anchor.shape[1]):
            d = per_anchor[:, k]
            d = d[np.isfinite(d)]
            t = stats.ttest_1samp(d, 0.0)
            try:
                w = stats.wilcoxon(d)
                p_w = float(w.pvalue)
            except ValueError:      # all differences zero
                p_w = 1.0
            out.append(
                {
                    "branch": self.branch_names[k + 1],
                    "reference": self.branch_names[0],
                    "n_anchors_used": int(len(d)),
                    "t_stat": float(t.statistic),
                    "p_ttest": float(t.pvalue),
                    "p_wilcoxon": p_w,
                }
            )
        return out

    # ------------------------------------------------------------------
    # Power: required draws per branch
    # ------------------------------------------------------------------
    def required_m(
        self,
        target_se: float = 0.02,
        n_boot: int = 500,
        seed: int = 1,
    ) -> dict:
        """Estimate M per branch needed to push the contrast SE to `target_se`.

        Uses the standard sqrt-scaling of independent-draw averages,
        SE(M) = SE(M_obs) * sqrt(M_obs / M), which is exact when draws are
        independent within anchor and conservative when they are positively
        correlated (e.g. shared-anchor sampling correlation). The half-M
        subsample SE is reported as a diagnostic of the realized scaling but
        is NOT used for extrapolation: a two-point power-law fit is far too
        noisy to be trusted (empirically it can even yield nonsense like
        'M=2 suffices' when both measured SEs sit near the target).
        """
        rng = np.random.default_rng(seed)
        A, Z, M = self.phi.shape
        out = []
        for k in range(Z - 1):
            mes = {}
            for m in sorted({max(2, M // 2), M}):
                subs = self.phi[:, k + 1, :m] - self.phi[:, 0, :m]
                pa = np.nanmean(subs, axis=1)                   # (A,)
                idx = rng.integers(0, A, size=(n_boot, A))
                se_m = pa[idx].mean(axis=1).std(ddof=1)
                mes[m] = float(se_m)
            s_now = mes[M]
            if s_now <= target_se:
                m_need, met = M, True
            else:
                m_need = int(np.ceil(M * (s_now / target_se) ** 2))
                met = False
            out.append(
                {
                    "branch": self.branch_names[k + 1],
                    "reference": self.branch_names[0],
                    "se_at_current_m": s_now,
                    "se_at_half_m": mes[max(2, M // 2)],
                    "target_se": float(target_se),
                    "target_met_at_current_m": bool(met),
                    "required_m_for_se": {str(target_se): m_need},
                    "scaling_assumption": "SE ~ M^-1/2 (independent draws)",
                }
            )
        return out

    def _per_anchor_contrast(self, k: int) -> np.ndarray:
        subs = self.phi[:, k + 1] - self.phi[:, 0]              # (A, M)
        return np.nanmean(subs, axis=1)

    # ------------------------------------------------------------------
    def summary(self, n_boot: int = 2000, seed: int = 0) -> dict:
        return {
            "design": {
                "anchors": int(self.n_anchors),
                "branches": self.branch_names,
                "m_per_branch": int(self.m_per_branch),
                "reference_branch": self.branch_names[0],
                "estimand": "within-anchor paired contrast of branch means",
            },
            "branch_means": {
                name: float(v) for name, v in zip(self.branch_names, self.branch_means())
            },
            "bootstrap": self.bootstrap(n_boot=n_boot, seed=seed),
            "paired_tests": self.paired_tests(),
        }
