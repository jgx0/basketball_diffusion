"""Conditional spatiotemporal diffusion model for multi-agent basketball trajectories.

Tensor contract
---------------
x          : (B, T, 11, 4)  trajectory tensor (x, y, vx, vy per agent)
conditioning: scheme (B,), star (B,), anchor (B, 11, 4)

The denoiser is a 1D temporal U-Net over the T axis. The agent/feature axes
(11 x 4 = 44 channels) are folded into the channel dimension, following the
sociology-of-motion / trajectory-diffusion literature. Scheme, star flag and
the anchor frame are injected as cross-attended tokens at the bottleneck.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import BALL_IDX, FEAT_DIM, N_AGENTS, SEQ_LEN


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding (standard DDPM formulation)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResidualBlock(nn.Module):
    """Conv1d residual block with timestep bias injection."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad)
        self.act = nn.SiLU()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(self.act(t_emb))[:, :, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class CrossAttention(nn.Module):
    """Tokens across time (queries) attend to conditioning tokens (keys/values)."""

    def __init__(self, channels: int, cond_channels: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(cond_channels, channels)
        self.to_v = nn.Linear(cond_channels, channels)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T') -> (B, T', C); ctx: (B, K, Cc)
        B, C, Tp = x.shape
        q = self.to_q(x.transpose(1, 2))
        k = self.to_k(ctx)
        v = self.to_v(ctx)
        q, k, v = (t.reshape(B, -1, self.heads, C // self.heads).transpose(1, 2) for t in (q, k, v))
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, Tp, C)
        out = self.proj(out)
        return (x.transpose(1, 2) + out).transpose(1, 2)  # residual, back to (B, C, T')


class ConditionEmbedder(nn.Module):
    """Encodes (scheme, star, anchor) into cross-attention tokens.

    Supports classifier-free-guidance dropout: with probability `drop` the
    labels and anchor are replaced by learned null tokens.
    """

    N_TOKENS = 6  # scheme, star, 4 anchor quadrant tokens

    def __init__(self, feat_dim: int = FEAT_DIM, n_agents: int = N_AGENTS, cond_dim: int = 256):
        super().__init__()
        self.scheme_emb = nn.Embedding(3, cond_dim)          # drop/switch/blitz
        self.star_emb = nn.Embedding(2, cond_dim)            # role/star handler
        self.null_scheme = nn.Parameter(torch.zeros(cond_dim))
        self.null_star = nn.Parameter(torch.zeros(cond_dim))
        anchor_in = n_agents * feat_dim
        self.anchor_proj = nn.Sequential(
            nn.Linear(anchor_in, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        # per-agent anchor embedding -> 4 quadrant tokens by mean-pooling groups
        self.agent_proj = nn.Sequential(
            nn.Linear(feat_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        self.null_anchor = nn.Parameter(torch.zeros(cond_dim))

    def forward(
        self,
        scheme: torch.Tensor,
        star: torch.Tensor,
        anchor: torch.Tensor,
        drop: float = 0.0,
        force_null: bool = False,
    ) -> torch.Tensor:
        """anchor: (B, 11, 4). Returns (B, N_TOKENS, cond_dim).

        force_null=True routes every input to the learned null tokens
        (used as the unconditional branch of classifier-free guidance).
        """
        B = anchor.shape[0]
        if force_null:
            drop_mask = torch.ones(B, dtype=torch.bool, device=anchor.device)
        else:
            drop_mask = torch.rand(B, device=anchor.device) < drop

        s = self.scheme_emb(scheme)
        k = self.star_emb(star)
        s = torch.where(drop_mask[:, None], self.null_scheme.expand(B, -1), s)
        k = torch.where(drop_mask[:, None], self.null_star.expand(B, -1), k)

        a_tokens = self.agent_proj(anchor)                    # (B, 11, cond_dim)
        groups = torch.stack(
            [
                a_tokens[:, 0:4].mean(1),   # offense core
                a_tokens[:, 4:5].mean(1),   # handler anchor
                a_tokens[:, 5:10].mean(1),  # defense
                a_tokens[:, BALL_IDX : BALL_IDX + 1].mean(1),  # ball
            ],
            dim=1,
        )                                                     # (B, 4, cond_dim)
        groups = torch.where(drop_mask[:, None, None], self.null_anchor.expand(B, 4, -1), groups)

        return torch.cat([s[:, None], k[:, None], groups], dim=1)  # (B, 6, cond_dim)


# ---------------------------------------------------------------------------
# Temporal U-Net denoiser
# ---------------------------------------------------------------------------
class TemporalUNet(nn.Module):
    """1D U-Net over T with cross-attention conditioning at the bottleneck."""

    def __init__(
        self,
        feat_dim: int = FEAT_DIM,
        n_agents: int = N_AGENTS,
        base: int = 128,
        time_dim: int = 256,
        cond_dim: int = 256,
    ):
        super().__init__()
        in_ch = n_agents * feat_dim
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.cond = ConditionEmbedder(feat_dim, n_agents, cond_dim)

        c1, c2, c3 = base, base * 2, base * 4
        self.stem = nn.Conv1d(in_ch, c1, 5, padding=2)

        self.down1 = ResidualBlock(c1, c1, time_dim)
        self.down1_pool = nn.Conv1d(c1, c1, 4, stride=2, padding=1)
        self.down2 = ResidualBlock(c1, c2, time_dim)
        self.down2_pool = nn.Conv1d(c2, c2, 4, stride=2, padding=1)
        self.mid1 = ResidualBlock(c2, c3, time_dim)
        self.attn = CrossAttention(c3, cond_dim, heads=4)
        self.mid2 = ResidualBlock(c3, c3, time_dim)

        self.mid_up = nn.ConvTranspose1d(c3, c3, 4, stride=2, padding=1)
        self.up2 = ResidualBlock(c3 + c2, c2, time_dim)
        self.up2_up = nn.ConvTranspose1d(c2, c2, 4, stride=2, padding=1)
        self.up1 = ResidualBlock(c2 + c1, c1, time_dim)
        self.out = nn.Conv1d(c1, in_ch, 5, padding=2)

    def forward(
        self,
        x_t: torch.Tensor,       # (B, T, 11, 4) noisy trajectory
        t: torch.Tensor,         # (B,) diffusion timestep
        scheme: torch.Tensor,    # (B,)
        star: torch.Tensor,      # (B,)
        anchor: torch.Tensor,    # (B, 11, 4)
        force_null_cond: bool = False,
        cond_drop: float = 0.0,
    ) -> torch.Tensor:
        B, T = x_t.shape[0], x_t.shape[1]
        ctx = self.cond(scheme, star, anchor, drop=cond_drop, force_null=force_null_cond)  # (B, 6, cond)
        temb = self.time_mlp(timestep_embedding(t, self.time_dim))  # (B, time_dim)

        h = x_t.reshape(B, T, -1).transpose(1, 2)                   # (B, 44, T)
        h = self.stem(h)

        h1 = self.down1(h, temb)                                    # (B, c1, T)
        h = self.down1_pool(h1)                                     # (B, c1, T/2)
        h2 = self.down2(h, temb)                                    # (B, c2, T/2)
        h = self.down2_pool(h2)                                     # (B, c2, T/4)

        h = self.mid1(h, temb)
        h = self.attn(h, ctx)
        h = self.mid2(h, temb)

        h = self.mid_up(h)                                          # (B, c3, T/2)
        h = self.up2(torch.cat([h, h2], dim=1), temb)               # (B, c2, T/2)
        h = self.up2_up(h)                                          # (B, c2, T)
        h = self.up1(torch.cat([h, h1], dim=1), temb)               # (B, c1, T)

        out = self.out(h)                                           # (B, 44, T)
        return out.transpose(1, 2).reshape(B, T, -1, x_t.shape[-1]) # (B, T, 11, 4)


# ---------------------------------------------------------------------------
# DDPM / DDIM wrapper
# ---------------------------------------------------------------------------
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule."""
    t = torch.linspace(0, 1, T + 1)
    alphas = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas = alphas / alphas[0]
    betas = 1 - (alphas[1:] / alphas[:-1])
    return betas.clamp(1e-4, 0.999)


class GaussianDiffusion(nn.Module):
    """Epsilon-prediction DDPM training + DDPM/DDIM sampling."""

    def __init__(self, model: TemporalUNet, n_steps: int = 1000, schedule: str = "cosine"):
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        if schedule == "cosine":
            betas = cosine_beta_schedule(n_steps)
        else:
            betas = torch.linspace(1e-4, 0.02, n_steps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.register_buffer("sqrt_ac", self.alphas_cumprod.sqrt())
        self.register_buffer("sqrt_om", (1 - self.alphas_cumprod).sqrt())

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward-process: corrupt x0 with noise at timestep t."""
        return (
            self.sqrt_ac[t][:, None, None, None] * x0
            + self.sqrt_om[t][:, None, None, None] * noise
        )

    def p_losses(self, x0: torch.Tensor, cond: dict, cond_drop: float = 0.0) -> torch.Tensor:
        """Standard noise-prediction MSE (physics losses are added externally)."""
        B = x0.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = self.model(
            x_t, t, cond["scheme"], cond["star"], cond["anchor"], cond_drop=cond_drop
        )
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def ddim_sample(
        self,
        cond: dict,
        seq_len: int = SEQ_LEN,
        n_samples: int | None = None,
        steps: int = 50,
        eta: float = 0.0,
        guidance: float = 1.0,
        device: str | torch.device = "cpu",
        anchor_blend_frames: int = 1,
        physics_guidance: float = 0.0,
        norm_stats: dict | None = None,
    ) -> torch.Tensor:
        """Sample trajectories; optionally classifier-free guidance.

        anchor_blend_frames: number of leading frames hard-set to the anchor
        trajectory so generated sequences start exactly at the observed state.
        physics_guidance: gradient-guidance strength pushing intermediate x0
        predictions toward physics compliance (0 disables; ~0.5 is a sane
        default; the gradient is computed under torch.enable_grad).
        norm_stats: optional {mean, std} (11, 4) arrays. Sampling runs in
        standardized space; outputs are de-standardized to meters before
        returning, and physics guidance is evaluated in physical units.
        """
        if n_samples is not None:
            for k in ("scheme", "star"):
                cond[k] = cond[k].expand(n_samples)
            cond["anchor"] = cond["anchor"].expand(n_samples, -1, -1)
        B = cond["scheme"].shape[0]
        use_cfg = guidance != 1.0

        def model_eps(x_t, t):
            if not use_cfg:
                return self.model(x_t, t, cond["scheme"], cond["star"], cond["anchor"])
            eps_c = self.model(x_t, t, cond["scheme"], cond["star"], cond["anchor"])
            eps_u = self.model(x_t, t, cond["scheme"], cond["star"], cond["anchor"], force_null_cond=True)
            return eps_u + guidance * (eps_c - eps_u)

        x = torch.randn(B, seq_len, cond["anchor"].shape[1], cond["anchor"].shape[2], device=device)
        ts = torch.linspace(self.n_steps - 1, 0, steps, device=device).long()
        acs = self.alphas_cumprod
        for i in range(len(ts)):
            t_cur, t_next = ts[i], ts[i + 1] if i + 1 < len(ts) else -1
            eps = model_eps(x, t_cur.expand(B))
            ac_t, ac_n = acs[t_cur], acs[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
            x0 = (x - (1 - ac_t).sqrt() * eps) / ac_t.sqrt()
            # Data-envelope clamp on the implied x0. In standardized space the
            # training data lives within ~|3| sigma, so a 4-sigma clamp kills
            # the ill-conditioned high-noise x0 estimates (error is amplified
            # by 1/sqrt(alpha-bar)) without touching legitimate structure.
            if norm_stats is not None:
                x0 = x0.clamp(-4.0, 4.0)
            else:
                # physical units: court coords x in [-10, 0], y in [-7.6, 7.6]
                x0 = x0.clamp(-15.0, 15.0)
            if physics_guidance > 0:
                x0 = self._physics_step(x0, physics_guidance, norm_stats)
            # DDIM update
            sigma = eta * ((1 - ac_n) / (1 - ac_t)).sqrt() * (1 - ac_t / ac_n).sqrt()
            dir_x0 = (1 - ac_n - sigma**2).sqrt() * eps
            x = ac_n.sqrt() * x0 + dir_x0
            if t_next < 0:
                x = x0

        if anchor_blend_frames > 0:
            # anchor: (B, 11, 4) -> (B, 1, 11, 4); broadcast over blended frames
            x[:, :anchor_blend_frames] = cond["anchor"][:, None, :, :].to(x.dtype)
        if norm_stats is not None:
            mean = torch.as_tensor(norm_stats["mean"], device=x.device, dtype=x.dtype)
            std = torch.as_tensor(norm_stats["std"], device=x.device, dtype=x.dtype)
            x = x * std + mean
        return x

    @staticmethod
    def _physics_step(x0: torch.Tensor, strength: float, norm_stats: dict | None = None) -> torch.Tensor:
        """One gradient-descent step on the physics loss of the x0 prediction.

        If norm_stats are provided, the physics functional is evaluated in
        physical units (meters, m/s) while the gradient flows through the
        affine map back into standardized space.
        """
        from src.loss import physics_loss

        with torch.enable_grad():
            x0_req = x0.detach().requires_grad_(True)
            x0_phys = x0_req
            if norm_stats is not None:
                mean = torch.as_tensor(norm_stats["mean"], device=x0.device, dtype=x0.dtype)
                std = torch.as_tensor(norm_stats["std"], device=x0.device, dtype=x0.dtype)
                x0_phys = x0_req * std + mean
            phys, _ = physics_loss(x0_phys, lam_acc=0.25)  # acc hinge is noisier
            (grad,) = torch.autograd.grad(phys, x0_req)
        return x0 - strength * grad
