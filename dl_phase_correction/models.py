#!/usr/bin/env python3
"""
Phase 4 — Dual-head physics-informed U-Net for phase aberration correction.

Input:  phi_total  [B, 1, H, W]  wrapped phase [-π, π]
Outputs:
  phi_sample  [B, 1, H, W]  estimated unwrapped sample phase
  aber_coeffs [B, N_MODES]  estimated Zernike aberration coefficients

Architecture:
  Shared U-Net encoder  (4 levels + bottleneck)
  ├─ Spatial decoder    (4 upsampling levels + skip connections) → phi_sample
  └─ Zernike MLP head   (global-avg-pool → 2-layer MLP)         → aber_coeffs

The Zernike basis is stored as a non-trainable buffer so the reconstructed
aberration surface  phi_aber = Σ_k c_k Z_k  is differentiable inside the
cycle-consistency loss in train_supervised.py.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from zernike_utils import build_zernike_basis, basis_to_tensor, N_MODES

H = W = 512   # extracted field size — must match simulate_phase_data.py


# ── Building blocks ───────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two Conv-BN-ReLU layers (standard U-Net double conv)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Bilinear ×2 upsample → concat encoder skip → ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([self.up(x), skip], dim=1))


# ── Main model ────────────────────────────────────────────────────────────────

class PhaseAberrationNet(nn.Module):
    """
    Dual-head physics-informed U-Net.

    Parameters
    ----------
    n_modes : number of Zernike modes (default N_MODES = 15)
    base_ch : base channel width; encoder uses base × {1,2,4,8,16}
    """

    def __init__(self, n_modes: int = N_MODES, base_ch: int = 64):
        super().__init__()
        self.n_modes = n_modes
        c = base_ch

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc1 = ConvBlock(1,     c)      # [B,  64, 512, 512]
        self.enc2 = ConvBlock(c,   2*c)      # [B, 128, 256, 256]
        self.enc3 = ConvBlock(2*c, 4*c)      # [B, 256, 128, 128]
        self.enc4 = ConvBlock(4*c, 8*c)      # [B, 512,  64,  64]
        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(8*c, 16*c)   # [B, 1024, 32, 32]

        # ── Zernike MLP head (branches from bottleneck) ───────────────────────
        self.zernike_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # [B, 1024, 1, 1]
            nn.Flatten(),               # [B, 1024]
            nn.Linear(16*c, 4*c),      # [B, 256]
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(4*c, n_modes),   # [B, N_MODES]
        )

        # ── Spatial decoder ───────────────────────────────────────────────────
        self.dec4 = UpBlock(16*c, 8*c, 8*c)   # 32  → 64
        self.dec3 = UpBlock( 8*c, 4*c, 4*c)   # 64  → 128
        self.dec2 = UpBlock( 4*c, 2*c, 2*c)   # 128 → 256
        self.dec1 = UpBlock( 2*c,   c,   c)   # 256 → 512

        self.out_conv = nn.Conv2d(c, 1, kernel_size=1)   # [B, 1, 512, 512]

        # ── Fixed Zernike basis (non-trainable) ───────────────────────────────
        basis, _, circ = build_zernike_basis(n_modes, H, W)
        self.register_buffer("basis", basis_to_tensor(basis))        # [N, H, W]
        self.register_buffer("circ",  torch.from_numpy(circ.copy())) # [H, W] bool

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : [B, 1, H, W]  wrapped phase (network input)

        Returns
        -------
        phi_sample  : [B, 1, H, W]  estimated unwrapped sample phase
        aber_coeffs : [B, N_MODES]  estimated Zernike coefficients
        """
        # Encoder
        e1 = self.enc1(x)                    # [B,  64, 512, 512]
        e2 = self.enc2(self.pool(e1))        # [B, 128, 256, 256]
        e3 = self.enc3(self.pool(e2))        # [B, 256, 128, 128]
        e4 = self.enc4(self.pool(e3))        # [B, 512,  64,  64]
        b  = self.bottleneck(self.pool(e4))  # [B, 1024, 32,  32]

        # Zernike head (global context → coefficients)
        aber_coeffs = self.zernike_head(b)   # [B, N_MODES]

        # Spatial decoder (local detail → sample phase)
        d = self.dec4(b,  e4)    # [B, 512,  64,  64]
        d = self.dec3(d,  e3)    # [B, 256, 128, 128]
        d = self.dec2(d,  e2)    # [B, 128, 256, 256]
        d = self.dec1(d,  e1)    # [B,  64, 512, 512]

        phi_sample = self.out_conv(d)        # [B, 1, 512, 512]

        # Enforce zero outside circular pupil
        phi_sample = phi_sample * self.circ[None, None]

        return phi_sample, aber_coeffs

    # ── Physics helpers ───────────────────────────────────────────────────────

    def reconstruct_aberration(self, aber_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the aberration phase surface from Zernike coefficients.
        phi_aber = Σ_k c_k Z_k

        Parameters
        ----------
        aber_coeffs : [B, N_MODES]

        Returns
        -------
        phi_aber : [B, 1, H, W]
        """
        phi_aber = torch.einsum("bn,nhw->bhw", aber_coeffs, self.basis)
        return phi_aber.unsqueeze(1)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = PhaseAberrationNet(n_modes=N_MODES, base_ch=64).to(device)
    model.eval()

    B = 2
    x = torch.randn(B, 1, H, W, device=device)

    with torch.no_grad():
        phi_sample, aber_coeffs = model(x)
        phi_aber = model.reconstruct_aberration(aber_coeffs)

    print(f"phi_sample  shape : {phi_sample.shape}")    # [2, 1, 512, 512]
    print(f"aber_coeffs shape : {aber_coeffs.shape}")   # [2, 15]
    print(f"phi_aber    shape : {phi_aber.shape}")       # [2, 1, 512, 512]

    # Check pupil masking
    corners = phi_sample[0, 0, 0, 0].item()
    print(f"Corner value (must be 0.0): {corners:.6f}")

    # Parameter count
    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_buffers   = sum(b.numel() for b in model.buffers())
    print(f"Trainable params : {n_trainable:,}")
    print(f"Fixed buffers    : {n_buffers:,}  (Zernike basis + circ mask)")
    print(f"Total params     : {n_total:,}")

    print("\nPhase 4 — models.py OK")
