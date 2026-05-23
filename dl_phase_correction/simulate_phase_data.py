#!/usr/bin/env python3
"""
Phase 3 — On-the-fly synthetic phase dataset for DL aberration correction.

Each sample contains:
  phi_total   = wrap(phi_sample + phi_aber + noise)   (network input, wrapped [-π,π], [1,H,W])
  phi_sample                                           (GT: true unwrapped sample phase, [1,H,W])
  aber_coeffs                                          (GT: Zernike coefficients, [N_MODES])

The network input is the *wrapped* phase — exactly what the hologram reconstruction
gives before any unwrapping step.  The targets are the *unwrapped* sample phase and
the Zernike coefficients of the aberration.  The network learns to simultaneously
unwrap and separate sample from aberration.

Sample pattern mix:
  30% USAF 1951 resolution target
  20% Ronchi binary gratings
  50% Cell-like Gaussian blobs

Aberrations drawn from Normal(mu_k, sigma_k) calibrated on mirror data.
Noise sigma taken from calibration residual (≈0.98 rad).
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

HERE  = Path(__file__).parent
BEADS = HERE.parent / "20260412_Beads"
sys.path.insert(0, str(HERE))

from zernike_utils import (
    build_zernike_basis, reconstruct_from_coeffs, basis_to_tensor,
    N_MODES,
)

# ── Constants ─────────────────────────────────────────────────────────────────
H = W = 512          # extracted field size (FIELD_SIZE from cell_pipeline)
STATS_PATH  = HERE / "calibration_stats/zernike_stats.npz"
# Large USAF image used for random crops (960×1016 native resolution)
USAF_LARGE = HERE.parent / "USAF_images/USAF-1951.svg.png"
USAF_CROP  = 600   # crop window before resizing to H×W

# ── Module-level preloads (built once, shared across workers) ─────────────────
_stats = np.load(STATS_PATH)
_basis, _, _circ = build_zernike_basis(N_MODES, H, W)  # [N,H,W], [H,W], [H,W]
_basis_t = basis_to_tensor(_basis)  # float32 torch tensor — used by model

if not USAF_LARGE.exists():
    raise FileNotFoundError(f"USAF image not found: {USAF_LARGE}")
# The file is LA mode: L channel is all zeros, pattern is in the alpha channel.
# Keep at native resolution; crops are taken in sample_usaf().
_usaf_full = np.array(Image.open(USAF_LARGE))[:, :, 1].astype(np.float32) / 255.0
_usaf_h, _usaf_w = _usaf_full.shape   # 1016 × 960


# ── Aberration sampler ────────────────────────────────────────────────────────

def sample_aberration(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw Zernike coefficients from calibrated Normal(mu_k, sigma_k).

    Returns
    -------
    phi_aber : [H, W] float32 — reconstructed aberration surface
    coeffs   : [N_MODES] float64 — Zernike coefficients
    """
    mu    = _stats["mean_pooled"]
    sigma = _stats["std_pooled"]
    low   = _stats["low_pooled"]
    high  = _stats["high_pooled"]
    coeffs = rng.normal(mu, sigma).clip(low, high)
    phi_aber = reconstruct_from_coeffs(coeffs, _basis, _circ)
    return phi_aber, coeffs


# ── Sample-phase generators ───────────────────────────────────────────────────

def sample_usaf(
    rng: np.random.Generator,
    amp_range: tuple[float, float] = (1.5, 3.0),
) -> np.ndarray:
    """Random USAF target patch scaled to a realistic phase amplitude."""
    # Random 600×600 crop from the native 960×1016 image, then resize to 512×512.
    # Retry until the crop has visible bar contrast (std > 0.05).
    for _ in range(50):
        y0 = rng.integers(0, _usaf_h - USAF_CROP + 1)
        x0 = rng.integers(0, _usaf_w - USAF_CROP + 1)
        crop = _usaf_full[y0:y0 + USAF_CROP, x0:x0 + USAF_CROP]
        if crop.std() > 0.05:
            break
    img  = np.array(
        Image.fromarray(crop).resize((W, H), Image.LANCZOS),
        dtype=np.float32,
    )
    if rng.random() < 0.5:      # random polarity
        img = 1.0 - img
    amp = rng.uniform(*amp_range)
    phi = (img * amp).astype(np.float32)
    phi[~_circ] = 0.0
    return phi


def sample_ronchi(
    rng: np.random.Generator,
    freq_range: tuple[float, float] = (3.0, 20.0),
    amp_range: tuple[float, float] = (0.3, 2.0),
) -> np.ndarray:
    """Binary grating at random orientation and spatial frequency."""
    freq  = rng.uniform(*freq_range)
    theta = rng.uniform(0.0, np.pi)
    amp   = rng.uniform(*amp_range)
    ax = np.linspace(-1.0, 1.0, W, dtype=np.float64)
    x, y = np.meshgrid(ax, ax)
    stripe = np.sin(np.pi * freq * (x * np.cos(theta) + y * np.sin(theta)))
    phi = amp * (stripe > 0).astype(np.float32)
    phi[~_circ] = 0.0
    return phi


def sample_cell_blobs(
    rng: np.random.Generator,
    n_range: tuple[int, int] = (10, 20),
    amp_range: tuple[float, float] = (0.5, 6.0),  # Δφ = 4π·Δn·h/λ, Δn≈0.02–0.05, h≈2–10µm, λ=860nm
) -> np.ndarray:
    """Superposition of random Gaussian ellipses mimicking cell OPL profiles."""
    n_cells = rng.integers(*n_range)
    ax = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    x, y = np.meshgrid(ax, ax)
    phi = np.zeros((H, W), dtype=np.float32)

    for _ in range(n_cells):
        # Rejection sample center inside 85% of unit circle
        while True:
            cx = rng.uniform(-0.85, 0.85)
            cy = rng.uniform(-0.85, 0.85)
            if cx**2 + cy**2 < 0.72**2:
                break
        rx    = rng.uniform(0.03, 0.18)   # semi-axes in normalised coords
        ry    = rng.uniform(0.03, 0.18)
        angle = rng.uniform(0.0, np.pi)
        amp   = rng.uniform(*amp_range)

        dx    = x - cx
        dy    = y - cy
        x_rot =  dx * np.cos(angle) + dy * np.sin(angle)
        y_rot = -dx * np.sin(angle) + dy * np.cos(angle)
        phi  += amp * np.exp(-0.5 * ((x_rot / rx) ** 2 + (y_rot / ry) ** 2))

    phi[~_circ] = 0.0
    return phi


# ── Full sample generator ─────────────────────────────────────────────────────

def generate_sample(
    rng: np.random.Generator,
    pattern: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Generate one synthetic (phi_total, phi_sample, aber_coeffs) tuple.

    Parameters
    ----------
    rng     : seeded or random numpy Generator
    pattern : "usaf" | "ronchi" | "cells" | None (random mix)

    Returns
    -------
    dict with keys:
      phi_total   [1, H, W] float32 — WRAPPED to [-π, π]  (network input)
      phi_sample  [1, H, W] float32 — UNWRAPPED true phase (GT output)
      aber_coeffs [N_MODES] float32 — Zernike coefficients  (GT output)
    """
    if pattern is None:
        r = rng.random()
        pattern = "usaf" if r < 0.30 else "ronchi" if r < 0.50 else "cells"

    if pattern == "usaf":
        phi_s = sample_usaf(rng)
    elif pattern == "ronchi":
        phi_s = sample_ronchi(rng)
    else:
        phi_s = sample_cell_blobs(rng)

    phi_aber, coeffs = sample_aberration(rng)

    noise_std = float(_stats["noise_std"])
    noise = rng.normal(0.0, noise_std, (H, W)).astype(np.float32)
    noise[~_circ] = 0.0

    # Unwrapped total (physics ground truth, for reference)
    phi_total_unwrapped = phi_s + phi_aber + noise

    # Wrap to [-π, π] — this is what hologram reconstruction gives before DCT unwrap
    phi_total_wrapped = np.angle(np.exp(1j * phi_total_unwrapped)).astype(np.float32)
    phi_total_wrapped[~_circ] = 0.0

    return {
        "phi_total":   phi_total_wrapped[np.newaxis],       # [1, H, W] wrapped input
        "phi_sample":  phi_s[np.newaxis],                   # [1, H, W] unwrapped GT
        "aber_coeffs": coeffs.astype(np.float32),           # [N_MODES]
    }


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class PhaseDataset(Dataset):
    """
    On-the-fly synthetic phase dataset.

    Parameters
    ----------
    n_samples : virtual epoch size (each call to __getitem__ generates fresh data)
    seed      : fixed seed for reproducible validation sets; None = random train
    """

    def __init__(self, n_samples: int = 4000, seed: int | None = None):
        self.n_samples = n_samples
        self.seed = seed

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.seed is not None:
            rng = np.random.default_rng(int(self.seed) * 1_000_000 + idx)
        else:
            rng = np.random.default_rng()
        sample = generate_sample(rng)
        return {k: torch.from_numpy(v) for k, v in sample.items()}


def make_train_val_datasets(
    n_train: int = 4000,
    n_val:   int = 500,
    val_seed: int = 42,
) -> tuple[PhaseDataset, PhaseDataset]:
    """Convenience constructor used by train_supervised.py."""
    return PhaseDataset(n_train, seed=None), PhaseDataset(n_val, seed=val_seed)


# ── Visualisation ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from zernike_utils import ZERNIKE_NAMES

    OUT = HERE / "calibration_stats"
    rng = np.random.default_rng(0)

    patterns   = ["usaf", "ronchi", "cells", "cells"]
    row_labels = ["USAF", "Ronchi", "Cells (A)", "Cells (B)"]
    col_labels = [
        "φ_sample\n(GT, unwrapped)",
        "φ_aber\n(GT, unwrapped)",
        "φ_total\n(unwrapped ref.)",
        "φ_total\n(WRAPPED — network input)",
        "3D view\n(φ_sample)",
    ]
    cmaps_2d = ["RdBu_r", "RdBu_r", "RdBu_r", "twilight"]

    nrows  = len(patterns)
    ncols  = len(col_labels)   # 5 columns
    STRIDE = 4                 # downsample for 3-D surface speed

    # Build axes manually so column 4 can use projection='3d'
    fig = plt.figure(figsize=(ncols * 3.8, nrows * 3.6))
    axes = []
    for ri in range(nrows):
        row = []
        for ci in range(ncols):
            proj = "3d" if ci == 4 else None
            ax   = fig.add_subplot(nrows, ncols, ri * ncols + ci + 1,
                                   projection=proj)
            row.append(ax)
        axes.append(row)

    fig.suptitle(
        "Simulated training examples\n"
        "Network input = wrapped φ_total  |  GT outputs = unwrapped φ_sample + Zernike coeffs",
        fontsize=12, y=1.01,
    )
    for ci, title in enumerate(col_labels):
        axes[0][ci].set_title(title, fontsize=9, pad=6)

    # Shared grid for 3-D surface
    _ax  = np.linspace(-1, 1, W // STRIDE)
    _X3, _Y3 = np.meshgrid(_ax, _ax)

    noise_std = float(_stats["noise_std"])

    for ri, (pat, row_label) in enumerate(zip(patterns, row_labels)):
        phi_aber, coeffs = sample_aberration(rng)
        if pat == "usaf":
            phi_s = sample_usaf(rng)
        elif pat == "ronchi":
            phi_s = sample_ronchi(rng)
        else:
            phi_s = sample_cell_blobs(rng)

        noise_map = rng.normal(0.0, noise_std, (H, W)).astype(np.float32)
        noise_map[~_circ] = 0.0

        phi_total_unwr = phi_s + phi_aber + noise_map
        phi_total_unwr[~_circ] = 0.0
        phi_total_wrap = np.angle(np.exp(1j * phi_total_unwr)).astype(np.float32)
        phi_total_wrap[~_circ] = 0.0

        # ── 2-D panels (columns 0–3) ─────────────────────────────────────────
        panels = [phi_s, phi_aber, phi_total_unwr, phi_total_wrap]
        for ci, (data, cmap) in enumerate(zip(panels, cmaps_2d)):
            ax = axes[ri][ci]
            d  = data.copy().astype(float)
            d[~_circ] = np.nan
            if cmap == "twilight":
                im = ax.imshow(d, cmap=cmap, origin="upper",
                               vmin=-np.pi, vmax=np.pi)
            else:
                vabs = np.nanpercentile(np.abs(d), 99)
                im = ax.imshow(d, cmap=cmap, origin="upper",
                               vmin=-vabs, vmax=vabs)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="rad")

        axes[ri][0].set_ylabel(row_label, fontsize=10, labelpad=6)
        top = int(np.argmax(np.abs(coeffs)))
        axes[ri][1].set_xlabel(
            f"dominant: Z{top} {ZERNIKE_NAMES[top]}  c={coeffs[top]:.2f} rad",
            fontsize=7, labelpad=3,
        )

        # ── 3-D surface (column 4) ───────────────────────────────────────────
        ax3 = axes[ri][4]
        Z   = phi_s.copy().astype(float)
        Z[~_circ] = np.nan
        Z_ds = Z[::STRIDE, ::STRIDE]

        vmin3, vmax3 = np.nanmin(Z_ds), np.nanmax(Z_ds)
        surf = ax3.plot_surface(
            _X3, _Y3, Z_ds,
            cmap="turbo",
            vmin=vmin3, vmax=vmax3,
            linewidth=0, antialiased=True, alpha=0.95,
        )
        ax3.view_init(elev=75, azim=-75)
        ax3.set_zlabel("rad", fontsize=7, labelpad=2)
        ax3.tick_params(labelsize=6, pad=1)
        ax3.set_xticks([])
        ax3.set_yticks([])
        fig.colorbar(surf, ax=ax3, fraction=0.03, pad=0.08,
                     label="rad", shrink=0.6)

    fig.tight_layout()
    out_path = OUT / "simulated_examples.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # ── Quick sanity check on the Dataset ────────────────────────────────────
    print("\nDataset sanity check ...")
    ds_train = PhaseDataset(n_samples=100, seed=None)
    ds_val   = PhaseDataset(n_samples=20, seed=42)

    batch = ds_train[0]
    print(f"  phi_total  shape : {batch['phi_total'].shape}  dtype={batch['phi_total'].dtype}")
    print(f"  phi_sample shape : {batch['phi_sample'].shape}")
    print(f"  aber_coeffs shape: {batch['aber_coeffs'].shape}")

    # Reproducibility: val[0] must be identical across two calls
    v0a = ds_val[0]["phi_total"]
    v0b = ds_val[0]["phi_total"]
    print(f"  Val reproducible : {torch.allclose(v0a, v0b)}")

    # Train must differ between calls (unseeded)
    t0a = ds_train[0]["phi_total"]
    t0b = ds_train[0]["phi_total"]
    print(f"  Train random     : {not torch.allclose(t0a, t0b)}")
    print("\nPhase 3 — simulate_phase_data.py OK")
