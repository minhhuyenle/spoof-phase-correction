#!/usr/bin/env python3
"""
Zernike polynomial utilities for the DL phase-aberration correction pipeline.

Indexing convention: OSA/ANSI single-index j — matches zernike_index_to_nm()
in 20260412_Beads/compare_corrections.py.

  j | (n,  m) | name
  --+---------+------------------
  0 | (0,  0) | Piston
  1 | (1, -1) | Tilt Y
  2 | (1, +1) | Tilt X
  3 | (2, -2) | Astig 45°
  4 | (2,  0) | Defocus
  5 | (2, +2) | Astig 0°
  6 | (3, -3) | Trefoil Y
  7 | (3, -1) | Coma Y
  8 | (3, +1) | Coma X
  9 | (3, +3) | Trefoil X
 10 | (4, -4) | Tetrafoil Y
 11 | (4, -2) | 2nd Astig 45°
 12 | (4,  0) | Spherical
 13 | (4, +2) | 2nd Astig 0°
 14 | (4, +4) | Tetrafoil X

N_MODES = 15 covers radial orders 0–4 (1+2+3+4+5 = 15 terms), capturing all
common aberrations up to and including spherical aberration and tetrafoil.
"""

import math
import numpy as np
import torch

N_MODES = 15

ZERNIKE_NAMES = [
    "Piston",
    "Tilt Y", "Tilt X",
    "Astig 45°", "Defocus", "Astig 0°",
    "Trefoil Y", "Coma Y", "Coma X", "Trefoil X",
    "Tetrafoil Y", "2nd Astig 45°", "Spherical", "2nd Astig 0°", "Tetrafoil X",
]


# ── Index conversion ──────────────────────────────────────────────────────────

def zernike_index_to_nm(j: int) -> tuple[int, int]:
    """OSA/ANSI single index j → (n, m) radial/azimuthal order pair."""
    n = 0
    while (n + 1) * (n + 2) // 2 <= j:
        n += 1
    m = 2 * (j - n * (n + 1) // 2) - n
    return n, m


# ── Polynomial evaluation ─────────────────────────────────────────────────────

def _radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Radial polynomial R_n^|m|(rho)."""
    m_abs = abs(m)
    result = np.zeros_like(rho, dtype=np.float64)
    for k in range((n - m_abs) // 2 + 1):
        c = (
            (-1) ** k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + m_abs) // 2 - k)
                * math.factorial((n - m_abs) // 2 - k)
            )
        )
        result += c * rho ** (n - 2 * k)
    return result


def _poly(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Full Zernike polynomial Z_n^m(rho, theta)."""
    radial = _radial(n, m, rho)
    if m >= 0:
        return radial * np.cos(m * theta)
    else:
        return radial * np.sin(-m * theta)


# ── Basis construction ────────────────────────────────────────────────────────

def build_zernike_basis(
    n_modes: int = N_MODES,
    H: int = 512,
    W: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Zernike basis on an H×W grid.

    The pupil unit disk is inscribed in the image: rho=1 at the shorter image
    half-edge. Pixels outside the unit disk are set to 0.

    Parameters
    ----------
    n_modes : number of Zernike modes (OSA/ANSI order)
    H, W    : image height and width in pixels

    Returns
    -------
    basis : float64 ndarray [n_modes, H, W]
    rho   : float64 ndarray [H, W]  — normalised radial coordinate
    circ  : bool    ndarray [H, W]  — True inside the unit disk
    """
    y_ax = np.linspace(-1, 1, H, dtype=np.float64)
    x_ax = np.linspace(-1, 1, W, dtype=np.float64)
    x, y = np.meshgrid(x_ax, y_ax)
    rho   = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    circ  = rho <= 1.0

    basis = np.zeros((n_modes, H, W), dtype=np.float64)
    for j in range(n_modes):
        n, m = zernike_index_to_nm(j)
        poly = _poly(n, m, rho, theta)
        poly[~circ] = 0.0
        basis[j] = poly

    return basis, rho, circ


# ── Fitting and reconstruction ────────────────────────────────────────────────

def fit_zernike_lstsq(
    phase_map: np.ndarray,
    circ_mask: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """
    Fit Zernike coefficients to an unwrapped phase map by least squares.

    Parameters
    ----------
    phase_map : [H, W] float — unwrapped phase in radians
    circ_mask : [H, W] bool  — True for pixels inside the pupil
    basis     : [N, H, W] float — precomputed Zernike basis

    Returns
    -------
    coeffs : [N] float64 — coefficients c_k such that Σ c_k Z_k ≈ phase_map
    """
    n_modes = basis.shape[0]
    pts = circ_mask.ravel()
    A = np.column_stack([basis[k].ravel()[pts] for k in range(n_modes)])
    b = phase_map.ravel()[pts]
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return coeffs


def reconstruct_from_coeffs(
    coeffs: np.ndarray,
    basis: np.ndarray,
    circ_mask: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct a phase surface from Zernike coefficients.

    Returns [H, W] float32 with zeros outside the pupil.
    """
    surface = np.einsum("k,khw->hw", coeffs, basis)
    surface[~circ_mask] = 0.0
    return surface.astype(np.float32)


def project_onto_zernike(
    phase_map: np.ndarray,
    basis: np.ndarray,
    circ_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project phase_map onto the Zernike subspace.

    Returns
    -------
    projected : [H, W] float32 — best-fit Zernike surface
    coeffs    : [N]    float64
    """
    coeffs = fit_zernike_lstsq(phase_map, circ_mask, basis)
    projected = reconstruct_from_coeffs(coeffs, basis, circ_mask)
    return projected, coeffs


# ── PyTorch helper ────────────────────────────────────────────────────────────

def basis_to_tensor(basis: np.ndarray) -> torch.Tensor:
    """Convert numpy basis [N, H, W] to float32 torch tensor."""
    return torch.from_numpy(basis.astype(np.float32))


def reconstruct_from_coeffs_torch(
    coeffs: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct phase surface inside the network forward pass.

    Parameters
    ----------
    coeffs : [B, N] — batch of Zernike coefficient vectors
    basis  : [N, H, W] — fixed Zernike basis (model buffer)

    Returns
    -------
    surface : [B, 1, H, W]
    """
    # einsum: b n, n h w -> b h w, then add channel dim
    surface = torch.einsum("bn,nhw->bhw", coeffs, basis)
    return surface.unsqueeze(1)


# ── Self-test / visualisation ─────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_dir = Path(__file__).parent / "calibration_stats"
    out_dir.mkdir(exist_ok=True)

    print(f"Building Zernike basis: {N_MODES} modes on 512×512 ...")
    basis, rho, circ = build_zernike_basis(N_MODES, 512, 512)
    print(f"  basis shape: {basis.shape}")

    # Orthogonality check inside the unit disk
    flat = basis.reshape(N_MODES, -1)[:, circ.ravel()]
    gram = flat @ flat.T / circ.sum()
    off_diag = gram - np.diag(np.diag(gram))
    print(f"  Gram matrix off-diagonal max: {np.abs(off_diag).max():.4f}  (ideally 0)")

    # Round-trip test: synthesise random coefficients, fit back, compare
    rng = np.random.default_rng(0)
    c_true = rng.uniform(-1, 1, N_MODES)
    phi_true = reconstruct_from_coeffs(c_true, basis, circ)
    c_fit = fit_zernike_lstsq(phi_true, circ, basis)
    print(f"  Round-trip coeff error: max={np.abs(c_fit - c_true).max():.2e}")

    # Plot all 15 modes in pyramid layout (one row per radial order)
    max_order = 4   # orders 0..4
    ncols = 2 * max_order + 1   # widest row has 5 modes → 9 columns
    nrows = max_order + 1       # 5 rows

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 1.6, nrows * 2.2))
    #fig.suptitle(f"Zernike basis — {N_MODES} modes (OSA/ANSI, 512×512)",
                 #fontsize=13)
    fig.subplots_adjust(wspace=0.05, hspace=0.15)

    # Hide all axes first, then fill in the pyramid positions
    for ax in axes.ravel():
        ax.axis("off")

    j = 0
    for n in range(max_order + 1):
        # azimuthal orders for radial order n: -n, -(n-2), ..., n-2, n
        m_values = list(range(-n, n + 1, 2))
        # centre the row in the ncols grid
        col_start = max_order - n
        for col_offset, m in enumerate(m_values):
            col = col_start + col_offset * 2
            ax = axes[n, col]
            data = basis[j].copy()
            data[~circ] = np.nan
            im = ax.imshow(data, cmap="RdBu_r", origin="upper",
                           vmin=-1, vmax=1)
            name = ZERNIKE_NAMES[j] if j < len(ZERNIKE_NAMES) else f"Z{j}"
            ax.set_title(f"Z{j}\n{name}\n(n={n}, m={m:+d})", fontsize=6.5)
            ax.axis("on")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            j += 1

        # Label each row with its radial order
        axes[n, 0].set_ylabel(f"n = {n}", fontsize=9, labelpad=4)

    out_path = out_dir / "zernike_basis_modes.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
    print("Phase 1 — zernike_utils.py OK")
