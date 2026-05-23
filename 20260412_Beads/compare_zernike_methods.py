#!/usr/bin/env python3
"""
Compare my Zernike vs your Zernike (reconstruction_cell.ipynb) on the beads dataset.

Key differences:
  Mine   — single-pass lstsq on all pixels inside the circular aperture
  Yours  — iterative lstsq with outlier rejection (rejects pixels where
            |residual| > sigma_thresh × std, then refits); bead pixels
            are progressively excluded so they don't bias the aberration fit

Usage:
  python3 compare_zernike_methods.py
  python3 compare_zernike_methods.py --data path/to/_raw.bin
  python3 compare_zernike_methods.py --sigma 1.0 --n-iter 5 --n-modes 32
"""

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.fft import fft2, fftshift, dctn, idctn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))
from cell_pipeline import (
    read_frames, make_extraction_mask, find_sideband_centers,
    extract_polarization, DATA_PATH, FIELD_SIZE
)

DEFAULT_DATA  = Path(__file__).parent / "Beads2/Beads_10um_83us_1024x1024x20_2/Beads_10um_83us_1024x1024x20_2_raw.bin"
N_ZERNIKE     = 32
SIGMA_THRESH  = 1.0
N_ITER        = 5


# ── Phase unwrapping (DCT, from your notebook) ────────────────────────────────

def dct_unwrap(wrapped: np.ndarray) -> np.ndarray:
    Ny, Nx = wrapped.shape
    dx  = np.angle(np.exp(1j * np.diff(wrapped, axis=1)))
    dy  = np.angle(np.exp(1j * np.diff(wrapped, axis=0)))
    rho = np.zeros_like(wrapped)
    rho[:, 1:-1] += dx[:, 1:] - dx[:, :-1]
    rho[:, 0]    += dx[:, 0]
    rho[:, -1]   -= dx[:, -1]
    rho[1:-1, :] += dy[1:, :] - dy[:-1, :]
    rho[0, :]    += dy[0, :]
    rho[-1, :]   -= dy[-1, :]
    rho_dct = dctn(rho, type=2, norm="ortho")
    m, n    = np.arange(Ny), np.arange(Nx)
    mm, nn  = np.meshgrid(m, n, indexing="ij")
    denom   = 2 * (np.cos(np.pi * mm / Ny) + np.cos(np.pi * nn / Nx) - 2)
    denom[0, 0] = 1.0
    phi = rho_dct / denom
    phi[0, 0] = 0.0
    return idctn(phi, type=2, norm="ortho")


# ── Zernike basis ─────────────────────────────────────────────────────────────

def zernike_radial(n, m, rho):
    m_abs  = abs(m)
    result = np.zeros_like(rho, dtype=float)
    for k in range((n - m_abs) // 2 + 1):
        c = ((-1)**k * math.factorial(n - k) /
             (math.factorial(k) *
              math.factorial((n + m_abs) // 2 - k) *
              math.factorial((n - m_abs) // 2 - k)))
        result += c * rho ** (n - 2 * k)
    return result


def zernike_poly(n, m, rho, theta):
    r = zernike_radial(n, m, rho)
    return r * (np.cos(m * theta) if m >= 0 else np.sin(-m * theta))


def zernike_nm(j):
    n = 0
    while (n + 1) * (n + 2) // 2 <= j:
        n += 1
    return n, 2 * (j - n * (n + 1) // 2) - n


def build_basis(n_modes, size):
    ax    = np.linspace(-1, 1, size)
    x, y  = np.meshgrid(ax, ax)
    rho   = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    basis = np.zeros((n_modes, size, size), dtype=np.float64)
    for i in range(n_modes):
        n, m = zernike_nm(i)
        if abs(m) <= n and (n - abs(m)) % 2 == 0:
            basis[i] = zernike_poly(n, m, rho, theta)
    return basis, rho


# ── Method A: my single-pass Zernike ─────────────────────────────────────────

def fit_zernike_singlepass(unwrapped, circ, basis):
    """Single lstsq on all pixels in the circular aperture — no outlier rejection."""
    pts    = circ.ravel()
    A      = np.column_stack([basis[i].ravel()[pts] for i in range(len(basis))])
    coeffs, _, _, _ = np.linalg.lstsq(A, unwrapped.ravel()[pts], rcond=None)
    surface = np.sum(coeffs[:, None, None] * basis, axis=0).astype(np.float32)
    surface[~circ] = np.nan
    return surface, coeffs, circ  # no pixels rejected


# ── Method B: your iterative Zernike (reconstruction_cell.ipynb) ──────────────

def fit_zernike_iterative(unwrapped, circ, basis,
                           n_iter=N_ITER, sigma_thresh=SIGMA_THRESH, label=""):
    """
    Iterative Zernike fit with outlier rejection — exact replication of
    fit_zernike_iterative() from reconstruction_cell.ipynb.

    Each iteration:
      1. Fit Zernike to current valid pixels
      2. Compute residual over whole field
      3. Reject pixels where |residual - mean| > sigma_thresh × std
      4. Repeat until no more rejections or n_iter reached
    """
    n_modes    = len(basis)
    fit_mask   = circ.copy()

    for it in range(n_iter):
        pts       = fit_mask.ravel()
        n_valid   = pts.sum()
        A         = np.column_stack([basis[i].ravel()[pts] for i in range(n_modes)])
        coeffs, _, _, _ = np.linalg.lstsq(A, unwrapped.ravel()[pts], rcond=None)

        surface  = np.sum(coeffs[:, None, None] * basis, axis=0)
        residual = unwrapped - surface

        res_valid  = residual[fit_mask]
        res_std    = np.std(res_valid)
        res_mean   = np.mean(res_valid)
        inliers    = np.abs(residual - res_mean) < sigma_thresh * res_std
        new_mask   = circ & inliers
        n_rejected = fit_mask.sum() - new_mask.sum()

        print(f"    [{label}] iter {it+1}: {n_valid} px → rejected {n_rejected}, "
              f"residual std = {res_std:.4f} rad")

        if n_rejected == 0:
            print(f"    [{label}] converged at iter {it+1}")
            break
        fit_mask = new_mask

    surface = np.sum(coeffs[:, None, None] * basis, axis=0).astype(np.float32)
    surface[~circ] = np.nan
    return surface, coeffs, fit_mask   # fit_mask shows which pixels survived


# ── Plot helpers ──────────────────────────────────────────────────────────────

def std_inside(arr, mask):
    return np.nanstd(arr[mask])


def plot_comparison(results, circ, out_dir, n_modes, sigma_thresh):
    fig, axes = plt.subplots(2, 6, figsize=(26, 9))
    fig.suptitle(
        f"My Zernike (single-pass) vs Your Zernike (iterative, σ={sigma_thresh})\n"
        f"{n_modes} Zernike modes",
        fontsize=13
    )

    col_titles = [
        "Raw phase", "Unwrapped phase",
        "My mask\n(single-pass)", "After my Zernike",
        "Your mask\n(iterative)", "After your Zernike"
    ]
    for ci, t in enumerate(col_titles):
        axes[0, ci].set_title(t, fontsize=9)

    for ri, label in enumerate(["P1", "P2"]):
        r = results[label]
        std_mine  = std_inside(r["my_residual"],   circ)
        std_yours = std_inside(r["your_residual"],  circ)

        axes[ri, 3].set_title(f"After my Zernike\n(σ={std_mine:.3f} rad)", fontsize=9)
        axes[ri, 5].set_title(f"After your Zernike\n(σ={std_yours:.3f} rad)", fontsize=9)

        panels = [
            (r["raw_phase"],     "twilight", (-np.pi, np.pi)),
            (r["unwrapped"],     "twilight",  None),
            (r["my_mask"],       "RdBu_r",    None),
            (r["my_residual"],   "twilight",  None),
            (r["your_mask"],     "RdBu_r",    None),
            (r["your_residual"], "twilight",  None),
        ]
        for ci, (data, cmap, clim) in enumerate(panels):
            ax  = axes[ri, ci]
            d   = data * circ if not np.isnan(data[circ][0:1]).any() else data
            vlo = clim[0] if clim else np.nanpercentile(d, 2)
            vhi = clim[1] if clim else np.nanpercentile(d, 98)
            im  = ax.imshow(d, cmap=cmap, origin="upper", vmin=vlo, vmax=vhi)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axes[ri, 0].set_ylabel(label, fontsize=12)

    fig.tight_layout()
    path = out_dir / "zernike_mine_vs_yours.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_rejected_pixels(results, circ, out_dir):
    """Show which pixels your iterative method rejected (likely beads)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Pixels rejected by iterative Zernike fit (likely beads/features)", fontsize=12)

    for ai, label in enumerate(["P1", "P2"]):
        r        = results[label]
        rejected = circ & ~r["your_inlier_mask"]
        n_rej    = rejected.sum()
        pct      = 100 * n_rej / circ.sum()

        axes[ai].imshow(r["unwrapped"] * circ, cmap="gray",
                        vmin=np.nanpercentile(r["unwrapped"][circ], 2),
                        vmax=np.nanpercentile(r["unwrapped"][circ], 98),
                        origin="upper")
        axes[ai].imshow(np.where(rejected, 1.0, np.nan),
                        cmap="Reds", alpha=0.7, origin="upper", vmin=0, vmax=1)
        axes[ai].set_title(f"{label} — {n_rej} px rejected ({pct:.1f}%)")
        axes[ai].axis("off")

    fig.tight_layout()
    path = out_dir / "zernike_rejected_pixels.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_coeffs(results, out_dir, n_modes):
    """Overlay coefficient bars for both methods."""
    zernike_names = [
        "Piston", "Tilt X", "Tilt Y", "Defocus",
        "Astig 0°", "Astig 45°", "Coma X", "Coma Y",
        "Trefoil X", "Trefoil Y", "Spherical",
        "2nd Astig X", "2nd Astig Y", "Tetrafoil X", "Tetrafoil Y",
    ]
    labels_x = [(zernike_names[i] if i < len(zernike_names) else f"Z{i}")
                for i in range(n_modes)]
    x = np.arange(n_modes)
    w = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle("Zernike coefficients: my single-pass vs your iterative fit", fontsize=12)

    for ai, label in enumerate(["P1", "P2"]):
        r = results[label]
        axes[ai].bar(x - w/2, np.abs(r["my_coeffs"]),   w, label="Mine (single-pass)", color="steelblue", alpha=0.8)
        axes[ai].bar(x + w/2, np.abs(r["your_coeffs"]), w, label="Yours (iterative)",  color="tomato",    alpha=0.8)
        axes[ai].set_xticks(x)
        axes[ai].set_xticklabels(labels_x, rotation=45, ha="right", fontsize=8)
        axes[ai].set_ylabel("|coefficient| (rad)")
        axes[ai].set_title(label)
        axes[ai].legend()

    fig.tight_layout()
    path = out_dir / "zernike_coeffs_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_summary_bars(results, circ, out_dir, n_modes, sigma_thresh):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Residual phase std: single-pass vs iterative Zernike", fontsize=12)

    for ai, label in enumerate(["P1", "P2"]):
        r = results[label]
        vals = [
            std_inside(r["unwrapped"] * circ, circ),
            std_inside(r["my_residual"],  circ),
            std_inside(r["your_residual"], circ),
        ]
        bars = axes[ai].bar(
            ["Unwrapped\n(no correction)",
             f"After my\nZernike\n({n_modes} modes)",
             f"After your\nZernike\n(iterative, σ={sigma_thresh})"],
            vals, color=["gray", "steelblue", "tomato"], width=0.5
        )
        for bar, v in zip(bars, vals):
            axes[ai].text(bar.get_x() + bar.get_width()/2, v + 0.02,
                          f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        axes[ai].set_ylabel("Phase std (rad)")
        axes[ai].set_title(label)
        axes[ai].set_ylim(0, vals[0] * 1.2)

    fig.tight_layout()
    path = out_dir / "zernike_summary_bars.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    type=Path, default=DEFAULT_DATA)
    parser.add_argument("--n-modes", type=int,  default=N_ZERNIKE)
    parser.add_argument("--n-iter",  type=int,  default=N_ITER)
    parser.add_argument("--sigma",   type=float,default=SIGMA_THRESH)
    args = parser.parse_args()

    out_dir = args.data.parent / "python_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load + extract
    print(f"Loading {args.data.name} ...")
    volume    = read_frames(args.data)
    n_frames  = volume.shape[0]
    extr_mask = make_extraction_mask(volume.shape[2])
    f_ref     = fftshift(fft2(volume[0].astype(np.float32)))
    fc1, fc2  = find_sideband_centers(f_ref)

    print("Extracting and averaging all frames...")
    p1_avg = np.mean(np.stack([
        extract_polarization(fftshift(fft2(volume[i].astype(np.float32))), fc1, extr_mask)
        for i in range(n_frames)]), axis=0)
    p2_avg = np.mean(np.stack([
        extract_polarization(fftshift(fft2(volume[i].astype(np.float32))), fc2, extr_mask)
        for i in range(n_frames)]), axis=0)

    # Circular aperture
    ax      = np.linspace(-1, 1, FIELD_SIZE)
    xg, yg  = np.meshgrid(ax, ax)
    circ    = (xg**2 + yg**2) <= 1.0

    # Build Zernike basis once (shared by both methods)
    print(f"Building Zernike basis ({args.n_modes} modes, {FIELD_SIZE}×{FIELD_SIZE})...")
    basis, _ = build_basis(args.n_modes, FIELD_SIZE)

    results = {}
    for label, field in [("P1", p1_avg), ("P2", p2_avg)]:
        print(f"\nProcessing {label}...")
        raw_phase = np.angle(field)
        unwrapped = dct_unwrap(raw_phase)

        # Method A: my single-pass — subtract directly in unwrapped domain (your notebook approach)
        print(f"  Method A — single-pass Zernike ({label})")
        my_mask, my_coeffs, _ = fit_zernike_singlepass(unwrapped, circ, basis)
        my_residual = unwrapped - np.nan_to_num(my_mask)
        my_residual[~circ] = np.nan
        my_corrected = np.angle(np.exp(-1j * np.nan_to_num(my_mask)) * field)   # for display only

        # Method B: your iterative — same direct subtraction
        print(f"  Method B — iterative Zernike ({label}), "
              f"σ={args.sigma}, max {args.n_iter} iters")
        your_mask, your_coeffs, your_inliers = fit_zernike_iterative(
            unwrapped, circ, basis,
            n_iter=args.n_iter, sigma_thresh=args.sigma, label=label
        )
        your_residual = unwrapped - np.nan_to_num(your_mask)
        your_residual[~circ] = np.nan
        your_corrected = np.angle(np.exp(-1j * np.nan_to_num(your_mask)) * field)   # for display only

        std_mine  = std_inside(my_residual,   circ)
        std_yours = std_inside(your_residual, circ)
        print(f"  Residual std — mine: {std_mine:.4f} rad  |  yours: {std_yours:.4f} rad")
        print(f"  Improvement from iterative: {std_mine - std_yours:.4f} rad "
              f"({100*(std_mine-std_yours)/std_mine:.1f}% better)")

        results[label] = dict(
            raw_phase=raw_phase, unwrapped=unwrapped,
            my_mask=my_mask,     my_residual=my_residual,   my_coeffs=my_coeffs,
            your_mask=your_mask, your_residual=your_residual, your_coeffs=your_coeffs,
            your_inlier_mask=your_inliers,
        )

    print("\nGenerating plots...")
    plot_comparison(results, circ, out_dir, args.n_modes, args.sigma)
    plot_rejected_pixels(results, circ, out_dir)
    plot_coeffs(results, out_dir, args.n_modes)
    plot_summary_bars(results, circ, out_dir, args.n_modes, args.sigma)
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
