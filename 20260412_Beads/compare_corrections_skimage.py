#!/usr/bin/env python3
"""
Compare quadratic vs Zernike phase correction on A549 cell data.
Uses skimage.restoration.unwrap_phase instead of DCT unwrapping.

Outputs (in python_results_skimage/):
  compare_quadratic_vs_zernike.png  — side-by-side corrected phase + residuals
  compare_residual_std.png          — how much aberration each method removes
  compare_zernike_coeffs.png        — bar chart of fitted Zernike coefficients
"""

import math
from pathlib import Path

import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.restoration import unwrap_phase
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── reuse helpers from cell_pipeline.py ──────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from cell_pipeline import (
    read_frames, make_extraction_mask, find_sideband_centers,
    extract_polarization, DATA_PATH, FIELD_SIZE
)


N_ZERNIKE = 21


# ── Phase unwrapping — skimage ────────────────────────────────────────────────

def ski_unwrap(wrapped: np.ndarray) -> np.ndarray:
    return unwrap_phase(wrapped)


# ── Method 1: Quadratic fit ───────────────────────────────────────────────────

def fit_quadratic(unwrapped: np.ndarray, circ: np.ndarray):
    """Fit 6-term quadratic (friend's SPoOF form) by least-squares."""
    size = unwrapped.shape[0]
    ax   = np.linspace(-1, 1, size, dtype=np.float64)
    x, y = np.meshgrid(ax, ax)
    pts  = circ.ravel()
    A    = np.column_stack([
        y.ravel()[pts], x.ravel()[pts],
        (y**2).ravel()[pts], (x**2).ravel()[pts],
        (x*y).ravel()[pts], np.ones(pts.sum()),
    ])
    coeffs, _, _, _ = np.linalg.lstsq(A, unwrapped.ravel()[pts], rcond=None)
    y1, x1, y2, x2, xyc, off = coeffs
    mask = (y1*y + x1*x + y2*y**2 + x2*x**2 + xyc*x*y + off).astype(np.float32)
    return mask, dict(y1=y1, x1=x1, y2=y2, x2=x2, xyc=xyc, off=off)


# ── Method 2: Zernike fit ────────────────────────────────────────────────────

def zernike_radial(n, m, rho):
    m_abs  = abs(m)
    result = np.zeros_like(rho, dtype=float)
    for k in range((n - m_abs) // 2 + 1):
        coeff = ((-1)**k * math.factorial(n - k) /
                 (math.factorial(k) *
                  math.factorial((n + m_abs) // 2 - k) *
                  math.factorial((n - m_abs) // 2 - k)))
        result += coeff * rho ** (n - 2 * k)
    return result


def zernike_poly(n, m, rho, theta):
    radial = zernike_radial(n, m, rho)
    return radial * (np.cos(m * theta) if m >= 0 else np.sin(-m * theta))


def zernike_index_to_nm(j):
    n = 0
    while (n + 1) * (n + 2) // 2 <= j:
        n += 1
    m = 2 * (j - n * (n + 1) // 2) - n
    return n, m


ZERNIKE_NAMES = [
    "Piston", "Tilt X", "Tilt Y",
    "Defocus", "Astig 0°", "Astig 45°",
    "Coma X", "Coma Y", "Trefoil X", "Trefoil Y",
    "Spherical", "2nd Astig X", "2nd Astig Y", "Tetrafoil X", "Tetrafoil Y",
]


def build_zernike_basis(n_modes: int, size: int):
    y_ax = np.linspace(-1, 1, size)
    x_ax = np.linspace(-1, 1, size)
    x, y = np.meshgrid(x_ax, y_ax)
    rho   = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    basis = np.zeros((n_modes, size, size), dtype=np.float64)
    for i in range(n_modes):
        n, m = zernike_index_to_nm(i)
        if abs(m) <= n and (n - abs(m)) % 2 == 0:
            basis[i] = zernike_poly(n, m, rho, theta)
    return basis, rho


def fit_zernike(unwrapped: np.ndarray, circ: np.ndarray,
                n_modes: int = N_ZERNIKE):
    size  = unwrapped.shape[0]
    basis, rho = build_zernike_basis(n_modes, size)
    pts   = circ.ravel()
    A     = np.column_stack([basis[i].ravel()[pts] for i in range(n_modes)])
    coeffs, _, _, _ = np.linalg.lstsq(A, unwrapped.ravel()[pts], rcond=None)
    mask  = np.sum(coeffs[:, None, None] * basis, axis=0).astype(np.float32)
    mask[~circ] = 0
    return mask, coeffs


# ── Main comparison ───────────────────────────────────────────────────────────

def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()

    data_path = args.data
    out_dir   = data_path.parent / "python_results_skimage"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {data_path.name} ...")
    volume    = read_frames(data_path)
    n_frames  = volume.shape[0]
    extr_mask = make_extraction_mask(volume.shape[2])

    ref = volume[0].astype(np.float32)
    f   = fftshift(fft2(ref))
    fc1, fc2 = find_sideband_centers(f)

    print("Extracting and averaging all frames (complex mean)...")
    p1_avg = np.mean(np.stack([
        extract_polarization(fftshift(fft2(volume[i].astype(np.float32))), fc1, extr_mask)
        for i in range(n_frames)]), axis=0)
    p2_avg = np.mean(np.stack([
        extract_polarization(fftshift(fft2(volume[i].astype(np.float32))), fc2, extr_mask)
        for i in range(n_frames)]), axis=0)

    ax   = np.linspace(-1, 1, FIELD_SIZE)
    xg, yg = np.meshgrid(ax, ax)
    circ = (xg**2 + yg**2) <= 1.0

    results = {}
    for label, field in [("P1", p1_avg), ("P2", p2_avg)]:
        print(f"\nProcessing {label}...")
        raw_phase  = np.angle(field)
        unwrapped  = ski_unwrap(raw_phase)

        print(f"  Fitting quadratic ({label})...")
        quad_mask, quad_coeffs = fit_quadratic(unwrapped, circ)
        quad_residual = unwrapped - quad_mask
        quad_residual[~circ] = np.nan
        quad_corrected = np.angle(np.exp(-1j * quad_mask) * field)

        print(f"  Fitting {N_ZERNIKE} Zernike modes ({label})...")
        zern_mask, zern_coeffs = fit_zernike(unwrapped, circ, N_ZERNIKE)
        zern_residual = unwrapped - zern_mask
        zern_residual[~circ] = np.nan
        zern_corrected = np.angle(np.exp(-1j * zern_mask) * field)

        results[label] = dict(
            raw_phase=raw_phase,
            unwrapped=unwrapped,
            quad_mask=quad_mask,
            quad_corrected=quad_corrected,
            quad_residual=quad_residual,
            zern_mask=zern_mask,
            zern_corrected=zern_corrected,
            zern_residual=zern_residual,
            zern_coeffs=zern_coeffs,
        )

        std_raw  = np.nanstd(unwrapped[circ])
        std_quad = np.nanstd(quad_residual[circ])
        std_zern = np.nanstd(zern_residual[circ])
        print(f"  Phase std (rad):  raw={std_raw:.3f}  "
              f"after quadratic={std_quad:.3f}  after Zernike={std_zern:.3f}")
        print(f"  Quadratic removes {100*(1-std_quad/std_raw):.1f}% of aberration variance")
        print(f"  Zernike   removes {100*(1-std_zern/std_raw):.1f}% of aberration variance")

    # ── Plot 1: side-by-side comparison ──────────────────────────────────────
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle("Quadratic vs Zernike (skimage unwrapping) — averaged over all frames",
                 fontsize=13)

    col_titles = ["Raw phase", "Quadratic mask", "After quadratic",
                  "Zernike mask", "After Zernike"]
    for ci, t in enumerate(col_titles):
        axes[0, ci].set_title(t, fontsize=10)

    for ri, label in enumerate(["P1", "P2"]):
        r = results[label]
        panels = [
            (r["raw_phase"],     "twilight", (-np.pi, np.pi)),
            (r["quad_mask"],     "RdBu_r",   None),
            (r["quad_residual"], "twilight",  None),
            (r["zern_mask"],     "RdBu_r",   None),
            (r["zern_residual"], "twilight",  None),
        ]
        std_q = np.nanstd(r["quad_residual"][circ])
        std_z = np.nanstd(r["zern_residual"][circ])
        axes[ri, 2].set_title(f"After quadratic  (σ={std_q:.3f} rad)", fontsize=9)
        axes[ri, 4].set_title(f"After Zernike    (σ={std_z:.3f} rad)", fontsize=9)

        for ci, (data, cmap, clim) in enumerate(panels):
            ax = axes[ri, ci]
            im = ax.imshow(data * circ, cmap=cmap, origin="upper",
                           vmin=clim[0] if clim else np.nanpercentile(data[circ], 1),
                           vmax=clim[1] if clim else np.nanpercentile(data[circ], 99))
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axes[ri, 0].set_ylabel(label, fontsize=12)

    fig.tight_layout()
    fig.savefig(out_dir / "compare_quadratic_vs_zernike.png", dpi=150)
    plt.close(fig)
    print("  Saved compare_quadratic_vs_zernike.png")

    # ── Plot 2: residual std bar chart ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("How much aberration each method removes (skimage unwrap)", fontsize=12)

    for ai, label in enumerate(["P1", "P2"]):
        r = results[label]
        std_raw  = np.nanstd(ski_unwrap(r["raw_phase"])[circ])
        std_quad = np.nanstd(r["quad_residual"][circ])
        std_zern = np.nanstd(r["zern_residual"][circ])

        bars = axes[ai].bar(
            ["Raw", "After\nquadratic\n(6 terms)", f"After\nZernike\n({N_ZERNIKE} modes)"],
            [std_raw, std_quad, std_zern],
            color=["steelblue", "tomato", "seagreen"], width=0.5
        )
        for bar, val in zip(bars, [std_raw, std_quad, std_zern]):
            axes[ai].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                          f"{val:.3f}", ha="center", va="bottom", fontsize=10)
        axes[ai].set_ylabel("Phase std (rad)")
        axes[ai].set_title(f"{label} — residual phase spread")
        axes[ai].set_ylim(0, std_raw * 1.2)

    fig.tight_layout()
    fig.savefig(out_dir / "compare_residual_std.png", dpi=150)
    plt.close(fig)
    print("  Saved compare_residual_std.png")

    # ── Plot 3: Zernike coefficient bar chart ─────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"Fitted Zernike coefficients ({N_ZERNIKE} modes, skimage unwrap)",
                 fontsize=12)

    mode_labels = [
        (ZERNIKE_NAMES[i] if i < len(ZERNIKE_NAMES) else f"Z{i}")
        for i in range(N_ZERNIKE)
    ]
    x = np.arange(N_ZERNIKE)
    quad_equiv = [1, 2, 3, 4, 5]

    for ai, label in enumerate(["P1", "P2"]):
        coeffs = results[label]["zern_coeffs"]
        colors = ["tomato" if i in quad_equiv else "steelblue" for i in range(N_ZERNIKE)]
        axes[ai].bar(x, np.abs(coeffs), color=colors, width=0.7)
        axes[ai].set_xticks(x)
        axes[ai].set_xticklabels(mode_labels, rotation=45, ha="right", fontsize=8)
        axes[ai].set_ylabel("|coefficient| (rad)")
        axes[ai].set_title(f"{label}  (red = terms captured by quadratic fit)")

    fig.tight_layout()
    fig.savefig(out_dir / "compare_zernike_coeffs.png", dpi=150)
    plt.close(fig)
    print("  Saved compare_zernike_coeffs.png")

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    run()
