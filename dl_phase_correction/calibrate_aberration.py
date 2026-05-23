#!/usr/bin/env python3
"""
Calibrate Zernike aberration statistics from mirror (empty-field) measurements.

Since there is no sample in front of the mirror, the reconstructed phase is
purely the system aberration:  φ_measured ≈ φ_aber + noise.

Mirror datasets used (all inside 20260412_Beads/):
  mirror_50fps_83us_1024x1024x20_5   →  22 frames
  mirror_50fps_83us_1024x1024x20_6   →  22 frames
  mirror_50fps_83us_1024x1024x500_5  → 254 frames
  ─────────────────────────────────────────────────
  Total: 298 frames × 2 polarisations = 596 phase maps

Outputs saved to dl_phase_correction/calibration_stats/:
  zernike_stats.npz           — per-mode μ, σ, min, max, sampling ranges
  calibration_histograms.png  — per-mode coefficient distributions
  calibration_examples.png    — 6 example reconstructed aberration surfaces
  calibration_mean_aber.png   — mean aberration surface (P1, P2, pooled)
"""

import sys
from pathlib import Path

import numpy as np
from scipy.fft import fft2, fftshift

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Import hologram-reconstruction helpers from existing pipeline ─────────────
HERE  = Path(__file__).parent
BEADS = HERE.parent / "20260412_Beads"
OUT   = HERE / "calibration_stats"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(BEADS))
from cell_pipeline import (
    read_frames, make_extraction_mask,
    find_sideband_centers, extract_polarization,
    dct_unwrap, FIELD_SIZE,
)

sys.path.insert(0, str(HERE))
from zernike_utils import (
    build_zernike_basis, fit_zernike_lstsq,
    reconstruct_from_coeffs, N_MODES, ZERNIKE_NAMES,
)

# ── Mirror file paths ─────────────────────────────────────────────────────────
MIRROR_FILES = [
    BEADS / "mirror_50fps_83us_1024x1024x20_5/mirror_50fps_83us_1024x1024x20_5_raw.bin",
    BEADS / "mirror_50fps_83us_1024x1024x20_6/mirror_50fps_83us_1024x1024x20_6_raw.bin",
    BEADS / "mirror_50fps_83us_1024x1024x500_5/mirror_50fps_83us_1024x1024x500_5_raw.bin",
]


# ── Core processing ───────────────────────────────────────────────────────────

def process_mirror_file(path: Path, basis: np.ndarray, circ: np.ndarray):
    """
    Reconstruct phase from every frame in a mirror .bin file and fit Zernike.

    Returns
    -------
    coeffs_p1 : [N_frames, N_MODES]
    coeffs_p2 : [N_frames, N_MODES]
    noise_std  : float — mean residual std after Zernike fit (phase noise estimate)
    """
    print(f"  Loading {path.name} ...")
    volume   = read_frames(path)
    n_frames = volume.shape[0]
    print(f"    {n_frames} frames of {volume.shape[1]}×{volume.shape[2]}")

    extr_mask = make_extraction_mask(volume.shape[2])

    # Detect sidebands from first frame (assumed stable across frames)
    f0       = fftshift(fft2(volume[0].astype(np.float32)))
    fc1, fc2 = find_sideband_centers(f0)
    print(f"    Sidebands: P1={fc1}  P2={fc2}")

    coeffs_p1 = np.zeros((n_frames, N_MODES), dtype=np.float64)
    coeffs_p2 = np.zeros((n_frames, N_MODES), dtype=np.float64)
    residuals = []

    for fi in range(n_frames):
        f = fftshift(fft2(volume[fi].astype(np.float32)))
        p1 = extract_polarization(f, fc1, extr_mask)
        p2 = extract_polarization(f, fc2, extr_mask)

        for pol_idx, field in enumerate([p1, p2]):
            unwrapped = dct_unwrap(np.angle(field))
            coeffs    = fit_zernike_lstsq(unwrapped, circ, basis)
            recon     = reconstruct_from_coeffs(coeffs, basis, circ)
            residual  = unwrapped - recon
            residuals.append(np.std(residual[circ]))

            if pol_idx == 0:
                coeffs_p1[fi] = coeffs
            else:
                coeffs_p2[fi] = coeffs

        print(f"\r    Frame {fi+1}/{n_frames}", end="", flush=True)

    print()
    noise_std = float(np.mean(residuals))
    print(f"    Mean residual std (noise): {noise_std:.4f} rad")
    return coeffs_p1, coeffs_p2, noise_std


def compute_stats(coeffs: np.ndarray, n_sigma: float = 3.0) -> dict:
    """Per-mode statistics for a [N_samples, N_modes] coefficient array."""
    mu   = coeffs.mean(axis=0)
    sigma = coeffs.std(axis=0)
    return dict(
        mean=mu,
        std=sigma,
        min=coeffs.min(axis=0),
        max=coeffs.max(axis=0),
        low=mu - n_sigma * sigma,
        high=mu + n_sigma * sigma,
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_histograms(all_coeffs: np.ndarray, stats: dict):
    """15 per-mode histograms with μ ± 3σ range marked."""
    ncols, nrows = 5, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
    fig.suptitle(
        f"Zernike coefficient distributions — {all_coeffs.shape[0]} phase maps "
        f"(P1+P2 pooled)",
        fontsize=11,
    )

    for j in range(N_MODES):
        ax   = axes[j // ncols, j % ncols]
        vals = all_coeffs[:, j]
        ax.hist(vals, bins=40, color="steelblue", alpha=0.8, edgecolor="none")
        ax.axvline(stats["mean"][j], color="k",      lw=1.5, label="μ")
        ax.axvline(stats["low"][j],  color="tomato", lw=1.2, ls="--", label="μ±3σ")
        ax.axvline(stats["high"][j], color="tomato", lw=1.2, ls="--")
        ax.set_title(f"Z{j}: {ZERNIKE_NAMES[j]}", fontsize=8)
        ax.set_xlabel("coefficient (rad)", fontsize=7)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=6)

    fig.tight_layout()
    path = OUT / "calibration_histograms.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_examples(basis: np.ndarray, circ: np.ndarray,
                  all_coeffs: np.ndarray, n_examples: int = 6):
    """Random example reconstructed aberration surfaces."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(all_coeffs), n_examples, replace=False)

    fig, axes = plt.subplots(2, n_examples // 2, figsize=(n_examples // 2 * 3.5, 7))
    fig.suptitle("Example reconstructed aberration surfaces (from mirror data)", fontsize=11)

    axes = axes.ravel()
    for i, sample_idx in enumerate(idx):
        surface = reconstruct_from_coeffs(all_coeffs[sample_idx], basis, circ)
        data    = surface.copy().astype(float)
        data[~circ] = np.nan
        vabs = np.nanpercentile(np.abs(data), 98)
        im = axes[i].imshow(data, cmap="RdBu_r", origin="upper",
                             vmin=-vabs, vmax=vabs)
        axes[i].set_title(f"Sample {sample_idx}", fontsize=9)
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04,
                     label="rad")

    fig.tight_layout()
    path = OUT / "calibration_examples.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_mean_aberration(basis: np.ndarray, circ: np.ndarray,
                         coeffs_p1: np.ndarray, coeffs_p2: np.ndarray):
    """Mean aberration surface for P1, P2, and pooled."""
    mean_p1     = reconstruct_from_coeffs(coeffs_p1.mean(axis=0), basis, circ)
    mean_p2     = reconstruct_from_coeffs(coeffs_p2.mean(axis=0), basis, circ)
    mean_pooled = reconstruct_from_coeffs(
        np.concatenate([coeffs_p1, coeffs_p2]).mean(axis=0), basis, circ
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Mean aberration surface from mirror data", fontsize=11)

    for ax, surface, title in zip(
        axes,
        [mean_p1, mean_p2, mean_pooled],
        ["P1 mean", "P2 mean", "Pooled mean"],
    ):
        data = surface.copy().astype(float)
        data[~circ] = np.nan
        vabs = np.nanpercentile(np.abs(data), 99)
        im = ax.imshow(data, cmap="RdBu_r", origin="upper",
                       vmin=-vabs, vmax=vabs)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="rad")

    fig.tight_layout()
    path = OUT / "calibration_mean_aber.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_stats_bar(stats_p1: dict, stats_p2: dict, stats_pooled: dict):
    """Bar chart of per-mode |μ| and σ for a quick overview."""
    x = np.arange(N_MODES)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Per-mode Zernike statistics from mirror calibration", fontsize=11)

    for ax, stats, title in zip(
        axes,
        [stats_p1, stats_p2],
        ["P1", "P2"],
    ):
        ax.bar(x, np.abs(stats["mean"]), color="steelblue", width=0.4,
               label="|μ|", align="center")
        ax.bar(x + 0.4, stats["std"], color="tomato", width=0.4,
               label="σ", align="center")
        ax.set_xticks(x + 0.2)
        ax.set_xticklabels(
            [f"Z{j}\n{ZERNIKE_NAMES[j][:8]}" for j in range(N_MODES)],
            fontsize=6, rotation=45, ha="right",
        )
        ax.set_ylabel("|coefficient| (rad)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = OUT / "calibration_stats_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Building Zernike basis ({N_MODES} modes, {FIELD_SIZE}×{FIELD_SIZE}) ...")
    basis, _, circ = build_zernike_basis(N_MODES, FIELD_SIZE, FIELD_SIZE)

    all_p1, all_p2, noise_stds = [], [], []

    for fpath in MIRROR_FILES:
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping.")
            continue
        cp1, cp2, ns = process_mirror_file(fpath, basis, circ)
        all_p1.append(cp1)
        all_p2.append(cp2)
        noise_stds.append(ns)

    coeffs_p1 = np.concatenate(all_p1, axis=0)   # [N_total, N_MODES]
    coeffs_p2 = np.concatenate(all_p2, axis=0)
    all_coeffs = np.concatenate([coeffs_p1, coeffs_p2], axis=0)

    n_total = coeffs_p1.shape[0]
    print(f"\nTotal frames: {n_total} per polarisation  ({2*n_total} phase maps)")

    # ── Statistics ────────────────────────────────────────────────────────────
    stats_p1     = compute_stats(coeffs_p1)
    stats_p2     = compute_stats(coeffs_p2)
    stats_pooled = compute_stats(all_coeffs)
    noise_std    = float(np.mean(noise_stds))

    print(f"\nPhase noise std (residual after Zernike fit): {noise_std:.4f} rad")
    print(f"\nPooled per-mode statistics (rad):")
    print(f"  {'Mode':<20} {'μ':>8} {'σ':>8} {'low(μ-3σ)':>12} {'high(μ+3σ)':>12}")
    for j in range(N_MODES):
        print(
            f"  Z{j:<2} {ZERNIKE_NAMES[j]:<16} "
            f"{stats_pooled['mean'][j]:>8.3f} "
            f"{stats_pooled['std'][j]:>8.3f} "
            f"{stats_pooled['low'][j]:>12.3f} "
            f"{stats_pooled['high'][j]:>12.3f}"
        )

    # ── Save stats ────────────────────────────────────────────────────────────
    np.savez(
        OUT / "zernike_stats.npz",
        # pooled (used for simulation sampling)
        mean_pooled=stats_pooled["mean"],
        std_pooled=stats_pooled["std"],
        low_pooled=stats_pooled["low"],
        high_pooled=stats_pooled["high"],
        # per-polarisation (diagnostic)
        mean_p1=stats_p1["mean"],  std_p1=stats_p1["std"],
        mean_p2=stats_p2["mean"],  std_p2=stats_p2["std"],
        # raw coefficient arrays
        coeffs_p1=coeffs_p1,
        coeffs_p2=coeffs_p2,
        # noise estimate
        noise_std=np.array(noise_std),
    )
    print(f"\n  Saved calibration_stats/zernike_stats.npz")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_histograms(all_coeffs, stats_pooled)
    plot_examples(basis, circ, all_coeffs)
    plot_mean_aberration(basis, circ, coeffs_p1, coeffs_p2)
    plot_stats_bar(stats_p1, stats_p2, stats_pooled)

    print(f"\nCalibration complete. All outputs in: {OUT}")


if __name__ == "__main__":
    main()
