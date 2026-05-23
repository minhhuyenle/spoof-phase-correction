#!/usr/bin/env python3
"""
A549 Cell SPoOF-OCM Pipeline

Data flow (mirrors friend's SPoOF_BatchAnalyze.m, adapted for raw uint16 input):

  Step 1 – Read raw uint16 frames from .bin file
  Step 2 – FFT each frame, detect the two polarization sideband lobes
  Step 3 – Extract each polarization: shift lobe to DC, crop to N/2×N/2, IFFT
           → produces two complex fields (P1, P2) per frame
  Step 4 – Auto-fit quadratic phase mask to each channel via least-squares
           on the unwrapped phase, then apply correction (same form as friend's code)

Usage:
  python3 cell_pipeline.py
  python3 cell_pipeline.py --data 20260421/A549_cell_1_83us_1024x1024x20_6/...bin
  python3 cell_pipeline.py --frame 5       # reference frame for fitting
  python3 cell_pipeline.py --avg-frames    # average all frames before fitting

Outputs saved to:  20260421/A549_cell_1_83us_1024x1024x20_6/python_results/
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.fft import dctn, idctn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── defaults ─────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
# DATA_PATH = BASE / "20260421/A549_cell_21_83us_1024x1024x20_8/A549_cell_21_83us_1024x1024x20_8_raw.bin"
#DATA_PATH = BASE / "20260424/A549_cell_21_83us_1024x1024x20_8/A549_cell_21_83us_1024x1024x20_8_raw.bin"
DATA_PATH = BASE / "USAF/USAF_Resolution_int_83us_1024x1024x20_7/USAF_Resolution_int_83us_1024x1024x20_7_raw.bin"

WIDTH  = 1024
HEIGHT = 1024
DTYPE  = np.uint16

# Extracted complex field size after cropping (N//2 × N//2 = 512×512)
FIELD_SIZE = WIDTH // 2

# Cosine-rolloff mask radius (matches notebook: radius=150, cos_ratio=0.9)
MASK_RADIUS    = 150
MASK_COS_RATIO = 0.9
DC_BLOCK_R     = 150   # radius to suppress DC when finding lobe centers


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Read raw data
# ═══════════════════════════════════════════════════════════════════════════════

def read_frames(path: Path) -> np.ndarray:
    """Load all frames as (N_FRAMES, HEIGHT, WIDTH) uint16."""
    raw = np.fromfile(path, dtype=DTYPE)
    n_frames = raw.size // (HEIGHT * WIDTH)
    return raw.reshape((n_frames, HEIGHT, WIDTH))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 – FFT + lobe detection
# ═══════════════════════════════════════════════════════════════════════════════

def make_extraction_mask(N: int) -> np.ndarray:
    """
    Circular mask with cosine rolloff (matches notebook).
    Used to window each extracted sideband before IFFT.
    """
    y, x = np.ogrid[-N/2:N/2, -N/2:N/2]
    dist  = np.sqrt(x**2 + y**2)
    mask  = np.zeros((N, N), dtype=np.float32)
    mask[dist <= MASK_COS_RATIO * MASK_RADIUS] = 1.0
    trans = (dist > MASK_COS_RATIO * MASK_RADIUS) & (dist < MASK_RADIUS)
    mask[trans] = np.cos(
        (dist[trans] - MASK_COS_RATIO * MASK_RADIUS)
        / ((1 - MASK_COS_RATIO) * MASK_RADIUS) * np.pi * 0.5
    )
    return mask


def find_sideband_centers(f_data: np.ndarray, dc_block: int = DC_BLOCK_R):
    """
    Find the two brightest lobes in the upper half of the FFT spectrum
    (top-left quadrant = P1, top-right quadrant = P2).
    Mirrors notebook find_sideband_centers().
    Returns (fc1, fc2) as (row, col) tuples.
    """
    N   = f_data.shape[0]
    mag = np.abs(f_data).copy()

    # Suppress DC region
    y, x     = np.ogrid[-N/2:N/2, -N/2:N/2]
    mag[x**2 + y**2 < dc_block**2] = 0

    # Top-left quadrant → P1
    tl          = mag[0:N//2, 0:N//2]
    y_tl, x_tl = np.unravel_index(np.argmax(tl), tl.shape)

    # Top-right quadrant → P2
    tr          = mag[0:N//2, N//2:N]
    y_tr, x_tr  = np.unravel_index(np.argmax(tr), tr.shape)
    x_tr       += N // 2

    return (y_tl, x_tl), (y_tr, x_tr)


def extract_polarization(f_data: np.ndarray, fc: tuple, mask: np.ndarray) -> np.ndarray:
    """
    Shift the sideband at fc to the DC centre, crop to [N/4 : 3N/4], apply mask, IFFT.
    Returns a (N/2, N/2) complex64 field. Mirrors notebook extract_state().
    """
    N       = f_data.shape[0]
    shift_y = int(N / 2 - fc[0])
    shift_x = int(N / 2 - fc[1])

    shifted      = np.roll(np.roll(f_data, shift_y, axis=0), shift_x, axis=1)
    cropped      = shifted[N//4 : 3*N//4, N//4 : 3*N//4]
    mask_cropped = mask  [N//4 : 3*N//4, N//4 : 3*N//4]

    return ifft2(ifftshift(cropped * mask_cropped)).astype(np.complex64)


def plot_fft_and_lobes(frame: np.ndarray, f_data: np.ndarray,
                        fc1: tuple, fc2: tuple, out_dir: Path):
    """Step 2 diagnostic: raw frame + FFT spectrum with detected lobe centers."""
    N   = frame.shape[0]
    mag = np.log1p(np.abs(f_data))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Step 2 – FFT and sideband lobe detection", fontsize=13)

    # Raw frame
    vmin, vmax = np.percentile(frame, (0.5, 99.5))
    axes[0].imshow(frame, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    axes[0].set_title("Raw frame (uint16)")
    axes[0].axis("off")

    # FFT spectrum
    axes[1].imshow(mag, cmap="inferno", origin="upper")
    axes[1].set_title("FFT magnitude (log scale)")
    axes[1].axis("off")

    # Mark detected centers
    axes[1].scatter([fc1[1], fc2[1]], [fc1[0], fc2[0]],
                    color="cyan", marker="+", s=200, linewidths=2, zorder=5,
                    label=f"P1 ({fc1[1]},{fc1[0]})  P2 ({fc2[1]},{fc2[0]})")

    for fc, col in [(fc1, "cyan"), (fc2, "lime")]:
        circ = plt.Circle((fc[1], fc[0]), MASK_RADIUS,
                           color=col, fill=False, lw=1.5, linestyle="--")
        axes[1].add_patch(circ)

    # DC block circle
    dc_circ = plt.Circle((N/2, N/2), DC_BLOCK_R,
                          color="orange", fill=False, lw=1.2, linestyle=":",
                          label="DC block")
    axes[1].add_patch(dc_circ)
    axes[1].legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / "step2_fft_lobes.png", dpi=150)
    plt.close(fig)
    print("  Saved step2_fft_lobes.png")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 – Extracted complex fields visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def plot_extracted_fields(p1: np.ndarray, p2: np.ndarray, out_dir: Path):
    """Step 3: amplitude and raw phase of the two extracted polarization fields."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Step 3 – Extracted polarization fields (raw, before phase correction)", fontsize=12)

    panels = [
        (axes[0, 0], np.abs(p1),    "P1 amplitude",   "viridis"),
        (axes[0, 1], np.abs(p2),    "P2 amplitude",   "viridis"),
        (axes[1, 0], np.angle(p1),  "P1 phase (rad)", "twilight"),
        (axes[1, 1], np.angle(p2),  "P2 phase (rad)", "twilight"),
    ]
    for ax, data, title, cmap in panels:
        clim = (-np.pi, np.pi) if "phase" in title else None
        im = ax.imshow(data, cmap=cmap, origin="upper",
                       vmin=clim[0] if clim else None,
                       vmax=clim[1] if clim else None)
        ax.set_title(title); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_dir / "step3_extracted_fields.png", dpi=150)
    plt.close(fig)
    print("  Saved step3_extracted_fields.png")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 – Auto-fit quadratic phase mask + correction
# ═══════════════════════════════════════════════════════════════════════════════

def dct_unwrap(wrapped: np.ndarray) -> np.ndarray:
    """
    DCT-based least-squares phase unwrapping (Ghiglia & Romero, 1994).
    Removes 2π jumps by solving the Poisson equation in DCT space.
    Copied from user's reconstruction_cell.ipynb.
    """
    Ny, Nx = wrapped.shape
    dx = np.angle(np.exp(1j * np.diff(wrapped, axis=1)))
    dy = np.angle(np.exp(1j * np.diff(wrapped, axis=0)))

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
    phi_dct     = rho_dct / denom
    phi_dct[0, 0] = 0.0
    return idctn(phi_dct, type=2, norm="ortho")


def fit_quadratic_mask(field: np.ndarray, circ_mask: np.ndarray) -> np.ndarray:
    """
    Fit the same 6-term quadratic used in friend's SPoOF_BatchAnalyze.m:
        mask = y1·y + x1·x + y2·y² + x2·x² + xyc·x·y + off
    by least-squares on the unwrapped phase inside the circular aperture.

    Returns the fitted mask surface (same shape as field, float32).
    Also prints the fitted coefficients in the same order as friend's PM dict.
    """
    size = field.shape[0]
    ax   = np.linspace(-1, 1, size, dtype=np.float64)
    x, y = np.meshgrid(ax, ax)

    # Unwrap phase
    unwrapped = dct_unwrap(np.angle(field))

    # Build design matrix for pixels inside the circular aperture
    pts   = circ_mask.ravel()
    A     = np.column_stack([
        y.ravel()[pts],          # y1
        x.ravel()[pts],          # x1
        (y**2).ravel()[pts],     # y2
        (x**2).ravel()[pts],     # x2
        (x*y).ravel()[pts],      # xyc
        np.ones(pts.sum()),      # off
    ])
    b = unwrapped.ravel()[pts]

    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    y1, x1, y2, x2, xyc, off = coeffs

    print(f"    Fitted: y1={y1:.3f}  x1={x1:.3f}  y2={y2:.3f}  "
          f"x2={x2:.3f}  xyc={xyc:.3f}  off={off:.3f}")

    mask = (y1*y + x1*x + y2*y**2 + x2*x**2 + xyc*x*y + off).astype(np.float32)
    return mask, dict(y1=y1, x1=x1, y2=y2, x2=x2, xyc=xyc, off=off)


def apply_phase_correction(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """corrected = exp(-i·mask) × field"""
    return (np.exp(-1j * mask) * field).astype(np.complex64)


def plot_phase_correction(p1_raw, p2_raw, p1_corr, p2_corr,
                           mask1, mask2, coeffs1, coeffs2, out_dir: Path):
    """Step 4: raw phase → fitted mask → corrected phase → difference."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Step 4 – Auto-fitted quadratic phase correction", fontsize=13)

    rows = [
        (p1_raw, mask1, p1_corr, "P1", coeffs1),
        (p2_raw, mask2, p2_corr, "P2", coeffs2),
    ]
    for ri, (raw, mask, corr, label, coeffs) in enumerate(rows):
        diff = np.angle(raw) - np.angle(corr)
        subtitle = (f"y1={coeffs['y1']:.1f}  x1={coeffs['x1']:.1f}  "
                    f"y2={coeffs['y2']:.1f}  x2={coeffs['x2']:.1f}  "
                    f"xyc={coeffs['xyc']:.1f}  off={coeffs['off']:.2f}")
        panels = [
            (axes[ri, 0], np.angle(raw),  f"{label} raw phase (rad)",       "twilight", (-np.pi, np.pi)),
            (axes[ri, 1], mask,            f"{label} fitted mask (rad)",     "RdBu_r",   (mask.min(), mask.max())),
            (axes[ri, 2], np.angle(corr),  f"{label} corrected phase (rad)", "twilight", (-np.pi, np.pi)),
            (axes[ri, 3], diff,            "Difference (raw − corrected)",   "RdBu_r",   None),
        ]
        for ci, (ax, data, title, cmap, clim) in enumerate(panels):
            im = ax.imshow(data, cmap=cmap, origin="upper",
                           vmin=clim[0] if clim else None,
                           vmax=clim[1] if clim else None)
            ax.set_title(title, fontsize=9); ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Print fitted coefficients under the row
        axes[ri, 0].set_xlabel(subtitle, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "step4_phase_correction.png", dpi=150)
    plt.close(fig)
    print("  Saved step4_phase_correction.png")


def plot_all_frames(volumes_p1, volumes_p2, masks, out_dir: Path):
    """
    Summary grid: one row per frame, columns = P1 amp / P1 corrected phase / P2 amp / P2 corrected phase.
    """
    mask1, mask2 = masks
    n = len(volumes_p1)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3 * n))
    fig.suptitle("All frames — P1 amp | P1 corrected phase | P2 amp | P2 corrected phase", fontsize=12)

    col_titles = ["P1 amplitude", "P1 corrected phase", "P2 amplitude", "P2 corrected phase"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9)

    for fi, (p1, p2) in enumerate(zip(volumes_p1, volumes_p2)):
        p1c = apply_phase_correction(p1, mask1)
        p2c = apply_phase_correction(p2, mask2)
        row_data = [
            (np.abs(p1),          "viridis",  None),
            (np.angle(p1c),       "twilight", (-np.pi, np.pi)),
            (np.abs(p2),          "viridis",  None),
            (np.angle(p2c),       "twilight", (-np.pi, np.pi)),
        ]
        for col, (data, cmap, clim) in enumerate(row_data):
            ax = axes[fi, col]
            ax.imshow(data, cmap=cmap, origin="upper",
                      vmin=clim[0] if clim else None,
                      vmax=clim[1] if clim else None)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"Frame {fi}", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "all_frames_summary.png", dpi=120)
    plt.close(fig)
    print("  Saved all_frames_summary.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="A549 cell SPoOF-OCM pipeline")
    parser.add_argument("--data",       type=Path, default=DATA_PATH, help="Path to _raw.bin file")
    parser.add_argument("--frame",      type=int,  default=0,         help="Reference frame for fitting")
    parser.add_argument("--avg-frames", action="store_true",
                        help="Average all frames before fitting the phase mask (better SNR)")
    args = parser.parse_args()

    out_dir = args.data.parent / "python_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: read ──────────────────────────────────────────────────────────
    print(f"\n[Step 1] Reading {args.data.name} ...")
    volume = read_frames(args.data)
    n_frames, H, W = volume.shape
    print(f"  Loaded {n_frames} frames of {H}×{W} uint16")

    # ── Step 2: FFT + lobe detection on reference frame ──────────────────────
    print(f"\n[Step 2] FFT and sideband detection (reference = frame {args.frame})...")
    ref_frame = volume[args.frame].astype(np.float32)
    f_ref     = fftshift(fft2(ref_frame))
    fc1, fc2  = find_sideband_centers(f_ref)
    print(f"  P1 lobe center: row={fc1[0]}, col={fc1[1]}")
    print(f"  P2 lobe center: row={fc2[0]}, col={fc2[1]}")
    extr_mask = make_extraction_mask(W)
    plot_fft_and_lobes(ref_frame, f_ref, fc1, fc2, out_dir)

    # ── Step 3: extract polarization fields for all frames ───────────────────
    print(f"\n[Step 3] Extracting P1/P2 for all {n_frames} frames...")
    p1_stack, p2_stack = [], []
    for fi in range(n_frames):
        f_frame = fftshift(fft2(volume[fi].astype(np.float32)))
        p1_stack.append(extract_polarization(f_frame, fc1, extr_mask))
        p2_stack.append(extract_polarization(f_frame, fc2, extr_mask))
        print(f"\r  Frame {fi+1}/{n_frames}", end="", flush=True)
    print()
    plot_extracted_fields(p1_stack[args.frame], p2_stack[args.frame], out_dir)

    # ── Step 4: auto-fit quadratic phase mask ─────────────────────────────────
    print(f"\n[Step 4] Fitting quadratic phase mask...")

    # Build circular aperture mask (same as friend's Win = RHO^2 <= 1)
    ax = np.linspace(-1, 1, FIELD_SIZE)
    xg, yg = np.meshgrid(ax, ax)
    circ = (xg**2 + yg**2) <= 1.0

    if args.avg_frames:
        # Average complex fields across all frames before fitting — reduces noise
        p1_fit = np.mean(np.stack(p1_stack), axis=0)
        p2_fit = np.mean(np.stack(p2_stack), axis=0)
        print("  Fitting on average of all frames")
    else:
        p1_fit = p1_stack[args.frame]
        p2_fit = p2_stack[args.frame]
        print(f"  Fitting on frame {args.frame}")

    print("  P1:")
    mask1, coeffs1 = fit_quadratic_mask(p1_fit, circ)
    print("  P2:")
    mask2, coeffs2 = fit_quadratic_mask(p2_fit, circ)

    p1_ref  = p1_stack[args.frame]
    p2_ref  = p2_stack[args.frame]
    p1_corr = apply_phase_correction(p1_ref, mask1)
    p2_corr = apply_phase_correction(p2_ref, mask2)

    plot_phase_correction(p1_ref, p2_ref, p1_corr, p2_corr,
                          mask1, mask2, coeffs1, coeffs2, out_dir)

    # ── Summary: all frames ───────────────────────────────────────────────────
    print(f"\n[Summary] Generating all-frames grid...")
    plot_all_frames(p1_stack, p2_stack, (mask1, mask2), out_dir)

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
