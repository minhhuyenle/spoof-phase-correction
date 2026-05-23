#!/usr/bin/env python3
"""
SPoOF-OCM Full Analysis Pipeline

Replicates the MATLAB SPoOF_BatchAnalyze.m pipeline in Python:
  Step 1  – Quadratic phase correction (compensates optical aberrations)
  Step 2  – Retardation angle: atan2(|i·T2−T1|, |i·T2+T1|) in degrees
  Step 3  – Temporal filtering: Gaussian smooth + notch at 110-130 Hz and 7-10 Hz
  Step 4  – ROI grid (4×4 px, gap=5) → time series per ROI
  Step 5  – PCA (10 components) + k-means clustering (k=5)
  Step 6  – Static PNG outputs for every step
  Step 7  – (--video) 6-panel MP4

Usage:
  python3 spoof_pipeline.py                    # full pipeline, all frames
  python3 spoof_pipeline.py --frames 5000      # quick test on first 5000 frames
  python3 spoof_pipeline.py --video            # also produce 6-panel MP4
  python3 spoof_pipeline.py --load             # skip extraction, load saved results
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

# ── paths ────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent / "D1_1000fps_512x512_C001H001S0001"
P1_PATH  = BASE / "Processed_P1.bin"
P2_PATH  = BASE / "Processed_P2.bin"
OUT_DIR  = BASE / "python_results"

# ── constants (from SPoOF_BatchAnalyze.m) ───────────────────────────────────
N_FRAMES = 43674
H = W    = 200       # ROI field size
FS       = 1000.0    # frames per second
ABS_WIN  = 4         # ROI box side (pixels), mean taken over ABS_WIN+1 × ABS_WIN+1
GAP      = 5         # spacing between ROI centres
BATCH    = 500       # frames per processing batch

# Phase mask 1 parameters
PM1 = dict(y1=-4.0,  y2=35.0,  x1=0.0,  x2=30.0,  xyc=-1.5, ycen=0, xcen=0, off=0.0)
# Phase mask 2 parameters
PM2 = dict(y1=2.0,   y2=-34.0, x1=1.3,  x2=-30.0, xyc=-4.0, ycen=0, xcen=0, off=0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Phase masks
# ═══════════════════════════════════════════════════════════════════════════════

def make_phase_mask(pm: dict) -> np.ndarray:
    """Quadratic phase mask on a [-1,1]×[-1,1] grid (H×W float32)."""
    x_ax = np.linspace(-1, 1, W, dtype=np.float32)
    y_ax = np.linspace(-1, 1, H, dtype=np.float32)
    x, y = np.meshgrid(x_ax, y_ax)
    return (pm["y1"] * y + pm["x1"] * x
            + pm["y2"] * (y - pm["ycen"]) ** 2
            + pm["x2"] * (x - pm["xcen"]) ** 2
            + pm["xyc"] * x * y + pm["off"]).astype(np.float32)


def circular_window() -> np.ndarray:
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(x, y)
    return (X ** 2 + Y ** 2 <= 1).astype(np.float32)


def plot_phase_masks(mask1, mask2, win, out_dir):
    # Load one real frame to show before/after correction
    frame_p1 = load_batch(P1_PATH, 0, 1)[0]   # (H, W) complex64
    frame_p2 = load_batch(P2_PATH, 0, 1)[0]

    raw_phase1  = np.angle(frame_p1)
    corr_phase1 = np.angle(np.exp(-1j * mask1) * frame_p1)
    raw_phase2  = np.angle(frame_p2)
    corr_phase2 = np.angle(np.exp(-1j * mask2) * frame_p2)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Step 1 – Phase masks and their effect on frame 0", fontsize=13)

    panels = [
        # row 0: P1
        (axes[0, 0], raw_phase1,  "P1 raw phase (rad)",        "twilight", (-np.pi, np.pi)),
        (axes[0, 1], mask1,       "P1 correction mask (rad)",  "RdBu_r",   (mask1.min(), mask1.max())),
        (axes[0, 2], corr_phase1, "P1 corrected phase (rad)",  "twilight", (-np.pi, np.pi)),
        (axes[0, 3], raw_phase1 - corr_phase1,
                                  "Difference (raw − corrected)", "RdBu_r", None),
        # row 1: P2
        (axes[1, 0], raw_phase2,  "P2 raw phase (rad)",        "twilight", (-np.pi, np.pi)),
        (axes[1, 1], mask2,       "P2 correction mask (rad)",  "RdBu_r",   (mask2.min(), mask2.max())),
        (axes[1, 2], corr_phase2, "P2 corrected phase (rad)",  "twilight", (-np.pi, np.pi)),
        (axes[1, 3], raw_phase2 - corr_phase2,
                                  "Difference (raw − corrected)", "RdBu_r", None),
    ]

    for ax, data, title, cmap, clim in panels:
        im = ax.imshow(data * win, cmap=cmap, origin="upper",
                       vmin=clim[0] if clim else None,
                       vmax=clim[1] if clim else None)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_dir / "step1_phase_masks.png", dpi=150)
    plt.close(fig)
    print("  Saved step1_phase_masks.png")


# ═══════════════════════════════════════════════════════════════════════════════
# ROI grid helper
# ═══════════════════════════════════════════════════════════════════════════════

def make_roi_grid():
    """
    Replicates MATLAB:
      [LocListX, LocListY] = meshgrid((3:5:197), (3:5:197))
      sorted by distance from (0,0), then XL = loc - AbsWin/2
    Returns XL, YL: 0-indexed top-left corners of each ROI box (int arrays).
    """
    # MATLAB 1-indexed 3:5:197 → Python 0-indexed 2:5:196
    locs = np.arange(ABS_WIN // 2 + 1 - 1, W - ABS_WIN // 2 - 1, GAP)  # 0-indexed centres
    XX, YY = np.meshgrid(locs, locs)          # X = col, Y = row (MATLAB convention)
    locs_xy = np.stack([XX.ravel(), YY.ravel()], axis=1)   # (N, 2)  [col, row]

    order = np.argsort(locs_xy[:, 0] ** 2 + locs_xy[:, 1] ** 2)
    locs_xy = locs_xy[order]

    XL = locs_xy[:, 0] - ABS_WIN // 2   # col start (0-indexed)
    YL = locs_xy[:, 1] - ABS_WIN // 2   # row start (0-indexed)
    return XL.astype(int), YL.astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 – Retardation extraction (frame-by-frame ROI means)
# ═══════════════════════════════════════════════════════════════════════════════

def load_batch(path: Path, start: int, count: int) -> np.ndarray:
    """Return (count, H, W) complex64 from binary file at frame `start`."""
    vals = H * W * 2
    offset = start * vals * 4          # float32 = 4 bytes
    raw = np.memmap(path, dtype=np.float32, mode="r",
                    offset=offset, shape=(count * vals,))
    return (raw[0::2] + 1j * raw[1::2]).reshape(count, H, W).astype(np.complex64)


def extract_roi_timeseries(n_frames: int) -> tuple:
    """
    Phase-correct every frame and compute complex mean over each 5×5 ROI.
    T1[roi, t] = mean(P1_corrected in roi)
    T2[roi, t] = conj(mean(P2_corrected in roi)) × exp(iπ/2)   ← MATLAB exact
    Returns T1, T2 (N_ROI × n_frames complex64), XL, YL.
    """
    mask1 = make_phase_mask(PM1)
    mask2 = make_phase_mask(PM2)
    e1 = np.exp(-1j * mask1).astype(np.complex64)   # (H, W)
    e2 = np.exp(-1j * mask2).astype(np.complex64)

    XL, YL = make_roi_grid()
    N_ROI = len(XL)
    T1 = np.zeros((N_ROI, n_frames), dtype=np.complex64)
    T2 = np.zeros((N_ROI, n_frames), dtype=np.complex64)

    done = 0
    while done < n_frames:
        b = min(BATCH, n_frames - done)
        p1 = load_batch(P1_PATH, done, b) * e1      # (b, H, W)
        p2 = load_batch(P2_PATH, done, b) * e2

        box = ABS_WIN + 1                            # 5 px side
        for i, (xl, yl) in enumerate(zip(XL, YL)):
            roi1 = p1[:, yl:yl + box, xl:xl + box]
            roi2 = p2[:, yl:yl + box, xl:xl + box]
            T1[i, done:done + b] = roi1.mean(axis=(1, 2))
            T2[i, done:done + b] = np.conj(roi2.mean(axis=(1, 2))) * np.exp(1j * np.pi / 2)

        done += b
        print(f"\r  Frame {done}/{n_frames} ({100*done/n_frames:.1f}%)", end="", flush=True)

    print()
    return T1, T2, XL, YL


def compute_retardation(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """Returns delta_s (n_frames × N_ROI) in degrees."""
    num = np.abs(1j * T2 - T1)
    den = np.abs(1j * T2 + T1)
    return np.degrees(np.arctan2(num, den)).T    # (n_frames, N_ROI)


def plot_retardation_sample(delta_s, XL, YL, out_dir):
    """Snapshot of retardation map at one time point + a few ROI time series."""
    n_frames, N_ROI = delta_s.shape
    taxis = np.arange(n_frames) / FS

    # Build spatial map from median retardation
    grid_n = int(np.round(np.sqrt(N_ROI)))
    ret_map = np.full((grid_n, grid_n), np.nan)
    gap = GAP
    for i in range(N_ROI):
        r = (YL[i] - YL.min()) // gap
        c = (XL[i] - XL.min()) // gap
        if r < grid_n and c < grid_n:
            ret_map[r, c] = np.median(delta_s[:, i])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Step 2 – Retardation angle atan2(|i·T2−T1|, |i·T2+T1|)", fontsize=12)

    im = axes[0].imshow(ret_map, cmap="inferno", origin="upper")
    axes[0].set_title("Median retardation (deg) — spatial map")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    for idx in range(0, min(8, N_ROI), max(1, N_ROI // 8)):
        axes[1].plot(taxis, delta_s[:, idx], alpha=0.7, linewidth=0.5, label=f"ROI {idx}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Retardation (deg)")
    axes[1].set_title("Sample ROI time series (raw)")
    axes[1].legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(out_dir / "step2_retardation_raw.png", dpi=150)
    plt.close(fig)
    print("  Saved step2_retardation_raw.png")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 – Temporal filtering
# ═══════════════════════════════════════════════════════════════════════════════

def apply_temporal_filters(delta_s: np.ndarray) -> np.ndarray:
    """
    1. Gaussian FIR smooth (window=10, alpha=2 → sigma≈2.25)
    2. Butterworth notch at 110–130 Hz  (order 4)
    3. Butterworth notch at 7–10 Hz     (order 2)
    4. Subtract mean over frames 2000–40000
    Returns filtered array same shape as input (n_frames × N_ROI).
    """
    n = delta_s.shape[0]
    sigma = (10 - 1) / (2 * 2.0)   # gausswin(10, alpha=2)
    b_g = signal.windows.gaussian(10, std=sigma)
    b_g /= b_g.sum()

    b1, a1 = signal.butter(4, [110, 130], btype="bandstop", fs=FS)
    b2, a2 = signal.butter(2, [7,   10],  btype="bandstop", fs=FS)

    out = signal.filtfilt(b_g, [1.0], delta_s, axis=0)
    out = signal.filtfilt(b1,  a1,    out,     axis=0)
    out = signal.filtfilt(b2,  a2,    out,     axis=0)

    lo, hi = min(2000, n), min(40000, n)
    out -= out[lo:hi].mean(axis=0, keepdims=True)
    return out


def plot_filter_comparison(delta_s_raw, delta_s_filt, roi_idx, out_dir):
    n = delta_s_raw.shape[0]
    taxis = np.arange(n) / FS
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)

    raw  = delta_s_raw[:,  roi_idx]
    filt = delta_s_filt[:, roi_idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    fig.suptitle(f"Step 3 – Temporal filtering (ROI {roi_idx})", fontsize=12)

    axes[0, 0].plot(taxis, raw,  lw=0.5, color="steelblue")
    axes[0, 0].set_title("Raw retardation (deg)"); axes[0, 0].set_xlabel("Time (s)")

    axes[0, 1].plot(taxis, filt, lw=0.5, color="tomato")
    axes[0, 1].set_title("Filtered (Gaussian + 110-130 Hz + 7-10 Hz notch)")
    axes[0, 1].set_xlabel("Time (s)")

    psd_raw  = np.abs(np.fft.rfft(raw))
    psd_filt = np.abs(np.fft.rfft(filt))
    axes[1, 0].semilogy(freqs, psd_raw,  lw=0.8, color="steelblue", label="raw")
    axes[1, 0].semilogy(freqs, psd_filt, lw=0.8, color="tomato",    label="filtered")
    axes[1, 0].axvspan(110, 130, alpha=0.2, color="red",  label="notch 110-130 Hz")
    axes[1, 0].axvspan(7,   10,  alpha=0.2, color="gold", label="notch 7-10 Hz")
    axes[1, 0].set_xlim(0, FS / 2); axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_title("Power spectrum"); axes[1, 0].legend(fontsize=8)

    axes[1, 1].axis("off")
    axes[1, 1].text(0.1, 0.6,
        "Filters applied (MATLAB-matched):\n"
        "  1. Gaussian FIR  (window=10, α=2)\n"
        "  2. Butterworth notch ord=4 → 110–130 Hz\n"
        "  3. Butterworth notch ord=2 → 7–10 Hz\n"
        "  4. Mean-subtract (frames 2000–40000)",
        transform=axes[1, 1].transAxes, fontsize=11,
        verticalalignment="top", family="monospace")

    fig.tight_layout()
    fig.savefig(out_dir / "step3_temporal_filters.png", dpi=150)
    plt.close(fig)
    print("  Saved step3_temporal_filters.png")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 – ROI grid visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def plot_roi_grid(XL, YL, out_dir):
    """Show where the ROI boxes sit on the 200×200 field."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"Step 4 – ROI grid ({len(XL)} ROIs, {ABS_WIN+1}×{ABS_WIN+1}px, gap={GAP}px)",
                 color="white")
    ax.tick_params(colors="white"); ax.spines[:].set_color("white")

    cmap = plt.cm.plasma
    N = len(XL)
    for i, (xl, yl) in enumerate(zip(XL, YL)):
        c = cmap(i / N)
        rect = plt.Rectangle((xl, yl), ABS_WIN + 1, ABS_WIN + 1,
                              linewidth=0.5, edgecolor=c, facecolor="none")
        ax.add_patch(rect)

    fig.tight_layout()
    fig.savefig(out_dir / "step4_roi_grid.png", dpi=150)
    plt.close(fig)
    print("  Saved step4_roi_grid.png")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 – PCA + k-means clustering
# ═══════════════════════════════════════════════════════════════════════════════

def run_pca_kmeans(T3: np.ndarray):
    """
    T3: (n_frames, N_ROI) filtered retardation.
    Returns: recon (n_frames, N_ROI), labels (N_ROI,), T3_mm (moving-mean smoothed).
    """
    # Moving mean (window=25)
    kernel = np.ones(25) / 25
    T3_mm = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), 0, T3)

    # Zero out pathological ROIs (variance > 40)
    bad = np.var(T3_mm, axis=0) > 40
    T3_mm[:, bad] = 0.0

    # PCA: fit on (N_ROI, n_frames), keep 10 components
    n_pca = min(10, T3_mm.shape[1] - 1)
    pca = PCA(n_components=n_pca)
    scores = pca.fit_transform(T3_mm.T)           # (N_ROI, n_pca)
    recon  = pca.inverse_transform(scores).T       # (n_frames, N_ROI)

    # K-means, repeat until all inter-cluster |corr| < 0.9
    n_clusters = 5
    for it in range(30):
        km = KMeans(n_clusters=n_clusters, n_init=10, max_iter=100, random_state=it)
        labels = km.fit_predict(recon.T)           # cluster each ROI's time series

        means = np.stack([recon[:, labels == k].mean(axis=1)
                          if (labels == k).any() else np.zeros(recon.shape[0])
                          for k in range(n_clusters)], axis=1)

        corr   = np.corrcoef(means.T)
        lo_tri = corr[np.tril_indices(n_clusters, k=-1)]
        max_cc = np.abs(lo_tri).max() if len(lo_tri) > 0 else 0.0
        print(f"  K-means iter {it+1}: max inter-cluster |r| = {max_cc:.3f}")
        if max_cc < 0.9:
            break

    return recon, labels, T3_mm


def plot_clustering(recon, labels, T3_mm, XL, YL, out_dir):
    n_frames, N_ROI = recon.shape
    taxis = np.arange(n_frames) / FS
    n_clusters = int(labels.max()) + 1
    cluster_colors = plt.cm.tab10(np.linspace(0, 0.9, n_clusters))

    # Spatial cluster map
    grid_n = int(np.round(np.sqrt(N_ROI)))
    gap    = GAP
    cmap_img = np.zeros((grid_n, grid_n))
    for i in range(N_ROI):
        r = (YL[i] - YL.min()) // gap
        c = (XL[i] - XL.min()) // gap
        if r < grid_n and c < grid_n:
            cmap_img[r, c] = labels[i] + 1

    # Cluster mean time series
    means = np.stack([recon[:, labels == k].mean(axis=1)
                      if (labels == k).any() else np.zeros(n_frames)
                      for k in range(n_clusters)], axis=1)

    # Inter-cluster correlation
    corr = np.corrcoef(means.T)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Step 5 – PCA (10 components) + K-means clustering (k=5)", fontsize=13)
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Spatial map
    ax0 = fig.add_subplot(gs[0, 0])
    cmap_disc = mcolors.ListedColormap(["#111111"] + [cluster_colors[k] for k in range(n_clusters)])
    im = ax0.imshow(cmap_img, cmap=cmap_disc, vmin=0, vmax=n_clusters, origin="upper")
    ax0.set_title("Cluster spatial map"); ax0.axis("off")

    # Cluster mean time series
    ax1 = fig.add_subplot(gs[0, 1:])
    for k in range(n_clusters):
        members = np.where(labels == k)[0]
        for m in members[:30]:          # light traces for each member
            ax1.plot(taxis, T3_mm[:, m] + k * 0.8, color=cluster_colors[k], alpha=0.05, lw=0.5)
        ax1.plot(taxis, means[:, k] + k * 0.8,
                 color=cluster_colors[k], lw=1.5, label=f"Cluster {k+1} (n={np.sum(labels==k)})")
    ax1.set_xlabel("Time (s)"); ax1.set_title("Cluster mean time series (+ individual traces)")
    ax1.legend(fontsize=8); ax1.set_yticks([])

    # Inter-cluster correlation matrix
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
    ax2.set_title("Inter-cluster correlation"); ax2.set_xticks(range(n_clusters))
    ax2.set_yticks(range(n_clusters))
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    for i in range(n_clusters):
        for j in range(n_clusters):
            ax2.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=8)

    # PCA-reconstructed vs. raw for one ROI
    sample_roi = N_ROI // 2
    ax3 = fig.add_subplot(gs[1, 1:])
    ax3.plot(taxis, T3_mm[:, sample_roi], lw=0.7, alpha=0.6, color="gray", label="Moving-mean filtered")
    ax3.plot(taxis, recon[:, sample_roi],  lw=1.0, color="tomato",          label="PCA reconstructed (10 PCs)")
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Retardation (deg)")
    ax3.set_title(f"PCA reconstruction vs. input (ROI {sample_roi})")
    ax3.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "step5_clustering.png", dpi=150)
    plt.close(fig)
    print("  Saved step5_clustering.png")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 – Combined summary figure
# ═══════════════════════════════════════════════════════════════════════════════

def plot_summary(delta_s_raw, delta_s_filt, recon, labels, XL, YL, out_dir):
    n, N_ROI = delta_s_raw.shape
    taxis   = np.arange(n) / FS
    gap     = GAP
    grid_n  = int(np.round(np.sqrt(N_ROI)))
    n_cls   = int(labels.max()) + 1
    colors  = plt.cm.tab10(np.linspace(0, 0.9, n_cls))

    def to_grid(vals):
        g = np.full((grid_n, grid_n), np.nan)
        for i in range(N_ROI):
            r = (YL[i] - YL.min()) // gap
            c = (XL[i] - XL.min()) // gap
            if r < grid_n and c < grid_n:
                g[r, c] = vals[i]
        return g

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("SPoOF-OCM Pipeline Summary", fontsize=14)

    # Raw retardation map (median)
    im0 = axes[0, 0].imshow(to_grid(np.median(delta_s_raw, axis=0)),
                             cmap="inferno", origin="upper")
    axes[0, 0].set_title("Median retardation (raw, deg)"); axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Filtered retardation map (std)
    im1 = axes[0, 1].imshow(to_grid(np.std(delta_s_filt, axis=0)),
                             cmap="hot", origin="upper")
    axes[0, 1].set_title("Std of filtered retardation"); axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Cluster spatial map
    cmap_img = to_grid(labels + 1)
    cmap_img[np.isnan(cmap_img)] = 0
    cmap_disc = mcolors.ListedColormap(["#111"] + [colors[k] for k in range(n_cls)])
    im2 = axes[0, 2].imshow(cmap_img, cmap=cmap_disc, vmin=0, vmax=n_cls, origin="upper")
    axes[0, 2].set_title("K-means cluster map (k=5)"); axes[0, 2].axis("off")

    # Cluster time series
    means = np.stack([recon[:, labels == k].mean(axis=1)
                      if (labels == k).any() else np.zeros(n)
                      for k in range(n_cls)], axis=1)
    for k in range(n_cls):
        axes[1, 0].plot(taxis, means[:, k] + k * 1.5, color=colors[k],
                        lw=1.2, label=f"C{k+1}")
    axes[1, 0].set_xlabel("Time (s)"); axes[1, 0].set_yticks([])
    axes[1, 0].set_title("Cluster mean time series"); axes[1, 0].legend(fontsize=8)

    # Moving-std spatial map
    T_std = np.std(recon.reshape(n, N_ROI), axis=0)
    im4 = axes[1, 1].imshow(to_grid(T_std), cmap="viridis", origin="upper")
    axes[1, 1].set_title("Std of PCA-reconstructed signal"); axes[1, 1].axis("off")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Power spectrum of all cluster means
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    for k in range(n_cls):
        psd = np.abs(np.fft.rfft(means[:, k]))
        axes[1, 2].semilogy(freqs, psd, color=colors[k], lw=0.8, alpha=0.8, label=f"C{k+1}")
    axes[1, 2].axvspan(110, 130, alpha=0.15, color="red",  label="notch 110-130 Hz")
    axes[1, 2].axvspan(7,   10,  alpha=0.15, color="gold", label="notch 7-10 Hz")
    axes[1, 2].set_xlim(0, 500); axes[1, 2].set_xlabel("Frequency (Hz)")
    axes[1, 2].set_title("Power spectra of cluster means"); axes[1, 2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "step6_summary.png", dpi=150)
    plt.close(fig)
    print("  Saved step6_summary.png")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 – 6-panel MP4 video
# ═══════════════════════════════════════════════════════════════════════════════

def generate_video(n_frames: int, out_dir: Path):
    """
    Replicates MATLAB HeartBeat.mp4: averages FAvg=10 frames per video frame.
    6 panels: [intensity P1|P2], [phase P1|P2], [intensity ratio], [retardation].
    """
    try:
        import imageio
    except ImportError:
        print("  imageio not installed. Run: pip install imageio imageio-ffmpeg")
        return

    mask1 = make_phase_mask(PM1)
    mask2 = make_phase_mask(PM2)
    e1 = np.exp(-1j * mask1).astype(np.complex64)
    e2 = np.exp( 1j * mask2).astype(np.complex64)   # video uses +PhaseMask2

    F_AVG     = 10
    FPS_OUT   = 150
    n_vframes = n_frames // F_AVG

    # Crop range (from MATLAB: yshow=21:200, xshow=1:180)
    yshow = slice(20, H)       # rows 21..200 (0-indexed 20..199)
    xshow = slice(0,  W - 20)  # cols  1..180 (0-indexed  0..179)
    gap_cols = np.zeros((H - 20, 10), dtype=np.float32)

    vid_path = out_dir / "HeartBeat.mp4"
    writer = imageio.get_writer(str(vid_path), fps=FPS_OUT, quality=6,
                                 codec="libx264", pixelformat="yuv420p")

    fig = plt.figure(figsize=(14, 8), facecolor="black")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.35)
    ax_int  = fig.add_subplot(gs[0, :2])   # intensity  (spans 2 cols)
    ax_ph   = fig.add_subplot(gs[1, :2])   # phase
    ax_rat  = fig.add_subplot(gs[0, 2])    # intensity ratio
    ax_ret  = fig.add_subplot(gs[1, 2])    # retardation

    for ax in [ax_int, ax_ph, ax_rat, ax_ret]:
        ax.axis("off")

    OCMPhaseMap = plt.cm.twilight
    taxis = np.arange(n_frames) / FS

    print(f"  Rendering {n_vframes} video frames...")
    for vi in range(n_vframes):
        start = vi * F_AVG
        p1 = load_batch(P1_PATH, start, F_AVG) * e1   # (F_AVG, H, W)
        p2 = load_batch(P2_PATH, start, F_AVG) * e2
        p2 = np.conj(p2) * np.exp(-1j * np.pi / 2)    # match MATLAB video branch

        delta_s = np.degrees(np.arctan2(
            np.abs(1j * p2 - p1), np.abs(1j * p2 + p1)))  # (F_AVG, H, W)

        # Intensity side-by-side
        int_img  = np.hstack([np.abs(p1).mean(0)[yshow, xshow], gap_cols,
                               np.abs(p2).mean(0)[yshow, xshow]])
        # Phase side-by-side (using middle frame)
        mid = F_AVG // 2
        ang1 = np.angle(p1[mid])
        ang2 = np.angle(p2[mid])
        ph_img = np.hstack([ang1[yshow, xshow], gap_cols, ang2[yshow, xshow]])
        # Intensity ratio
        a1 = np.abs(p1[yshow, xshow]).mean(0)
        a2 = np.abs(p2[yshow, xshow]).mean(0)
        rat_img = a1 / (a1 + a2 + 1e-9)
        # Retardation (middle frame)
        ret_img = delta_s[mid][yshow, xshow]

        # Clear & redraw
        ax_int.cla();  ax_ph.cla();  ax_rat.cla();  ax_ret.cla()
        for ax in [ax_int, ax_ph, ax_rat, ax_ret]:
            ax.axis("off")

        ax_int.imshow(int_img,  cmap="gray",       vmin=0,    vmax=5000,  origin="upper")
        ax_int.set_title(f"OCT Intensity  t={taxis[start]:.3f}s", color="white", fontsize=10)

        ax_ph.imshow(ph_img,   cmap=OCMPhaseMap,  vmin=-np.pi, vmax=np.pi, origin="upper")
        ax_ph.set_title("Phase (rad)", color="white", fontsize=10)

        ax_rat.imshow(rat_img, cmap="RdBu_r",     vmin=0.25, vmax=0.75, origin="upper")
        ax_rat.set_title("Intensity ratio", color="white", fontsize=10)

        ax_ret.imshow(ret_img, cmap="inferno",    vmin=0,    vmax=90,   origin="upper")
        ax_ret.set_title("Retardation (°)", color="white", fontsize=10)

        fig.canvas.draw()
        frame_rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame_rgb = frame_rgb.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(frame_rgb)

        if vi % 100 == 0:
            print(f"\r    Video frame {vi}/{n_vframes}", end="", flush=True)

    print()
    plt.close(fig)
    writer.close()
    print(f"  Saved {vid_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SPoOF-OCM Python pipeline")
    parser.add_argument("--frames", type=int, default=N_FRAMES,
                        help=f"Number of frames to process (default: all {N_FRAMES})")
    parser.add_argument("--video",  action="store_true", help="Generate 6-panel MP4")
    parser.add_argument("--load",   action="store_true",
                        help="Load saved results.npz and skip extraction")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUT_DIR / "pipeline_results.npz"

    n_frames = min(args.frames, N_FRAMES)

    # ── Step 1: phase masks ─────────────────────────────────────────────────
    print("\n[Step 1] Building phase masks...")
    mask1 = make_phase_mask(PM1)
    mask2 = make_phase_mask(PM2)
    win   = circular_window()
    plot_phase_masks(mask1, mask2, win, OUT_DIR)

    if args.load and npz_path.exists():
        print(f"\n[Loading saved results from {npz_path}]")
        npz = np.load(npz_path)
        T3_raw   = npz["T3_raw"]
        T3_filt  = npz["T3_filt"]
        recon    = npz["recon"]
        labels   = npz["labels"]
        XL       = npz["XL"]
        YL       = npz["YL"]
        n_frames = T3_raw.shape[0]
    else:
        # ── Step 2: ROI extraction + retardation ───────────────────────────
        print(f"\n[Step 2] Extracting ROI time series ({n_frames} frames)...")
        XL, YL = make_roi_grid()
        print(f"  ROI grid: {len(XL)} ROIs")
        T1, T2, XL, YL = extract_roi_timeseries(n_frames)
        T3_raw = compute_retardation(T1, T2)   # (n_frames, N_ROI) degrees
        print(f"  Retardation shape: {T3_raw.shape}")
        plot_retardation_sample(T3_raw, XL, YL, OUT_DIR)

        # ── Step 3: temporal filtering ─────────────────────────────────────
        print("\n[Step 3] Applying temporal filters...")
        T3_filt = apply_temporal_filters(T3_raw)
        plot_filter_comparison(T3_raw, T3_filt, roi_idx=len(XL) // 2, out_dir=OUT_DIR)

        # ── Step 4: ROI grid plot ──────────────────────────────────────────
        print("\n[Step 4] Plotting ROI grid...")
        plot_roi_grid(XL, YL, OUT_DIR)

        # ── Step 5: PCA + k-means ──────────────────────────────────────────
        print("\n[Step 5] Running PCA + k-means clustering...")
        recon, labels, T3_mm = run_pca_kmeans(T3_filt)
        plot_clustering(recon, labels, T3_mm, XL, YL, OUT_DIR)

        # Save results
        np.savez_compressed(npz_path,
                             T3_raw=T3_raw, T3_filt=T3_filt,
                             recon=recon, labels=labels,
                             XL=XL, YL=YL)
        print(f"  Results saved to {npz_path}")

    # ── Step 6: summary figure ─────────────────────────────────────────────
    print("\n[Step 6] Generating summary figure...")
    plot_summary(T3_raw, T3_filt, recon, labels, XL, YL, OUT_DIR)

    # ── Step 7: optional video ─────────────────────────────────────────────
    if args.video:
        print(f"\n[Step 7] Generating 6-panel MP4 ({n_frames} frames)...")
        generate_video(n_frames, OUT_DIR)

    print(f"\nDone. All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
