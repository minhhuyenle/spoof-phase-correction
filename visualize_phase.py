"""
Visualize SPoOF-OCM complex phase data from Processed_P1.bin / Processed_P2.bin.

Data layout: 43674 frames of 200x200 complex float32
  Each frame = 200*200*2 float32 values [real, imag, real, imag, ...]

Usage:
  python3 visualize_phase.py                  # shows frame 0
  python3 visualize_phase.py --frame 100      # shows frame 100
  python3 visualize_phase.py --frame 100 --save
  python3 visualize_phase.py --strip          # time strip across 12 frames
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent / "D1_1000fps_512x512_C001H001S0001"
P1_PATH = BASE / "Processed_P1.bin"
P2_PATH = BASE / "Processed_P2.bin"

N_FRAMES = 43674
ROI_H, ROI_W = 200, 200
VALS_PER_FRAME = ROI_H * ROI_W * 2   # interleaved real/imag float32


def load_frame(path: Path, frame_idx: int) -> np.ndarray:
    """Return (200, 200) complex64 array for the requested frame."""
    offset = frame_idx * VALS_PER_FRAME * np.dtype(np.float32).itemsize
    raw = np.memmap(path, dtype=np.float32, mode="r",
                    offset=offset, shape=(VALS_PER_FRAME,))
    return (raw[0::2] + 1j * raw[1::2]).reshape(ROI_H, ROI_W)


def retardation(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Polarization retardation angle (degrees) — matches BatchAnalyze.m formula."""
    return np.degrees(np.arctan2(np.abs(1j * p2 - p1), np.abs(1j * p2 + p1)))


def show_frame(frame_idx: int, save: bool = False):
    p1 = load_frame(P1_PATH, frame_idx)
    p2 = load_frame(P2_PATH, frame_idx)
    ret = retardation(p1, p2)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f"SPoOF-OCM  —  Frame {frame_idx}/{N_FRAMES-1}"
        f"  (t = {frame_idx/1000:.3f} s @ 1000 fps)",
        fontsize=13
    )

    panels = [
        (axes[0, 0], np.abs(p1),                               "P1 Amplitude",       "viridis"),
        (axes[0, 1], np.abs(p2),                               "P2 Amplitude",       "viridis"),
        (axes[1, 0], np.angle(p1),                             "P1 Phase (rad)",     "twilight"),
        (axes[1, 1], np.angle(p2),                             "P2 Phase (rad)",     "twilight"),
        (axes[0, 2], np.abs(p1) / (np.abs(p2) + 1e-9),        "Amplitude P1/P2",    "RdBu_r"),
        (axes[1, 2], ret,                                       "Retardation (deg)",  "inferno"),
    ]

    for ax, data, title, cmap in panels:
        im = ax.imshow(data, cmap=cmap, origin="upper")
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save:
        out = Path(f"frame_{frame_idx:05d}.png")
        fig.savefig(out, dpi=150)
        print(f"Saved -> {out}")
    else:
        plt.show()


def show_strip(n: int = 12, save: bool = False):
    """Show n evenly-spaced frames: amplitude, phase, retardation."""
    indices = np.linspace(0, N_FRAMES - 1, n, dtype=int)
    fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 7))
    fig.suptitle(f"Time strip — {n} frames across full recording", fontsize=13)

    row_labels = ["P1 Amplitude", "P1 Phase (rad)", "Retardation (deg)"]
    cmaps      = ["viridis",       "twilight",        "inferno"]

    for col, fi in enumerate(indices):
        p1 = load_frame(P1_PATH, fi)
        p2 = load_frame(P2_PATH, fi)
        rows = [np.abs(p1), np.angle(p1), retardation(p1, p2)]

        for row, (data, cmap) in enumerate(zip(rows, cmaps)):
            ax = axes[row, col]
            ax.imshow(data, cmap=cmap, origin="upper")
            ax.axis("off")
            if row == 0:
                ax.set_title(f"t={fi/1000:.2f}s", fontsize=8)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)

    plt.tight_layout()
    if save:
        out = Path("time_strip.png")
        fig.savefig(out, dpi=150)
        print(f"Saved -> {out}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SPoOF-OCM phase data")
    parser.add_argument("--frame", type=int, default=0,
                        help=f"Frame index (0-{N_FRAMES-1})")
    parser.add_argument("--save", action="store_true",
                        help="Save to PNG instead of showing interactively")
    parser.add_argument("--strip", action="store_true",
                        help="Show a time strip of 12 evenly-spaced frames")
    args = parser.parse_args()

    if args.strip:
        show_strip(save=args.save)
    else:
        if not (0 <= args.frame < N_FRAMES):
            parser.error(f"--frame must be 0-{N_FRAMES-1}")
        show_frame(args.frame, save=args.save)
