#!/usr/bin/env python3
"""
Phase 5 — Supervised training for PhaseAberrationNet.

Loss (three physics-motivated terms):
  L = λ_sample × L1(φ̂_sample, φ_sample_GT)       — sample phase accuracy
    + λ_coeff  × MSE(ĉ, c_true)                    — Zernike coefficient accuracy
    + λ_cycle  × PhLoss(φ̂_sample + φ̂_aber, φ_total) — physics cycle consistency

The cycle term uses a cos/sin decomposition so it is insensitive to the 2π wrap
discontinuity in the (wrapped) network input φ_total.

Usage:
  python train_supervised.py                        # default settings
  python train_supervised.py --epochs 200 --batch-size 4 --base-ch 32
  python train_supervised.py --resume checkpoints/latest.pth
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from models import PhaseAberrationNet
from simulate_phase_data import make_train_val_datasets
from zernike_utils import N_MODES


# ── Loss utilities ────────────────────────────────────────────────────────────

def complex_phase_loss(
    phi_unwrapped: torch.Tensor,
    phi_wrapped: torch.Tensor,
) -> torch.Tensor:
    """
    Phase-consistent MSE between an unwrapped prediction and a wrapped target.
    Compares cos(φ) and sin(φ) so 2π jumps don't contribute to the loss.
    Both tensors are [B, 1, H, W].
    """
    return 0.5 * (
        F.mse_loss(torch.cos(phi_unwrapped), torch.cos(phi_wrapped)) +
        F.mse_loss(torch.sin(phi_unwrapped), torch.sin(phi_wrapped))
    )


def compute_loss(
    model: PhaseAberrationNet,
    batch: dict,
    lambdas: dict,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """
    Forward pass + loss computation for one batch.

    Returns
    -------
    loss       : scalar tensor (backpropagatable)
    components : dict with float values for logging
    """
    phi_total     = batch["phi_total"].to(device)     # [B,1,H,W] wrapped
    phi_sample_gt = batch["phi_sample"].to(device)    # [B,1,H,W] unwrapped GT
    coeffs_gt     = batch["aber_coeffs"].to(device)   # [B, N_MODES]

    phi_sample_pred, coeffs_pred = model(phi_total)
    phi_aber_pred  = model.reconstruct_aberration(coeffs_pred)  # [B,1,H,W]
    phi_total_pred = phi_sample_pred + phi_aber_pred            # [B,1,H,W]

    L_sample = F.l1_loss(phi_sample_pred, phi_sample_gt)
    L_coeff  = F.mse_loss(coeffs_pred, coeffs_gt)
    L_cycle  = complex_phase_loss(phi_total_pred, phi_total)

    loss = (lambdas["sample"] * L_sample +
            lambdas["coeff"]  * L_coeff  +
            lambdas["cycle"]  * L_cycle)

    return loss, {
        "total":  loss.item(),
        "sample": L_sample.item(),
        "coeff":  L_coeff.item(),
        "cycle":  L_cycle.item(),
    }


# ── Train / validate ──────────────────────────────────────────────────────────

def run_epoch(
    model: PhaseAberrationNet,
    loader: DataLoader,
    optimizer,
    lambdas: dict,
    device: torch.device,
    train: bool,
) -> dict:
    model.train(train)
    totals = {"total": 0.0, "sample": 0.0, "coeff": 0.0, "cycle": 0.0}
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            loss, comp = compute_loss(model, batch, lambdas, device)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            for k in totals:
                totals[k] += comp[k]
            n += 1

    return {k: v / n for k, v in totals.items()}


# ── Loss curve ────────────────────────────────────────────────────────────────

def save_loss_curve(history: dict, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs      = range(1, len(history["train"]) + 1)
        train_total = [m["total"]  for m in history["train"]]
        val_total   = [m["total"]  for m in history["val"]]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Training history", fontsize=12)

        axes[0].plot(epochs, train_total, label="train")
        axes[0].plot(epochs, val_total,   label="val")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("Total loss"); axes[0].legend()
        axes[0].set_yscale("log")

        colors = {"sample": "steelblue", "coeff": "tomato", "cycle": "seagreen"}
        for key, color in colors.items():
            axes[1].plot(epochs, [m[key] for m in history["val"]],
                         label=f"val {key}", color=color)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
        axes[1].set_title("Val loss components"); axes[1].legend()
        axes[1].set_yscale("log")

        fig.tight_layout()
        path = out_dir / "loss_curve.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  Loss curve → {path}")
    except Exception as e:
        print(f"  Warning: could not save loss curve ({e})")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train PhaseAberrationNet")
    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--batch-size",      type=int,   default=8)
    p.add_argument("--n-train",         type=int,   default=4000)
    p.add_argument("--n-val",           type=int,   default=500)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--lambda-sample",   type=float, default=1.0,
                   help="L1 weight for sample phase (default 1.0)")
    p.add_argument("--lambda-coeff",    type=float, default=0.1,
                   help="MSE weight for Zernike coefficients (default 0.1)")
    p.add_argument("--lambda-cycle",    type=float, default=0.1,
                   help="Cycle-consistency weight (default 0.1)")
    p.add_argument("--base-ch",         type=int,   default=64,
                   help="U-Net base channel width (use 32 to halve memory)")
    p.add_argument("--out-dir",         type=Path,  default=HERE / "checkpoints")
    p.add_argument("--resume",          type=Path,  default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--num-workers",     type=int,   default=4)
    p.add_argument("--gpu-id",          type=int,   default=0,
                   help="CUDA device index (default 0); ignored if no GPU available")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch size   : {args.batch_size}")
    print(f"LR           : {args.lr}")
    print(f"λ sample/coeff/cycle : {args.lambda_sample}/{args.lambda_coeff}/{args.lambda_cycle}")
    print(f"Checkpoints  : {args.out_dir}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    ds_train, ds_val = make_train_val_datasets(args.n_train, args.n_val)
    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    print(f"Train samples: {len(ds_train)}   Val samples: {len(ds_val)}\n")

    # ── Model & optimizer ─────────────────────────────────────────────────────
    model     = PhaseAberrationNet(n_modes=N_MODES, base_ch=args.base_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )
    lambdas = {
        "sample": args.lambda_sample,
        "coeff":  args.lambda_coeff,
        "cycle":  args.lambda_cycle,
    }

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}\n")

    start_epoch = 1
    best_val    = float("inf")
    history     = {"train": [], "val": []}

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume and args.resume.exists():
        ckpt        = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", float("inf"))
        history     = ckpt.get("history", history)
        print(f"Resumed from epoch {ckpt['epoch']}  best_val={best_val:.4f}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    header = f"{'Epoch':>6}  {'Train':>9}  {'Val':>9}  {'LR':>8}  {'Time':>6}"
    print(header)
    print("-" * len(header))

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_m = run_epoch(model, train_loader, optimizer, lambdas, device, train=True)
        val_m   = run_epoch(model, val_loader,   optimizer, lambdas, device, train=False)
        scheduler.step(val_m["total"])

        lr      = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        history["train"].append(train_m)
        history["val"].append(val_m)

        print(
            f"{epoch:>6}  {train_m['total']:>9.4f}  {val_m['total']:>9.4f}  "
            f"{lr:>8.1e}  {elapsed:>5.1f}s"
            f"  [s={val_m['sample']:.3f} c={val_m['coeff']:.3f} cy={val_m['cycle']:.3f}]"
        )

        ckpt = dict(
            epoch=epoch, model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            best_val=best_val, history=history, args=vars(args),
        )
        torch.save(ckpt, args.out_dir / "latest.pth")

        if val_m["total"] < best_val:
            best_val = val_m["total"]
            torch.save(ckpt, args.out_dir / "best.pth")
            print(f"         ★ new best  val={best_val:.4f}")

    print(f"\nTraining complete.  Best val loss: {best_val:.4f}")
    save_loss_curve(history, args.out_dir)


if __name__ == "__main__":
    main()
