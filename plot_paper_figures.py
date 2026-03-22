#!/usr/bin/env python3
"""Generate paper-ready PNG figures from Ultralytics run directory."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_training_curves(run_dir: Path, out_dir: Path) -> None:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print(f"Skip curves: {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    x = df["epoch"] if "epoch" in df.columns else range(len(df))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180)

    metric_candidates = [
        ("train/box_loss", "Train Box Loss"),
        ("train/cls_loss", "Train Cls Loss"),
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95"),
    ]

    for ax, (col, title) in zip(axes.flatten(), metric_candidates):
        if col in df.columns:
            ax.plot(x, df[col], linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
        else:
            ax.set_title(f"{title} (missing)")
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "paper_training_curves.png")
    plt.close(fig)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.write_bytes(src.read_bytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=str, help="e.g. runs/bird_train/yolo11_bird_hier")
    ap.add_argument("--out-dir", default="runs/paper_figures", type=str)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_training_curves(run_dir, out_dir)

    # Ultralytics generated figures
    copy_if_exists(run_dir / "confusion_matrix.png", out_dir / "paper_confusion_matrix.png")
    copy_if_exists(run_dir / "PR_curve.png", out_dir / "paper_pr_curve.png")
    copy_if_exists(run_dir / "F1_curve.png", out_dir / "paper_f1_curve.png")
    copy_if_exists(run_dir / "P_curve.png", out_dir / "paper_precision_curve.png")
    copy_if_exists(run_dir / "R_curve.png", out_dir / "paper_recall_curve.png")

    print("Done. Paper figures at:", out_dir)


if __name__ == "__main__":
    main()
