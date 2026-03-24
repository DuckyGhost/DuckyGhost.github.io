#!/usr/bin/env python3
"""Reliable trainer for processed CUB+web bird dataset.

Strategy:
1) Build a mixed train split under processed/mixed with source ratio CUB:web = 1:2.
2) Keep validation as union of CUB val and web val.
3) Train YOLO on mixed dataset.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from ultralytics import YOLO


def list_pairs(root: Path, source: str, split: str) -> List[Tuple[Path, Path]]:
    img_dir = root / source / "images" / split
    lbl_dir = root / source / "labels" / split
    pairs = []
    for img in img_dir.glob("*.*"):
        lbl = lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def class_id_from_label(lbl: Path) -> int:
    lines = [ln.strip() for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return int(lines[0].split()[0]) if lines else -1


def safe_link_or_copy(src: Path, dst: Path) -> None:
    try:
        dst.symlink_to(src.resolve())
    except Exception:
        shutil.copy2(src, dst)


def build_mixed_dataset(processed_root: Path, batch: int, steps_per_epoch: int, seed: int) -> Path:
    if batch % 3 != 0:
        raise ValueError("batch must be divisible by 3")

    cub_train = list_pairs(processed_root, "cub", "train")
    web_train = list_pairs(processed_root, "web", "train")
    cub_val = list_pairs(processed_root, "cub", "val")
    web_val = list_pairs(processed_root, "web", "val")

    if not cub_train:
        raise RuntimeError("No CUB train samples found")
    if not web_train:
        raise RuntimeError("No web train samples found. Check webbird_pseudobox_stats.json")

    # Build class-balanced map for web
    web_by_cls: Dict[int, List[Tuple[Path, Path]]] = defaultdict(list)
    for p in web_train:
        web_by_cls[class_id_from_label(p[1])].append(p)
    web_classes = [c for c in web_by_cls.keys() if c >= 0]
    if not web_classes:
        raise RuntimeError("No valid web classes found in labels")

    mixed_root = processed_root / "mixed"
    if mixed_root.exists():
        shutil.rmtree(mixed_root)

    for split in ["train", "val"]:
        (mixed_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (mixed_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    rnd = random.Random(seed)
    cub_per_batch = batch // 3
    web_per_batch = batch - cub_per_batch

    idx = 0
    # Create ordered train set with per-batch 1:2 source composition
    for _ in range(steps_per_epoch):
        batch_items: List[Tuple[Path, Path]] = []
        batch_items.extend(rnd.choice(cub_train) for _ in range(cub_per_batch))

        # class-balanced sampling for web
        for _ in range(web_per_batch):
            c = rnd.choice(web_classes)
            batch_items.append(rnd.choice(web_by_cls[c]))

        rnd.shuffle(batch_items)
        for img, lbl in batch_items:
            stem = f"mix_{idx:07d}_{img.stem}"
            img_dst = mixed_root / "images" / "train" / f"{stem}{img.suffix.lower()}"
            lbl_dst = mixed_root / "labels" / "train" / f"{stem}.txt"
            safe_link_or_copy(img, img_dst)
            safe_link_or_copy(lbl, lbl_dst)
            idx += 1

    # Validation = cub val + web val
    vidx = 0
    for img, lbl in cub_val + web_val:
        stem = f"val_{vidx:07d}_{img.stem}"
        img_dst = mixed_root / "images" / "val" / f"{stem}{img.suffix.lower()}"
        lbl_dst = mixed_root / "labels" / "val" / f"{stem}.txt"
        safe_link_or_copy(img, img_dst)
        safe_link_or_copy(lbl, lbl_dst)
        vidx += 1

    # load class names
    names = json.loads((processed_root / "class_names.json").read_text(encoding="utf-8"))
    dataset_yaml = mixed_root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {mixed_root.resolve()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(names)}",
                f"names: {names}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return dataset_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="model/processed", type=str)
    ap.add_argument("--model", default="yolo11l.pt", type=str)
    ap.add_argument("--epochs", default=100, type=int)
    ap.add_argument("--batch", default=30, type=int)
    ap.add_argument("--imgsz", default=960, type=int)
    ap.add_argument("--steps-per-epoch", default=200, type=int)
    ap.add_argument("--workers", default=8, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--project", default="runs/bird_train", type=str)
    ap.add_argument("--name", default="yolo11_bird_hier", type=str)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    dataset_yaml = build_mixed_dataset(dataset_root, args.batch, args.steps_per_epoch, args.seed)

    model = YOLO(args.model)
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        seed=args.seed,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    print("Done training. Check:", Path(args.project) / args.name / "weights" / "last.pt")


if __name__ == "__main__":
    main()
