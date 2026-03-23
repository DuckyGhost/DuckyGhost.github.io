#!/usr/bin/env python3
"""
Training entry for YOLOv11 bird detection+hierarchical classification.
Implements a practical batch-level CUB:web-bird=1:2 sampling by generating
an epoch-specific train split with exact source ratio.
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from ultralytics import YOLO


def read_names(data_yaml: Path) -> List[str]:
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    return d["names"]


def collect_train_pairs(dataset_root: Path) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], Dict[int, List[Tuple[Path, Path]]]]:
    img_dir = dataset_root / "images" / "train"
    lbl_dir = dataset_root / "labels" / "train"

    cub_pairs: List[Tuple[Path, Path]] = []
    web_pairs: List[Tuple[Path, Path]] = []
    web_cls_map: Dict[int, List[Tuple[Path, Path]]] = defaultdict(list)

    for img in img_dir.glob("*.*"):
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        pair = (img, lbl)
        if img.name.startswith("web__"):
            web_pairs.append(pair)
            lines = [ln.strip() for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                cls_id = int(lines[0].split()[0])
                web_cls_map[cls_id].append(pair)
        else:
            cub_pairs.append(pair)

    if not cub_pairs:
        raise RuntimeError("No CUB samples found in processed train split.")
    if not web_pairs:
        raise RuntimeError(
            "No web-bird samples found in processed train split. "
            "Check model/processed/webbird_pseudobox_stats.json and rerun "
            "prepare_bird_data.py with lower --web-conf or --web-fallback-fullbox."
        )

    return cub_pairs, web_pairs, web_cls_map


def sample_web_balanced(web_cls_map: Dict[int, List[Tuple[Path, Path]]], n: int, rng: random.Random) -> List[Tuple[Path, Path]]:
    classes = list(web_cls_map.keys())
    out: List[Tuple[Path, Path]] = []
    while len(out) < n:
        rng.shuffle(classes)
        for c in classes:
            out.append(rng.choice(web_cls_map[c]))
            if len(out) >= n:
                break
    return out


def build_epoch_trainset(
    dataset_root: Path,
    epoch_root: Path,
    cub_pairs: List[Tuple[Path, Path]],
    web_cls_map: Dict[int, List[Tuple[Path, Path]]],
    batch_size: int,
    steps_per_epoch: int,
    seed: int,
) -> Path:
    """Generate images/labels/train with strict 1:2 source ratio at batch granularity.
    Requires batch_size divisible by 3.
    """
    if batch_size % 3 != 0:
        raise ValueError("batch-size must be divisible by 3 for exact 1:2 batch-level sampling.")

    rng = random.Random(seed)
    num_cub_per_batch = batch_size // 3
    num_web_per_batch = batch_size - num_cub_per_batch

    total_cub = steps_per_epoch * num_cub_per_batch
    total_web = steps_per_epoch * num_web_per_batch

    sampled_cub = [rng.choice(cub_pairs) for _ in range(total_cub)]  # with replacement
    sampled_web = sample_web_balanced(web_cls_map, total_web, rng)

    train_img = epoch_root / "images" / "train"
    train_lbl = epoch_root / "labels" / "train"
    val_img = epoch_root / "images" / "val"
    val_lbl = epoch_root / "labels" / "val"

    if epoch_root.exists():
        shutil.rmtree(epoch_root)
    train_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)
    val_img.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)

    # Keep validation as original via symlink
    src_val_img = dataset_root / "images" / "val"
    src_val_lbl = dataset_root / "labels" / "val"
    for p in src_val_img.glob("*.*"):
        (val_img / p.name).symlink_to(p.resolve())
    for p in src_val_lbl.glob("*.txt"):
        (val_lbl / p.name).symlink_to(p.resolve())

    idx = 0
    cub_cursor = 0
    web_cursor = 0
    for _ in range(steps_per_epoch):
        batch_pairs = []
        batch_pairs.extend(sampled_cub[cub_cursor : cub_cursor + num_cub_per_batch])
        batch_pairs.extend(sampled_web[web_cursor : web_cursor + num_web_per_batch])
        rng.shuffle(batch_pairs)
        cub_cursor += num_cub_per_batch
        web_cursor += num_web_per_batch

        for img, lbl in batch_pairs:
            dst_stem = f"ep_{idx:07d}_{img.stem}"
            (train_img / f"{dst_stem}{img.suffix.lower()}").symlink_to(img.resolve())
            (train_lbl / f"{dst_stem}.txt").symlink_to(lbl.resolve())
            idx += 1

    names = read_names(dataset_root / "dataset.yaml")
    ep_yaml = epoch_root / "dataset_epoch.yaml"
    ep_yaml.write_text(
        "\n".join(
            [
                f"path: {epoch_root.resolve()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(names)}",
                f"names: {names}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return ep_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="model/processed", type=str)
    ap.add_argument("--project", default="runs/bird_train", type=str)
    ap.add_argument("--name", default="yolo11_bird_hier", type=str)
    ap.add_argument("--model", default="yolo11l.pt", type=str)
    ap.add_argument("--epochs", default=100, type=int)
    ap.add_argument("--batch", default=30, type=int, help="must be divisible by 3")
    ap.add_argument("--imgsz", default=960, type=int)
    ap.add_argument("--steps-per-epoch", default=200, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--workers", default=8, type=int)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    temp_epoch_root = dataset_root / "_epoch_tmp"

    cub_pairs, _, web_cls_map = collect_train_pairs(dataset_root)
    model = YOLO(args.model)
    last_ckpt = None

    for ep in range(args.epochs):
        ep_yaml = build_epoch_trainset(
            dataset_root=dataset_root,
            epoch_root=temp_epoch_root,
            cub_pairs=cub_pairs,
            web_cls_map=web_cls_map,
            batch_size=args.batch,
            steps_per_epoch=args.steps_per_epoch,
            seed=args.seed + ep,
        )

        train_model = last_ckpt if last_ckpt else args.model
        model = YOLO(train_model)
        results = model.train(
            data=str(ep_yaml),
            epochs=1,
            batch=args.batch,
            imgsz=args.imgsz,
            workers=args.workers,
            project=args.project,
            name=args.name,
            exist_ok=True,
            seed=args.seed + ep,
            resume=False,
            close_mosaic=0,
        )

        save_dir = Path(results.save_dir)
        candidate = save_dir / "weights" / "last.pt"
        if candidate.exists():
            last_ckpt = str(candidate)
        print(f"Finished epoch {ep + 1}/{args.epochs}, checkpoint={last_ckpt}")

    if last_ckpt:
        print("Training complete. Best checkpoint directory:", Path(last_ckpt).parent)


if __name__ == "__main__":
    main()
