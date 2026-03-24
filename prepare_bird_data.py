#!/usr/bin/env python3
"""Prepare CUB_200_2011 + web-bird into YOLO detection dataset.

Assumed extracted layout:
- CUB_200_2011/
  - images/, classes.txt, images.txt, image_class_labels.txt, bounding_boxes.txt, train_test_split.txt
- web-bird/
  - train/<class_name>/*.jpg
  - val/<class_name>/*.jpg
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import tarfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from ultralytics import YOLO


def extract_if_needed(archive: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        tar.extractall(dst)


def find_existing(base: Path, names: List[str]) -> Path:
    for n in names:
        p = base / n
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing required path in {base}, tried: {names}")


def normalize_class_name(name: str) -> str:
    # CUB class line example: 1 001.Black_footed_Albatross
    if "." in name:
        name = name.split(".", 1)[1]
    return name.strip()


def coarse_from_name(name: str) -> str:
    n = name.lower().replace("_", " ").replace("-", " ")
    rules = {
        "gull": ["gull", "tern"],
        "waterfowl": ["duck", "goose", "swan", "grebe"],
        "shorebird": ["sandpiper", "plover", "willet", "snipe", "oystercatcher"],
        "wader": ["ibis", "heron", "egret", "bittern", "crane", "stork"],
        "raptor": ["hawk", "eagle", "falcon", "kite", "vulture", "owl"],
        "songbird": ["sparrow", "warbler", "finch", "vireo", "thrush", "oriole", "wren", "jay", "crow"],
        "woodpecker": ["woodpecker", "flicker", "sapsucker"],
        "seabird": ["albatross", "auklet", "puffin", "petrel", "booby", "cormorant", "frigatebird"],
        "gamebird": ["grouse", "quail", "pheasant", "turkey", "partridge"],
        "pigeon_dove": ["pigeon", "dove"],
        "hummingbird": ["hummingbird"],
    }
    for coarse, keys in rules.items():
        if any(k in n for k in keys):
            return coarse
    return "other_bird"


def parse_cub_metadata(cub_root: Path):
    images_txt = cub_root / "images.txt"
    bboxes_txt = cub_root / "bounding_boxes.txt"
    classes_txt = cub_root / "classes.txt"
    image_cls_txt = cub_root / "image_class_labels.txt"
    split_txt = cub_root / "train_test_split.txt"

    for p in [images_txt, bboxes_txt, classes_txt, image_cls_txt, split_txt]:
        if not p.exists():
            raise FileNotFoundError(f"Missing CUB metadata file: {p}")

    id2rel = {}
    with images_txt.open("r", encoding="utf-8") as f:
        for line in f:
            i, rel = line.strip().split(" ", 1)
            id2rel[int(i)] = rel

    id2bbox = {}
    with bboxes_txt.open("r", encoding="utf-8") as f:
        for line in f:
            i, x, y, w, h = line.strip().split()
            id2bbox[int(i)] = (float(x), float(y), float(w), float(h))

    cid2name = {}
    with classes_txt.open("r", encoding="utf-8") as f:
        for line in f:
            i, name = line.strip().split(" ", 1)
            cid2name[int(i)] = normalize_class_name(name)

    id2cid = {}
    with image_cls_txt.open("r", encoding="utf-8") as f:
        for line in f:
            i, cid = line.strip().split()
            id2cid[int(i)] = int(cid)

    id2is_train = {}
    with split_txt.open("r", encoding="utf-8") as f:
        for line in f:
            i, is_train = line.strip().split()
            id2is_train[int(i)] = int(is_train) == 1

    return id2rel, id2bbox, cid2name, id2cid, id2is_train


def ensure_dirs(root: Path) -> None:
    for src in ["cub", "web", "mixed"]:
        for split in ["train", "val"]:
            (root / src / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / src / "labels" / split).mkdir(parents=True, exist_ok=True)


def save_yolo_label(path: Path, cls_id: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> None:
    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    path.write_text(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")


def convert_cub(cub_root: Path, out_root: Path, coarse2id: Dict[str, int], fine2coarse: Dict[str, str]) -> int:
    id2rel, id2bbox, cid2name, id2cid, id2is_train = parse_cub_metadata(cub_root)
    written = 0

    for img_id, rel in id2rel.items():
        img_path = cub_root / "images" / rel
        split = "train" if id2is_train[img_id] else "val"

        fine = cid2name[id2cid[img_id]]
        coarse = fine2coarse.setdefault(fine, coarse_from_name(fine))
        if coarse not in coarse2id:
            coarse2id[coarse] = len(coarse2id)
        cls_id = coarse2id[coarse]

        dst_name = f"cub__{rel.replace('/', '__')}"
        dst_img = out_root / "cub" / "images" / split / dst_name
        shutil.copy2(img_path, dst_img)

        with Image.open(dst_img) as im:
            w, h = im.size

        x, y, bw, bh = id2bbox[img_id]
        x1, y1, x2, y2 = x, y, x + bw, y + bh
        lbl = out_root / "cub" / "labels" / split / f"{Path(dst_name).stem}.txt"
        save_yolo_label(lbl, cls_id, x1, y1, x2, y2, w, h)
        written += 1

    return written


def collect_web_images(web_root: Path, split: str) -> List[Tuple[str, Path]]:
    root = web_root / split
    if not root.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    pairs = []
    for class_dir in [p for p in root.iterdir() if p.is_dir()]:
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                pairs.append((class_dir.name, p))
    return pairs


def convert_web(
    web_root: Path,
    out_root: Path,
    coarse2id: Dict[str, int],
    conf: float,
    pseudo_model: str,
    seed: int,
) -> Dict[str, int]:
    random.seed(seed)
    model = YOLO(pseudo_model)

    stats = Counter()
    for split in ["train", "val"]:
        items = collect_web_images(web_root, split)
        stats[f"{split}_images_found"] = len(items)

        for cls_name, img_path in items:
            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im2:
                    w, h = im2.size
            except Exception:
                stats["broken_or_unreadable"] += 1
                continue

            try:
                pred = model.predict(source=str(img_path), conf=conf, verbose=False)
                boxes = pred[0].boxes
            except Exception:
                stats["predict_failed"] += 1
                continue

            bird = [b for b in boxes if int(b.cls.item()) == 14]
            if not bird:
                stats["no_bird_detected"] += 1
                continue

            coarse = coarse_from_name(cls_name)
            if coarse not in coarse2id:
                coarse2id[coarse] = len(coarse2id)
            cls_id = coarse2id[coarse]

            dst_name = f"web__{split}__{cls_name}__{img_path.stem}{img_path.suffix.lower()}"
            dst_img = out_root / "web" / "images" / split / dst_name
            shutil.copy2(img_path, dst_img)

            lbl = out_root / "web" / "labels" / split / f"{Path(dst_name).stem}.txt"
            with lbl.open("w", encoding="utf-8") as f:
                for b in bird:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    xc = ((x1 + x2) / 2.0) / w
                    yc = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            stats[f"{split}_kept"] += 1

    return dict(stats)


def write_metadata(out_root: Path, coarse2id: Dict[str, int], fine2coarse: Dict[str, str], web_stats: Dict[str, int]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    names = [None] * len(coarse2id)
    for k, v in coarse2id.items():
        names[v] = k

    (out_root / "hierarchy_map.json").write_text(
        json.dumps(
            {
                "coarse_to_id": {k: int(v) for k, v in sorted(coarse2id.items(), key=lambda x: x[1])},
                "fine_to_coarse": fine2coarse,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (out_root / "webbird_pseudobox_stats.json").write_text(json.dumps(web_stats, indent=2), encoding="utf-8")

    (out_root / "class_names.json").write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model", type=str)
    ap.add_argument("--out-dir", default="model/processed", type=str)
    ap.add_argument("--pseudo-model", default="yolo11n.pt", type=str)
    ap.add_argument("--web-conf", default=0.20, type=float)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_root = Path(args.out_dir)
    extract_root = model_dir / "_extracted"

    cub_archive = find_existing(model_dir, ["CUB_200_2011.tgz", "CUB-200-2011.tar", "CUB_200_2011.tar.gz"])
    web_archive = find_existing(model_dir, ["web-bird.tar.gz", "web_bird.tar.gz", "webbird.tar.gz"])

    # Extract only if target roots don't exist
    if not (extract_root / "CUB_200_2011").exists() and not (extract_root / "CUB-200-2011").exists():
        extract_if_needed(cub_archive, extract_root)
    if not (extract_root / "web-bird").exists() and not (extract_root / "web_bird").exists():
        extract_if_needed(web_archive, extract_root)

    cub_root = find_existing(extract_root, ["CUB_200_2011", "CUB-200-2011"])
    web_root = find_existing(extract_root, ["web-bird", "web_bird", "webbird"])

    if out_root.exists():
        shutil.rmtree(out_root)
    ensure_dirs(out_root)

    coarse2id: Dict[str, int] = {}
    fine2coarse: Dict[str, str] = {}

    cub_written = convert_cub(cub_root, out_root, coarse2id, fine2coarse)
    web_stats = convert_web(web_root, out_root, coarse2id, args.web_conf, args.pseudo_model, args.seed)
    write_metadata(out_root, coarse2id, fine2coarse, web_stats)

    print(f"Done. CUB labels: {cub_written}")
    print(f"Done. web stats: {web_stats}")
    print(f"Processed root: {out_root}")


if __name__ == "__main__":
    main()
