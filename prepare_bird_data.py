#!/usr/bin/env python3
"""
Prepare CUB-200-2011 + web-bird into YOLO format with:
- CUB conversion from original txt annotations
- Weakly-supervised pseudo-box generation for web-bird via COCO bird detector
- Hierarchical label map (fine -> coarse)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def safe_extract_tar(archive_path: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as tar:
        tar.extractall(dst)


def normalize_name(name: str) -> str:
    return name.lower().replace("_", " ").replace("-", " ")


def coarse_from_class_name(name: str) -> str:
    """Heuristic hierarchy mapping for bird classes.
    Users can later edit generated hierarchy_map.json for better taxonomy.
    """
    n = normalize_name(name)
    rules = {
        "gull": ["gull", "tern"],
        "waterfowl": ["duck", "goose", "swan", "grebe"],
        "shorebird": ["sandpiper", "plover", "willet", "snipe", "oystercatcher"],
        "wader": ["ibis", "heron", "egret", "bittern", "crane", "stork"],
        "raptor": ["hawk", "eagle", "falcon", "kite", "vulture", "owl"],
        "songbird": [
            "sparrow",
            "warbler",
            "finch",
            "vireo",
            "thrush",
            "oriole",
            "wren",
            "chickadee",
            "nuthatch",
            "swallow",
            "blackbird",
            "jay",
            "crow",
            "starling",
            "kinglet",
        ],
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


def parse_cub(cub_root: Path, out_root: Path, val_ratio: float, seed: int) -> Tuple[Dict[str, int], Dict[str, str]]:
    images_txt = cub_root / "images.txt"
    bboxes_txt = cub_root / "bounding_boxes.txt"
    classes_txt = cub_root / "classes.txt"
    img_class_txt = cub_root / "image_class_labels.txt"

    for p in [images_txt, bboxes_txt, classes_txt, img_class_txt]:
        if not p.exists():
            raise FileNotFoundError(f"Missing CUB file: {p}")

    id2path: Dict[int, str] = {}
    with images_txt.open("r", encoding="utf-8") as f:
        for line in f:
            i, rel = line.strip().split(" ", 1)
            id2path[int(i)] = rel

    id2bbox: Dict[int, Tuple[float, float, float, float]] = {}
    with bboxes_txt.open("r", encoding="utf-8") as f:
        for line in f:
            i, x, y, w, h = line.strip().split()
            id2bbox[int(i)] = (float(x), float(y), float(w), float(h))

    class_id2name: Dict[int, str] = {}
    with classes_txt.open("r", encoding="utf-8") as f:
        for line in f:
            i, name = line.strip().split(" ", 1)
            class_id2name[int(i)] = name.split(".", 1)[-1]

    id2class: Dict[int, int] = {}
    with img_class_txt.open("r", encoding="utf-8") as f:
        for line in f:
            i, cid = line.strip().split()
            id2class[int(i)] = int(cid)

    entries = list(id2path.keys())
    rnd = random.Random(seed)
    rnd.shuffle(entries)
    n_val = int(len(entries) * val_ratio)
    val_set = set(entries[:n_val])

    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    coarse_names = sorted({coarse_from_class_name(v) for v in class_id2name.values()})
    coarse2id = {c: i for i, c in enumerate(coarse_names)}

    fine2coarse: Dict[str, str] = {}
    for k, v in class_id2name.items():
        fine2coarse[v] = coarse_from_class_name(v)

    for img_id in entries:
        rel_path = id2path[img_id]
        src_img = cub_root / "images" / rel_path
        split = "val" if img_id in val_set else "train"

        class_name = class_id2name[id2class[img_id]]
        coarse = fine2coarse[class_name]
        coarse_id = coarse2id[coarse]

        dst_img_name = rel_path.replace("/", "__")
        dst_img = out_root / "images" / split / dst_img_name
        shutil.copy2(src_img, dst_img)

        with Image.open(dst_img) as im:
            w_img, h_img = im.size

        x, y, bw, bh = id2bbox[img_id]
        x_c = (x + bw / 2.0) / w_img
        y_c = (y + bh / 2.0) / h_img
        w_n = bw / w_img
        h_n = bh / h_img

        lbl = out_root / "labels" / split / f"{Path(dst_img_name).stem}.txt"
        with lbl.open("w", encoding="utf-8") as f:
            f.write(f"{coarse_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

    return coarse2id, fine2coarse


def load_webbird_classes(web_root: Path) -> List[str]:
    classes = [p.name for p in web_root.iterdir() if p.is_dir()]
    classes.sort()
    return classes


def generate_webbird_pseudoboxes(
    web_root: Path,
    out_root: Path,
    coarse2id: Dict[str, int],
    val_ratio: float,
    seed: int,
    conf_thres: float,
) -> None:
    from ultralytics import YOLO

    model = YOLO("yolo11x.pt")

    all_images: List[Tuple[str, Path]] = []
    for cls in load_webbird_classes(web_root):
        for p in (web_root / cls).glob("*.*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                all_images.append((cls, p))

    rnd = random.Random(seed)
    rnd.shuffle(all_images)
    n_val = int(len(all_images) * val_ratio)
    val_names = {str(p) for _, p in all_images[:n_val]}

    for split in ["train", "val"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0

    for cls, img_path in all_images:
        split = "val" if str(img_path) in val_names else "train"

        pred = model.predict(source=str(img_path), conf=conf_thres, verbose=False)
        boxes = pred[0].boxes
        bird_boxes = []
        for b in boxes:
            c = int(b.cls.item())
            # COCO class 14 => bird
            if c == 14:
                bird_boxes.append(b)

        if not bird_boxes:
            dropped += 1
            continue

        with Image.open(img_path) as im:
            w_img, h_img = im.size

        coarse = coarse_from_class_name(cls)
        if coarse not in coarse2id:
            coarse2id[coarse] = max(coarse2id.values(), default=-1) + 1
        coarse_id = coarse2id[coarse]

        dst_name = f"web__{cls}__{img_path.stem}{img_path.suffix.lower()}"
        dst_img = out_root / "images" / split / dst_name
        shutil.copy2(img_path, dst_img)

        lbl = out_root / "labels" / split / f"{Path(dst_name).stem}.txt"
        with lbl.open("w", encoding="utf-8") as f:
            for b in bird_boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                bw = x2 - x1
                bh = y2 - y1
                x_c = (x1 + x2) / 2.0 / w_img
                y_c = (y1 + y2) / 2.0 / h_img
                w_n = bw / w_img
                h_n = bh / h_img
                f.write(f"{coarse_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
        kept += 1

    stats = {"webbird_kept": kept, "webbird_dropped_no_bird_box": dropped}
    with (out_root / "webbird_pseudobox_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def write_dataset_yaml(out_root: Path, coarse2id: Dict[str, int]) -> None:
    names = [None] * len(coarse2id)
    for k, v in coarse2id.items():
        names[v] = k

    yaml_text = "\n".join(
        [
            f"path: {out_root.resolve()}",
            "train: images/train",
            "val: images/val",
            f"nc: {len(names)}",
            f"names: {names}",
            "",
        ]
    )
    (out_root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model", type=str)
    ap.add_argument("--out-dir", default="model/processed", type=str)
    ap.add_argument("--val-ratio", default=0.1, type=float)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--web-conf", default=0.25, type=float)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)

    cub_tar = model_dir / "CUB-200-2011.tar"
    web_tar = model_dir / "web-bird.tar.gz"

    extract_root = model_dir / "_extracted"
    cub_ext = extract_root / "CUB-200-2011"
    web_ext = extract_root / "web-bird"

    if not cub_ext.exists():
        safe_extract_tar(cub_tar, extract_root)
    if not web_ext.exists():
        safe_extract_tar(web_tar, extract_root)

    out_dir.mkdir(parents=True, exist_ok=True)

    coarse2id, fine2coarse = parse_cub(cub_ext, out_dir, args.val_ratio, args.seed)
    generate_webbird_pseudoboxes(web_ext, out_dir, coarse2id, args.val_ratio, args.seed, args.web_conf)
    write_dataset_yaml(out_dir, coarse2id)

    hierarchy = {
        "fine_to_coarse": fine2coarse,
        "coarse_to_id": {k: int(v) for k, v in sorted(coarse2id.items(), key=lambda kv: kv[1])},
        "sampling_policy": {
            "ratio": "CUB:web-bird = 1:2 (batch-level)",
            "cub_replacement": True,
            "webbird_class_balanced": True,
        },
    }
    with (out_dir / "hierarchy_map.json").open("w", encoding="utf-8") as f:
        json.dump(hierarchy, f, indent=2, ensure_ascii=False)

    # A small summary for quick review
    lbl_train = list((out_dir / "labels" / "train").glob("*.txt"))
    cls_counter = Counter()
    for p in lbl_train:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cls_counter[int(line.split()[0])] += 1
    summary = {
        "train_label_files": len(lbl_train),
        "train_boxes_per_class": dict(sorted(cls_counter.items())),
    }
    with (out_dir / "prepare_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done. dataset.yaml:", out_dir / "dataset.yaml")


if __name__ == "__main__":
    main()
