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


def resolve_first_existing(base: Path, candidates: List[str]) -> Path:
    for name in candidates:
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of candidates exist under {base}: {candidates}")


def find_cub_root(extract_root: Path) -> Path:
    """Find CUB root folder containing required metadata txt files."""
    required = {"images.txt", "bounding_boxes.txt", "classes.txt", "image_class_labels.txt"}
    common_candidates = [
        extract_root / "CUB-200-2011",
        extract_root / "CUB_200_2011",
        extract_root / "CUB_200_2011" / "CUB_200_2011",
    ]
    for c in common_candidates:
        if c.exists() and required.issubset({p.name for p in c.glob("*.txt")}):
            return c

    # Fallback: search a few levels deep
    for p in extract_root.glob("**/*"):
        if not p.is_dir():
            continue
        txt_names = {x.name for x in p.glob("*.txt")}
        if required.issubset(txt_names):
            return p
    raise FileNotFoundError(
        f"Unable to locate CUB root under {extract_root}. "
        f"Expected files: {sorted(required)}"
    )


def find_webbird_root(extract_root: Path) -> Path:
    """Find web-bird ImageFolder root (each child dir is a class)."""
    common = [extract_root / "web-bird", extract_root / "web_bird", extract_root / "webbird"]
    for c in common:
        if c.exists() and any(x.is_dir() for x in c.iterdir()):
            return c

    # Fallback: choose the directory with most subdirectories (class folders)
    best = None
    best_cnt = -1
    for p in extract_root.glob("**/*"):
        if p.is_dir():
            cnt = sum(1 for x in p.iterdir() if x.is_dir())
            if cnt > best_cnt:
                best, best_cnt = p, cnt
    if best is None:
        raise FileNotFoundError(f"Unable to locate web-bird root under {extract_root}")
    return best


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


def collect_webbird_images(web_root: Path) -> List[Tuple[str, Path]]:
    """Collect (class_name, image_path) from ImageFolder-like layouts.

    Supports:
    - web_root/<class>/*.jpg
    - web_root/<split>/<class>/*.jpg
    - deeper nested folders (uses parent dir name as class)
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # 1) Try one-level ImageFolder first
    pairs: List[Tuple[str, Path]] = []
    for class_dir in [p for p in web_root.iterdir() if p.is_dir()]:
        imgs = [p for p in class_dir.glob("*.*") if p.suffix.lower() in exts]
        for p in imgs:
            pairs.append((class_dir.name, p))
    if pairs:
        return pairs

    # 2) Fallback recursive search (split/class/image etc.)
    for p in web_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            pairs.append((p.parent.name, p))
    return pairs


def generate_webbird_pseudoboxes(
    web_root: Path,
    out_root: Path,
    coarse2id: Dict[str, int],
    val_ratio: float,
    seed: int,
    conf_thres: float,
    fallback_fullbox: bool,
) -> None:
    from ultralytics import YOLO

    model = YOLO("yolo11x.pt")

    all_images = collect_webbird_images(web_root)
    if not all_images:
        raise RuntimeError(
            f"No images found in web-bird root: {web_root}. "
            "Please check extracted folder structure."
        )

    rnd = random.Random(seed)
    rnd.shuffle(all_images)
    n_val = int(len(all_images) * val_ratio)
    val_names = {str(p) for _, p in all_images[:n_val]}

    for split in ["train", "val"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    broken_or_unreadable = 0

    for cls, img_path in all_images:
        split = "val" if str(img_path) in val_names else "train"

        try:
            # Pre-check readability to avoid downstream OpenCV/Numpy stack errors
            with Image.open(img_path) as im:
                im.verify()
        except Exception:
            broken_or_unreadable += 1
            continue

        try:
            pred = model.predict(source=str(img_path), conf=conf_thres, verbose=False)
            boxes = pred[0].boxes
        except Exception:
            broken_or_unreadable += 1
            continue
        bird_boxes = []
        for b in boxes:
            c = int(b.cls.item())
            # COCO class 14 => bird
            if c == 14:
                bird_boxes.append(b)

        if not bird_boxes:
            if fallback_fullbox:
                bird_boxes = [None]  # sentinel: full-image box
            else:
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
                if b is None:
                    x1, y1, x2, y2 = 0.0, 0.0, float(w_img), float(h_img)
                else:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                bw = x2 - x1
                bh = y2 - y1
                x_c = (x1 + x2) / 2.0 / w_img
                y_c = (y1 + y2) / 2.0 / h_img
                w_n = bw / w_img
                h_n = bh / h_img
                f.write(f"{coarse_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
        kept += 1

    stats = {
        "webbird_total_found_images": len(all_images),
        "webbird_kept": kept,
        "webbird_dropped_no_bird_box": dropped,
        "webbird_broken_or_unreadable": broken_or_unreadable,
        "fallback_fullbox_enabled": fallback_fullbox,
    }
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
    ap.add_argument(
        "--web-fallback-fullbox",
        action="store_true",
        help="If no bird is detected on a web image, use whole image as fallback bbox.",
    )
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)

    cub_tar = resolve_first_existing(model_dir, ["CUB-200-2011.tar", "CUB_200_2011.tgz", "CUB_200_2011.tar.gz"])
    web_tar = resolve_first_existing(model_dir, ["web-bird.tar.gz", "web_bird.tar.gz", "webbird.tar.gz"])

    extract_root = model_dir / "_extracted"
    extract_root.mkdir(parents=True, exist_ok=True)

    if not any((extract_root / n).exists() for n in ["CUB-200-2011", "CUB_200_2011"]):
        safe_extract_tar(cub_tar, extract_root)
    if not any((extract_root / n).exists() for n in ["web-bird", "web_bird", "webbird"]):
        safe_extract_tar(web_tar, extract_root)

    cub_ext = find_cub_root(extract_root)
    web_ext = find_webbird_root(extract_root)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Resolved CUB root: {cub_ext}")
    print(f"Resolved web-bird root: {web_ext}")

    coarse2id, fine2coarse = parse_cub(cub_ext, out_dir, args.val_ratio, args.seed)
    generate_webbird_pseudoboxes(
        web_ext,
        out_dir,
        coarse2id,
        args.val_ratio,
        args.seed,
        args.web_conf,
        args.web_fallback_fullbox,
    )
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
