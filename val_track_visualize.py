#!/usr/bin/env python3
"""Validation-time video inference + ByteTrack visualization + ROI keyframes."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=str)
    ap.add_argument("--source", required=True, type=str, help="video file or folder")
    ap.add_argument("--out-dir", default="runs/bird_val_track", type=str)
    ap.add_argument("--conf", default=0.25, type=float)
    ap.add_argument("--iou", default=0.5, type=float)
    ap.add_argument("--imgsz", default=960, type=int)
    ap.add_argument("--roi-interval", default=30, type=int, help="save ROI every N frames per track")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    roi_dir = out_dir / "roi_keyframes"
    ensure_dir(roi_dir)

    model = YOLO(args.weights)

    tracks_log = []
    best_roi = defaultdict(lambda: {"conf": -1.0, "img": None, "meta": None})

    results = model.track(
        source=args.source,
        stream=True,
        tracker="bytetrack.yaml",
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        persist=True,
        save=True,
        project=str(out_dir),
        name="vis",
        exist_ok=True,
        verbose=False,
    )

    frame_idx = 0
    for r in results:
        frame = r.orig_img
        h, w = frame.shape[:2]
        if r.boxes is None or len(r.boxes) == 0:
            frame_idx += 1
            continue

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [-1] * len(boxes_xyxy)

        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1, x2, y2 = clamp(x1, 0, w - 1), clamp(y1, 0, h - 1), clamp(x2, 0, w - 1), clamp(y2, 0, h - 1)
            tid = int(ids[i])
            c = int(cls[i])
            cf = float(conf[i])

            tracks_log.append(
                {
                    "frame": frame_idx,
                    "track_id": tid,
                    "class_id": c,
                    "conf": cf,
                    "bbox_xyxy": [x1, y1, x2, y2],
                }
            )

            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                if frame_idx % args.roi_interval == 0 and tid >= 0:
                    roi_path = roi_dir / f"track{tid:04d}_f{frame_idx:06d}_c{c}_p{cf:.3f}.png"
                    cv2.imwrite(str(roi_path), crop)

                if tid >= 0 and cf > best_roi[tid]["conf"]:
                    best_roi[tid] = {
                        "conf": cf,
                        "img": crop.copy(),
                        "meta": {"frame": frame_idx, "class_id": c, "bbox_xyxy": [x1, y1, x2, y2]},
                    }

        frame_idx += 1

    best_dir = out_dir / "roi_best_per_track"
    ensure_dir(best_dir)
    best_meta = {}
    for tid, info in best_roi.items():
        if info["img"] is None:
            continue
        p = best_dir / f"track{tid:04d}_best.png"
        cv2.imwrite(str(p), info["img"])
        best_meta[tid] = info["meta"]

    (out_dir / "tracks.json").write_text(json.dumps(tracks_log, indent=2), encoding="utf-8")
    (out_dir / "roi_best_meta.json").write_text(json.dumps(best_meta, indent=2), encoding="utf-8")
    print("Done. Visualized video under:", out_dir / "vis")


if __name__ == "__main__":
    main()
