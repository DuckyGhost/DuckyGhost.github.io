# YOLOv11 + ByteTrack 鸟类检测跟踪训练流程

本方案实现：
- CUB-200-2011 原始格式转 YOLO
- web-bird 弱监督框生成（COCO bird 类）
- 层级标签（细类 -> 大类）
- 训练时 batch-level `CUB:web-bird = 1:2`
- CUB 有放回采样 + web-bird class-balanced 采样
- 验证视频 ByteTrack 可视化 + ROI 关键帧导出
- 论文 PNG 图导出

## 1) 环境

```bash
pip install ultralytics opencv-python pillow pyyaml pandas matplotlib
```

## 2) 数据准备

将数据放在：
- `model/CUB-200-2011.tar`
- `model/web-bird.tar.gz`

执行：

```bash
python prepare_bird_data.py --model-dir model --out-dir model/processed --val-ratio 0.1 --web-conf 0.25
```

输出：
- `model/processed/dataset.yaml`
- `model/processed/hierarchy_map.json`
- `model/processed/webbird_pseudobox_stats.json`

> 可手动编辑 `hierarchy_map.json` 以修正大类映射。

## 3) 训练

```bash
python train_yolov11_birds.py \
  --dataset-root model/processed \
  --model yolo11l.pt \
  --epochs 100 \
  --batch 30 \
  --imgsz 960 \
  --steps-per-epoch 200 \
  --project runs/bird_train \
  --name yolo11_bird_hier
```

说明：
- `--batch` 必须能被 3 整除，用于严格 1:2（CUB:web）
- 最终权重在 `runs/bird_train/yolo11_bird_hier/weights/last.pt`

## 4) 验证视频 + 跟踪可视化

```bash
python val_track_visualize.py \
  --weights runs/bird_train/yolo11_bird_hier/weights/last.pt \
  --source your_val_video.mp4 \
  --out-dir runs/bird_val_track
```

输出：
- 跟踪可视化视频：`runs/bird_val_track/vis/`
- 轨迹日志：`runs/bird_val_track/tracks.json`
- ROI关键帧：`runs/bird_val_track/roi_keyframes/`
- 每轨最高置信ROI：`runs/bird_val_track/roi_best_per_track/`

## 5) 论文图生成（PNG）

```bash
python plot_paper_figures.py --run-dir runs/bird_train/yolo11_bird_hier --out-dir runs/paper_figures
```

输出：
- `paper_training_curves.png`
- `paper_confusion_matrix.png`
- `paper_pr_curve.png`
- `paper_f1_curve.png`
- `paper_precision_curve.png`
- `paper_recall_curve.png`


## 6) AutoDL 一键脚本（推荐）

已提供 `run_autodl_train.sh`，可在 AutoDL 直接一键跑通“数据准备+训练”，并带有 OOM 自动降档重试。

```bash
bash run_autodl_train.sh
```

常用环境变量（可选覆盖默认值）：

```bash
MODEL_DIR=model
OUT_DIR=model/processed
RUN_PROJECT=runs/bird_train
RUN_NAME=yolo11_bird_hier
MODEL_WEIGHTS=yolo11l.pt
EPOCHS=100
STEPS_PER_EPOCH=200
BATCH=30
IMGSZ=960
PREPARE_DATA=1
bash run_autodl_train.sh
```

说明：
- 若不设置 `BATCH/IMGSZ`，脚本会根据 `nvidia-smi` 检测显存自动选择。
- `BATCH` 必须可被 3 整除（保证 CUB:web = 1:2）。
- 若训练失败（常见为显存不足），脚本会自动降 `BATCH/IMGSZ` 并重试。
