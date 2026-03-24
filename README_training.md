# YOLOv11 + ByteTrack 鸟类训练（重写稳定版）

## 1. 数据集放置（压缩包）
放在 `model/` 目录下：
- `CUB_200_2011.tgz`（也支持 `CUB-200-2011.tar` / `CUB_200_2011.tar.gz`）
- `web-bird.tar.gz`（也支持 `web_bird.tar.gz` / `webbird.tar.gz`）

## 2. 环境安装
```bash
pip install ultralytics opencv-python pillow pyyaml pandas matplotlib
```

## 3. 数据准备（按你给的解压层级）
```bash
python prepare_bird_data.py \
  --model-dir model \
  --out-dir model/processed \
  --pseudo-model yolo11n.pt \
  --web-conf 0.20
```

输出：
- `model/processed/cub/...`（CUB转YOLO）
- `model/processed/web/...`（web-bird伪框）
- `model/processed/hierarchy_map.json`
- `model/processed/class_names.json`
- `model/processed/webbird_pseudobox_stats.json`

## 4. 训练
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
- 训练前会构建 `model/processed/mixed`。
- 采样策略：每个构建批次按 `CUB:web = 1:2` 组织，CUB有放回，web按类别均衡采样。
- `batch` 必须可被 3 整除。

## 5. 验证跟踪可视化
```bash
python val_track_visualize.py \
  --weights runs/bird_train/yolo11_bird_hier/weights/last.pt \
  --source your_val_video.mp4 \
  --out-dir runs/bird_val_track
```

## 6. 论文图导出（PNG）
```bash
python plot_paper_figures.py --run-dir runs/bird_train/yolo11_bird_hier --out-dir runs/paper_figures
```

## 7. AutoDL 一键执行
```bash
bash run_autodl_train.sh
```
常用覆盖参数：
```bash
PSEUDO_MODEL=yolo11n.pt WEB_CONF=0.20 EPOCHS=100 BATCH=30 IMGSZ=960 bash run_autodl_train.sh
```
