#!/usr/bin/env bash
set -euo pipefail

# AutoDL one-click training launcher for YOLOv11 bird pipeline.
# Features:
# - optional conda env bootstrap
# - auto GPU memory probe -> selects safer default batch/imgsz
# - optional weak-supervision prep skip
# - OOM fallback retries with smaller batch size

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

MODEL_DIR="${MODEL_DIR:-model}"
OUT_DIR="${OUT_DIR:-model/processed}"
RUN_PROJECT="${RUN_PROJECT:-runs/bird_train}"
RUN_NAME="${RUN_NAME:-yolo11_bird_hier}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-yolo11l.pt}"
EPOCHS="${EPOCHS:-100}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-200}"
WORKERS="${WORKERS:-8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
WEB_CONF="${WEB_CONF:-0.25}"
WEB_FALLBACK_FULLBOX="${WEB_FALLBACK_FULLBOX:-1}"  # 1 enable full-image box when detector misses
SEED="${SEED:-42}"
PREPARE_DATA="${PREPARE_DATA:-1}"  # 1: run prepare, 0: skip

log() { echo "[run_autodl] $*"; }

check_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1"; exit 1; }
}

check_cmd python

find_first_existing() {
  local base="$1"; shift
  for n in "$@"; do
    if [[ -f "$base/$n" ]]; then
      echo "$base/$n"
      return 0
    fi
  done
  return 1
}

CUB_ARCHIVE="$(find_first_existing "$MODEL_DIR" "CUB-200-2011.tar" "CUB_200_2011.tgz" "CUB_200_2011.tar.gz" || true)"
WEB_ARCHIVE="$(find_first_existing "$MODEL_DIR" "web-bird.tar.gz" "web_bird.tar.gz" "webbird.tar.gz" || true)"

if [[ -z "${CUB_ARCHIVE}" ]]; then
  echo "Missing CUB archive in $MODEL_DIR (supported: CUB-200-2011.tar / CUB_200_2011.tgz / CUB_200_2011.tar.gz)"
  exit 1
fi
if [[ -z "${WEB_ARCHIVE}" ]]; then
  echo "Missing web-bird archive in $MODEL_DIR (supported: web-bird.tar.gz / web_bird.tar.gz / webbird.tar.gz)"
  exit 1
fi

GPU_MEM_MB="0"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
fi

# Default auto policy (can be overridden by env BATCH/IMGSZ)
if [[ -z "${BATCH:-}" || -z "${IMGSZ:-}" ]]; then
  if [[ "$GPU_MEM_MB" -ge 28000 ]]; then
    AUTO_BATCH=30
    AUTO_IMGSZ=960
  elif [[ "$GPU_MEM_MB" -ge 18000 ]]; then
    AUTO_BATCH=24
    AUTO_IMGSZ=896
  elif [[ "$GPU_MEM_MB" -ge 11000 ]]; then
    AUTO_BATCH=18
    AUTO_IMGSZ=768
  else
    AUTO_BATCH=12
    AUTO_IMGSZ=640
  fi
  # ensure divisible by 3 for strict 1:2 CUB:web ratio
  AUTO_BATCH=$((AUTO_BATCH - AUTO_BATCH % 3))
  [[ "$AUTO_BATCH" -lt 3 ]] && AUTO_BATCH=3
  BATCH="${BATCH:-$AUTO_BATCH}"
  IMGSZ="${IMGSZ:-$AUTO_IMGSZ}"
else
  BATCH="${BATCH}"
  IMGSZ="${IMGSZ}"
fi

if (( BATCH % 3 != 0 )); then
  echo "BATCH must be divisible by 3, got $BATCH"
  exit 1
fi

log "GPU total memory (MB): ${GPU_MEM_MB}"
log "Using BATCH=${BATCH}, IMGSZ=${IMGSZ}, EPOCHS=${EPOCHS}, STEPS_PER_EPOCH=${STEPS_PER_EPOCH}"
log "Using archives: CUB=$(basename "$CUB_ARCHIVE"), WEB=$(basename "$WEB_ARCHIVE")"

if [[ "$PREPARE_DATA" == "1" ]]; then
  log "[1/3] Preparing dataset"
  python prepare_bird_data.py \
    --model-dir "$MODEL_DIR" \
    --out-dir "$OUT_DIR" \
    --val-ratio "$VAL_RATIO" \
    --seed "$SEED" \
    --web-conf "$WEB_CONF" \
    $([[ "$WEB_FALLBACK_FULLBOX" == "1" ]] && echo "--web-fallback-fullbox")
else
  log "[1/3] Skip data preparation (PREPARE_DATA=0)"
fi

# OOM-aware training retries
try_train() {
  local batch="$1"
  local imgsz="$2"
  set +e
  python train_yolov11_birds.py \
    --dataset-root "$OUT_DIR" \
    --project "$RUN_PROJECT" \
    --name "$RUN_NAME" \
    --model "$MODEL_WEIGHTS" \
    --epochs "$EPOCHS" \
    --batch "$batch" \
    --imgsz "$imgsz" \
    --steps-per-epoch "$STEPS_PER_EPOCH" \
    --seed "$SEED" \
    --workers "$WORKERS"
  local code=$?
  set -e
  return $code
}

log "[2/3] Training start"
if try_train "$BATCH" "$IMGSZ"; then
  log "Training finished with BATCH=$BATCH IMGSZ=$IMGSZ"
else
  log "Training failed, trying safer settings"
  BATCH2=$((BATCH - 6))
  [[ "$BATCH2" -lt 6 ]] && BATCH2=6
  BATCH2=$((BATCH2 - BATCH2 % 3))
  IMGSZ2=$((IMGSZ - 128))
  [[ "$IMGSZ2" -lt 640 ]] && IMGSZ2=640

  log "Retry #1 with BATCH=$BATCH2 IMGSZ=$IMGSZ2"
  if try_train "$BATCH2" "$IMGSZ2"; then
    BATCH="$BATCH2"; IMGSZ="$IMGSZ2"
  else
    BATCH3=6
    IMGSZ3=640
    log "Retry #2 with BATCH=$BATCH3 IMGSZ=$IMGSZ3"
    try_train "$BATCH3" "$IMGSZ3"
    BATCH="$BATCH3"; IMGSZ="$IMGSZ3"
  fi
fi

LAST_PT="$RUN_PROJECT/$RUN_NAME/weights/last.pt"
if [[ ! -f "$LAST_PT" ]]; then
  echo "Training completed but $LAST_PT not found"
  exit 1
fi

log "[3/3] Training done"
log "Checkpoint: $LAST_PT"
log "Optional next step (tracking validation):"
echo "python val_track_visualize.py --weights $LAST_PT --source your_val_video.mp4 --out-dir runs/bird_val_track"
log "Optional paper figures:"
echo "python plot_paper_figures.py --run-dir $RUN_PROJECT/$RUN_NAME --out-dir runs/paper_figures"
