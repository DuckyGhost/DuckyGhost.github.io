#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

MODEL_DIR="${MODEL_DIR:-model}"
OUT_DIR="${OUT_DIR:-model/processed}"
RUN_PROJECT="${RUN_PROJECT:-runs/bird_train}"
RUN_NAME="${RUN_NAME:-yolo11_bird_hier}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-yolo11l.pt}"
PSEUDO_MODEL="${PSEUDO_MODEL:-yolo11n.pt}"
EPOCHS="${EPOCHS:-100}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-200}"
WORKERS="${WORKERS:-8}"
WEB_CONF="${WEB_CONF:-0.20}"
SEED="${SEED:-42}"
PREPARE_DATA="${PREPARE_DATA:-1}"

log() { echo "[run_autodl] $*"; }

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

CUB_ARCHIVE="$(find_first_existing "$MODEL_DIR" "CUB_200_2011.tgz" "CUB-200-2011.tar" "CUB_200_2011.tar.gz" || true)"
WEB_ARCHIVE="$(find_first_existing "$MODEL_DIR" "web-bird.tar.gz" "web_bird.tar.gz" "webbird.tar.gz" || true)"

if [[ -z "$CUB_ARCHIVE" || -z "$WEB_ARCHIVE" ]]; then
  echo "[ERROR] Missing dataset archives in $MODEL_DIR"
  echo "Need CUB_200_2011.tgz (or CUB-200-2011.tar) and web-bird.tar.gz"
  exit 1
fi

GPU_MEM_MB="0"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
fi

if [[ -z "${BATCH:-}" || -z "${IMGSZ:-}" ]]; then
  if [[ "$GPU_MEM_MB" -ge 28000 ]]; then
    BATCH=30; IMGSZ=960
  elif [[ "$GPU_MEM_MB" -ge 18000 ]]; then
    BATCH=24; IMGSZ=896
  else
    BATCH=12; IMGSZ=640
  fi
else
  BATCH="${BATCH}"; IMGSZ="${IMGSZ}"
fi

BATCH=$((BATCH - BATCH % 3))
[[ "$BATCH" -lt 3 ]] && BATCH=3

log "GPU=${GPU_MEM_MB}MB BATCH=${BATCH} IMGSZ=${IMGSZ}"
log "CUB=$(basename "$CUB_ARCHIVE") WEB=$(basename "$WEB_ARCHIVE")"

if [[ "$PREPARE_DATA" == "1" ]]; then
  log "[1/2] Preparing dataset"
  python prepare_bird_data.py \
    --model-dir "$MODEL_DIR" \
    --out-dir "$OUT_DIR" \
    --pseudo-model "$PSEUDO_MODEL" \
    --web-conf "$WEB_CONF" \
    --seed "$SEED"
fi

log "[2/2] Training"
python train_yolov11_birds.py \
  --dataset-root "$OUT_DIR" \
  --model "$MODEL_WEIGHTS" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --imgsz "$IMGSZ" \
  --steps-per-epoch "$STEPS_PER_EPOCH" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  --project "$RUN_PROJECT" \
  --name "$RUN_NAME"

LAST_PT="$RUN_PROJECT/$RUN_NAME/weights/last.pt"
log "Done: $LAST_PT"
