#!/usr/bin/env bash
# U-Net v2 / PolypSeg 训练（Linux，论文配方，RTX 3090）
#
#   bash train.sh                                        # 用下面的默认路径
#   bash train.sh <train_dir> <test_dir> <save_dir>      # 临时指定
#   bash train.sh <train> <test> <save> --augmentation   # 后面还能追加别的参数
#
set -e
cd "$(dirname "$0")"

# ==================== 按需修改这三行 ====================
PY="${PY:-/root/miniconda3/envs/pc/bin/python}"
DATA="${DATA:-$HOME/unetv2/unetv2_data}"
SAVE="${SAVE:-$HOME/unetv2/runs}"
# =======================================================

TRAIN_PATH="${1:-$DATA/TrainDataset}"
TEST_PATH="${2:-$DATA/TestDataset}"
SAVE_DIR="${3:-$SAVE}"
EXTRA=("${@:4}")

if [ ! -x "$PY" ]; then
  echo "[ERROR] 找不到 python: $PY"
  echo "        改 train.sh 开头的 PY，或者用 PY=/path/to/python bash train.sh"
  exit 1
fi
if [ ! -d "$TRAIN_PATH/images" ]; then
  echo "[ERROR] 找不到训练数据: $TRAIN_PATH/images"
  echo "        先跑:  python make_data_layout.py --help"
  exit 1
fi

mkdir -p "$SAVE_DIR"

echo "=============================================================="
echo "  U-Net v2 / PolypSeg   -   TRAINING"
echo "  python : $PY"
echo "  train  : $TRAIN_PATH"
echo "  test   : $TEST_PATH"
echo "  save   : $SAVE_DIR"
echo "  recipe : Adam lr 1e-3, poly decay 0.9, 300 epochs"
echo "           batch 16, fp32 + TF32, workers 8, val every 5"
echo "  results: $SAVE_DIR/UNetV2/results_log.txt"
echo "  best   : $SAVE_DIR/UNetV2/best/best.pth"
echo "=============================================================="

exec "$PY" -u Train.py \
  --train_path "$TRAIN_PATH" \
  --test_path  "$TEST_PATH" \
  --train_save "$SAVE_DIR" \
  --epoch 300 --batchsize 16 --lr 1e-3 \
  --optimizer Adam --scheduler poly --poly_power 0.9 \
  --val_every 5 --num_workers 8 --device auto --seed 1234 \
  "${EXTRA[@]}"
