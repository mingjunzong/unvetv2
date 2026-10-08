#!/usr/bin/env bash
# 先跑这个！5 个 epoch 快速确认 loss 在降、dsc 在涨，再去跑 300 epoch。
# 如果 5 轮后 dsc 还是 0.00，那百分之百是数据问题，跑 300 轮也是白跑。
#
#   bash verify.sh
#   bash verify.sh <train_dir> <test_dir> <save_dir>
#
set -e
cd "$(dirname "$0")"

PY="${PY:-/root/miniconda3/envs/pc/bin/python}"
DATA="${DATA:-$HOME/unetv2/unetv2_data}"
SAVE="${SAVE:-$HOME/unetv2/verify}"

TRAIN_PATH="${1:-$DATA/TrainDataset}"
TEST_PATH="${2:-$DATA/TestDataset}"
SAVE_DIR="${3:-$SAVE}"
EXTRA=("${@:4}")

if [ ! -x "$PY" ]; then
  echo "[ERROR] 找不到 python: $PY"; exit 1
fi
mkdir -p "$SAVE_DIR"

echo "=============================================================="
echo "  U-Net v2 / PolypSeg   -   QUICK VERIFY  (only 5 epochs)"
echo "=============================================================="
echo "  看 dsc 有没有离开 0.00 : $SAVE_DIR/UNetV2/results_log.txt"
echo "=============================================================="

exec "$PY" -u Train.py \
  --train_path "$TRAIN_PATH" \
  --test_path  "$TEST_PATH" \
  --train_save "$SAVE_DIR" \
  --epoch 5 --batchsize 16 --lr 1e-3 \
  --optimizer Adam --scheduler poly --poly_power 0.9 \
  --val_every 1 --num_workers 8 --device auto --seed 1234 \
  "${EXTRA[@]}"
