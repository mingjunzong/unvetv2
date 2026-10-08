#!/usr/bin/env bash
# U-Net v2 / PolypSeg 测试：用 best.pth 在测试集上出 dsc/iou/mae 表
#
#   bash test.sh                          # 用下面的默认路径
#   bash test.sh <test_dir> <ckpt_path>   # 临时指定
#
set -e
cd "$(dirname "$0")"

PY="${PY:-/root/miniconda3/envs/pc/bin/python}"
DATA="${DATA:-$HOME/unetv2/unetv2_data}"
SAVE="${SAVE:-$HOME/unetv2/runs}"

TEST_PATH="${1:-$DATA/TestDataset}"
CKPT="${2:-$SAVE/UNetV2/best/best.pth}"
EXTRA=("${@:3}")

if [ ! -x "$PY" ]; then
  echo "[ERROR] 找不到 python: $PY"; exit 1
fi
if [ ! -f "$CKPT" ]; then
  echo "[ERROR] 找不到权重: $CKPT"
  echo "        先跑 bash train.sh"
  exit 1
fi

echo "=============================================================="
echo "  U-Net v2 / PolypSeg   -   TEST"
echo "  test : $TEST_PATH"
echo "  ckpt : $CKPT"
echo "=============================================================="

exec "$PY" -u Test.py \
  --test_path "$TEST_PATH" \
  --pth_path  "$CKPT" \
  --save_path "$SAVE/test_result" \
  --device auto --save_masks \
  "${EXTRA[@]}"
