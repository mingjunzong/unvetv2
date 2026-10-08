"""从训练日志画出训练曲线，输出 PNG（适合放进汇报材料）。

数据来源（两个都在 <save> 目录下）：
    <save>/UNetV2/results_log.txt   每个评估周期的 dsc / iou / mae
    <save>/train.log                每一步的 loss / lr（nohup 重定向产生的那份）

输出：
    curve.png    2x2 的四联图：Loss / DSC+IoU / MAE / Learning rate
    metrics.csv  每个评估周期的数值表（可以直接丢进 Excel 再画）

用法：
    pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
    python plot_curve.py --save /data/unetv2_runs
    python plot_curve.py --save /data/unetv2_runs --out /data/unetv2_runs/curve.png

注：图里的文字全部用英文 —— 服务器上 matplotlib 默认字体没有中文，
用中文会显示成一堆方块。要中文得先装字体再设 font.sans-serif。
"""
import argparse
import csv
import os
import re

import matplotlib
matplotlib.use('Agg')          # 服务器没有桌面环境，必须用非交互后端
import matplotlib.pyplot as plt

# [E001/300] step    5/  35 ( 14.3%)  loss 2.6507  avg 2.8359  lr 1.00e-03  elapsed ...
RE_STEP = re.compile(
    r'\[E(\d+)/(\d+)\]\s+step\s+(\d+)/\s*(\d+)\s+\([^)]*\)\s+'
    r'loss\s+([-\d.eE+]+)\s+avg\s+([-\d.eE+]+)\s+lr\s+([-\d.eE+]+)')

RE_EPOCH = re.compile(r'^epoch\s+(\d+)')
RE_MEAN = re.compile(r'^mean\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)')
RE_BEST = re.compile(r'best mean dice\s*=\s*([-\d.]+)')


def parse_metrics(path):
    """解析 results_log.txt -> epochs, dsc, iou, mae"""
    epochs, dsc, iou, mae = [], [], [], []
    cur = None
    # utf-8-sig：兼容带 BOM 的文件（比如被 Windows 记事本编辑过）
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            m = RE_EPOCH.match(line.strip())
            if m:
                cur = int(m.group(1))
                continue
            m = RE_MEAN.match(line.strip())
            if m and cur is not None:
                epochs.append(cur)
                dsc.append(float(m.group(1)))
                iou.append(float(m.group(2)))
                mae.append(float(m.group(3)))
                cur = None
    return epochs, dsc, iou, mae


def parse_train_log(path):
    """解析 train.log -> 每一步的 (epoch, step, loss, avg, lr)，以及每轮的 avg loss"""
    steps, losses, avgs, lrs = [], [], [], []
    epoch_last_avg = {}
    total_epochs = None
    with open(path, encoding='utf-8-sig', errors='ignore') as f:
        for line in f:
            m = RE_STEP.search(line)
            if not m:
                continue
            ep = int(m.group(1))
            total_epochs = int(m.group(2))
            steps.append(len(steps) + 1)
            losses.append(float(m.group(5)))
            avgs.append(float(m.group(6)))
            lrs.append(float(m.group(7)))
            epoch_last_avg[ep] = float(m.group(6))
    return steps, losses, avgs, lrs, epoch_last_avg, total_epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', required=True, help='训练输出目录，例如 /data/unetv2_runs')
    ap.add_argument('--out', default=None, help='PNG 输出路径，默认 <save>/curve.png')
    ap.add_argument('--dpi', type=int, default=150)
    opt = ap.parse_args()

    save = os.path.abspath(opt.save)
    metrics_path = os.path.join(save, 'UNetV2', 'results_log.txt')
    out_png = opt.out or os.path.join(save, 'curve.png')
    out_csv = os.path.join(save, 'metrics.csv')

    # train.log 可能在这两个位置：<save>/train.log（nohup 重定向）
    # 或 <save>/UNetV2/train.log（Train.py 自己写的）
    log_candidates = [os.path.join(save, 'train.log'),
                      os.path.join(save, 'UNetV2', 'train.log')]
    log_path = next((p for p in log_candidates if os.path.isfile(p)), log_candidates[0])

    if not os.path.isfile(metrics_path):
        raise SystemExit('[ERROR] 找不到 {}'.format(metrics_path))
    epochs, dsc, iou, mae = parse_metrics(metrics_path)
    if not epochs:
        raise SystemExit('[ERROR] {} 里没解析到指标，检查格式'.format(metrics_path))
    print('[read] {} 个评估周期，epoch {} ~ {}'.format(len(epochs), epochs[0], epochs[-1]))

    have_log = os.path.isfile(log_path)
    if have_log:
        steps, losses, avgs, lrs, epoch_avg, total_ep = parse_train_log(log_path)
        print('[read] train.log 里 {} 条 step 记录'.format(len(steps)))
    else:
        steps = losses = avgs = lrs = []
        epoch_avg = {}
        total_ep = None
        print('[warn] 没找到 {}，跳过 loss / lr 曲线'.format(log_path))

    # ---- CSV ----
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'dsc', 'iou', 'mae'])
        for row in zip(epochs, dsc, iou, mae):
            w.writerow(row)
    print('[write] {}'.format(out_csv))

    # ---- 画图 ----
    best_i = max(range(len(dsc)), key=lambda i: dsc[i])
    has_log = bool(steps)

    if has_log:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        ax_loss, ax_dsc, ax_mae, ax_lr = axes[0][0], axes[0][1], axes[1][0], axes[1][1]
        ax_iou = None
    else:
        # 没有 train.log（比如前台跑的，没重定向）：改成 1x3，不留空面板
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
        ax_dsc, ax_iou, ax_mae = axes[0], axes[1], axes[2]
        ax_loss = ax_lr = None
    fig.suptitle('U-Net v2 / PolypSeg  -  training curves', fontsize=15, fontweight='bold')

    if ax_loss is not None:
        ax = ax_loss
        ax.plot(steps, losses, color='#9ecae1', lw=0.8, alpha=0.9, label='step loss (raw)')
        if epoch_avg:
            ex = sorted(epoch_avg)
            ax.plot(ex, [epoch_avg[e] for e in ex], color='#08519c', lw=2.0,
                    label='epoch mean loss')
        ax.set_xlabel('iteration')
        ax.set_ylabel('loss')
        ax.set_title('Training loss')
        ax.grid(alpha=0.3)
        ax.legend()

    ax = ax_dsc
    ax.plot(epochs, dsc, 'o-', color='#d62728', ms=3, lw=1.8, label='DSC')
    if ax_iou is None:
        ax.plot(epochs, iou, 's-', color='#1f77b4', ms=3, lw=1.8, label='IoU')
    ax.plot(epochs[best_i], dsc[best_i], '*', color='gold', ms=18,
            markeredgecolor='k', zorder=5,
            label='best DSC = {:.4f} @ep{}'.format(dsc[best_i], epochs[best_i]))
    ax.set_xlabel('epoch')
    ax.set_ylabel('DSC')
    ax.set_title('Validation DSC / IoU' if ax_iou is None else 'Validation DSC')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')

    if ax_iou is not None:
        ax = ax_iou
        ax.plot(epochs, iou, 's-', color='#1f77b4', ms=3, lw=1.8)
        ax.set_xlabel('epoch')
        ax.set_ylabel('IoU')
        ax.set_title('Validation IoU')
        ax.grid(alpha=0.3)

    ax = ax_mae
    ax.plot(epochs, mae, '^-', color='#2ca02c', ms=3, lw=1.8)
    ax.set_xlabel('epoch')
    ax.set_ylabel('MAE')
    ax.set_title('Validation MAE (lower is better)')
    ax.grid(alpha=0.3)

    if ax_lr is not None:
        ax = ax_lr
        ax.plot(steps, lrs, color='#ff7f0e', lw=1.6)
        ax.set_xlabel('iteration')
        ax.set_ylabel('learning rate')
        ax.set_title('Learning rate  (poly, power 0.9)')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=opt.dpi)
    print('[write] {}'.format(out_png))

    print()
    print('best DSC = {:.4f} @ epoch {}   (IoU {:.4f}, MAE {:.4f})'.format(
        dsc[best_i], epochs[best_i], iou[best_i], mae[best_i]))
    print('last DSC = {:.4f} @ epoch {}'.format(dsc[-1], epochs[-1]))


if __name__ == '__main__':
    main()
