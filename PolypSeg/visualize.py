"""U-Net v2 推理 + 可视化：每张测试图输出一张 5 联图。

    原图 | Ground Truth | 概率图(带色标) | 二值 | 红色叠加
    标题:  文件名 | Dice / IoU / MAE

用法：
    # 整个测试集（默认先出 20 张）
    python visualize.py \
        --test_path /data/pc/unetv2_data/TestDataset \
        --pth_path  /data/pc/unetv2_runs/UNetV2/best/best.pth \
        --save_path /data/pc/unetv2_runs/vis

    # 指定张数 / 指定单个数据集 / 单张图
    python visualize.py --test_path ... --pth_path ... --num 50
    python visualize.py --test_path ... --pth_path ... --dataset CVC-ClinicDB
    python visualize.py --image /path/1.png --mask /path/1.png --pth_path ... --save_path ./vis

注：图里的文字全部用英文 —— 服务器上 matplotlib 默认字体没有中文，用中文会变成方块。
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use('Agg')                       # 服务器无桌面环境
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from unet_v2.UNet_v2 import UNetV2
from utils.dataloader import test_dataset

CANDIDATE_TEST_SETS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']


# --------------------------------------------------------------------------- #
def calcuate_score(gt, pred):
    mae = float(np.mean(np.abs(gt - pred)))
    intersection = float(np.sum((pred == 1) & (gt == 1)))
    num_gt = float(np.sum(gt))
    num_pred = float(np.sum(pred))
    smooth = 1e-5
    iou = (intersection + smooth) / (num_gt + num_pred - intersection + smooth)
    dice = (2 * intersection + smooth) / (num_gt + num_pred + smooth)
    return dice, iou, mae


def resolve_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def load_model(ckpt, device, n_classes=1):
    if not os.path.isfile(ckpt):
        raise SystemExit('[ERROR] 找不到权重: {}'.format(ckpt))
    model = UNetV2(n_classes=n_classes, deep_supervision=False, pretrained_path=None)
    state = torch.load(ckpt, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print('[warn] 多余 key: {}'.format(unexpected[:3]))
    if missing:
        print('[warn] 缺少 key: {}'.format(missing[:3]))
    model.to(device)
    model.eval()
    print('[ckpt] {}'.format(ckpt))
    return model


@torch.no_grad()
def predict(model, original, device, testsize):
    """original: PIL RGB。返回 prob (352x352)。"""
    img = original.resize((testsize, testsize), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    x = x.to(device)
    y = model(x)
    if isinstance(y, (list, tuple)):
        y = y[0]
    prob = torch.sigmoid(y).float().cpu().numpy().squeeze()
    return prob


def make_figure(orig, gt, prob, pred, name, dice, iou, mae, out_path, alpha=0.55):
    fig, axes = plt.subplots(1, 5, figsize=(23, 4.8))
    fig.suptitle('{}   |   Dice: {:.4f}    IoU: {:.4f}    MAE: {:.4f}'.format(
        name, dice, iou, mae), fontsize=16, fontweight='bold')

    axes[0].imshow(orig)
    axes[0].set_title('Original Image')

    axes[1].imshow(gt, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth')

    im = axes[2].imshow(prob, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title('Prediction (Probability)')
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=9)

    axes[3].imshow(pred.astype(np.uint8), cmap='gray', vmin=0, vmax=1)
    axes[3].set_title('Prediction (Binary)')

    overlay = orig.copy()
    m = pred.astype(bool)
    overlay[m] = (orig[m] * (1 - alpha) + np.array([255, 0, 0]) * alpha).astype(np.uint8)
    axes[4].imshow(overlay)
    axes[4].set_title('Overlay (Red = Pred)')

    for ax in axes:
        ax.axis('off')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_one(model, img_path, msk_path, device, opt, out_dir):
    original = Image.open(img_path).convert('RGB')
    prob_small = predict(model, original, device, opt.testsize)          # (352,352)

    W, H = original.size
    prob = cv2.resize(prob_small, (W, H), interpolation=cv2.INTER_LINEAR)

    if msk_path:
        gt = np.asarray(Image.open(msk_path).convert('L'), np.float32)
        gt = (gt > 127).astype(np.float32)
    else:
        gt = np.zeros((H, W), np.float32)

    # 和 Test.py 同一个口径：min-max 归一化后再 0.5 阈值
    res = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
    pred = res >= opt.threshold

    dice, iou, mae = calcuate_score(gt.reshape(-1), pred.reshape(-1))

    name = os.path.basename(img_path)
    stem = os.path.splitext(name)[0]
    out_path = os.path.join(out_dir, stem + '.png')
    orig = np.asarray(original, np.uint8)
    make_figure(orig, gt, prob, pred, name, dice, iou, mae, out_path, opt.alpha)
    return dice, iou, mae, out_path


def find_pair_in(ds_dir, stem):
    """在某个数据集目录里按文件名找 <dir>/images/<stem>.* 和 <dir>/masks/<stem>.*"""
    img_dir = os.path.join(ds_dir, 'images')
    msk_dir = os.path.join(ds_dir, 'masks')
    if not (os.path.isdir(img_dir) and os.path.isdir(msk_dir)):
        return None, None
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    img = msk = None
    for f in os.listdir(img_dir):
        s, e = os.path.splitext(f)
        if s == stem and e.lower() in exts:
            img = os.path.join(img_dir, f)
            break
    for f in os.listdir(msk_dir):
        s, e = os.path.splitext(f)
        if s == stem and e.lower() in exts:
            msk = os.path.join(msk_dir, f)
            break
    return img, msk


def discover_datasets(test_path):
    test_path = os.path.abspath(test_path)
    def has_pair(d):
        return os.path.isdir(os.path.join(d, 'images')) and os.path.isdir(os.path.join(d, 'masks'))
    if has_pair(test_path):
        return os.path.dirname(test_path), [os.path.basename(test_path)]
    names = [n for n in CANDIDATE_TEST_SETS if has_pair(os.path.join(test_path, n))]
    for n in sorted(os.listdir(test_path)):
        if n not in names and has_pair(os.path.join(test_path, n)):
            names.append(n)
    return test_path, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pth_path', required=True, help='模型权重 best.pth')
    ap.add_argument('--save_path', default='./vis', help='可视化输出目录')
    ap.add_argument('--test_path', default=None, help='测试集根目录')
    ap.add_argument('--dataset', default=None, help='只跑某一个数据集（如 CVC-ClinicDB）')
    ap.add_argument('--image', default=None, help='单张图片路径（和 --mask 一起用）')
    ap.add_argument('--mask', default=None, help='单张图片对应的 GT')
    ap.add_argument('--names', nargs='+', default=None,
                    help='只推理这几个文件（不带扩展名），例如 --names 117 10 130')
    ap.add_argument('--search', nargs='+', default=None,
                    help='--names 模式下额外的搜索目录（会自动加上同级的 TrainDataset）')
    ap.add_argument('--num', type=int, default=20, help='每个数据集最多可视化几张')
    ap.add_argument('--start', type=int, default=0, help='从第几张开始')
    ap.add_argument('--flat', action='store_true',
                    help='不建数据集子目录，图片直接铺在 --save_path 下面')
    ap.add_argument('--testsize', type=int, default=352)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--alpha', type=float, default=0.55, help='叠加层里红色的占比 0~1')
    ap.add_argument('--n_classes', type=int, default=1)
    ap.add_argument('--device', default='auto')
    opt = ap.parse_args()

    device = resolve_device(opt.device)
    print('[device] {}'.format(device))
    model = load_model(opt.pth_path, device, opt.n_classes)
    os.makedirs(opt.save_path, exist_ok=True)

    # ---- 单张模式 ----
    if opt.image:
        if not os.path.isfile(opt.image):
            raise SystemExit('[ERROR] 找不到图片: {}'.format(opt.image))
        dice, iou, mae, out_path = run_one(model, opt.image, opt.mask, device, opt, opt.save_path)
        print('  {}  Dice={:.4f} IoU={:.4f} MAE={:.4f}'.format(
            os.path.basename(opt.image), dice, iou, mae))
        print('[done] {}'.format(out_path))
        return

    # ---- 指定文件名模式：--names 117 10 130 ... ----
    if opt.names:
        sources = []
        if opt.test_path:
            tr, dss = discover_datasets(opt.test_path)
            for ds in dss:
                sources.append((os.path.join(tr, ds), ds))
            # 自动加上同级的 TrainDataset（那几张图可能在训练集里）
            guess = os.path.join(os.path.dirname(tr.rstrip('/\\')), 'TrainDataset')
            if os.path.isdir(guess):
                sources.append((guess, 'TrainDataset'))
        for d in (opt.search or []):
            if os.path.isdir(d):
                sources.append((d, os.path.basename(os.path.normpath(d))))
        if not sources:
            raise SystemExit('[ERROR] --names 模式需要 --test_path 或 --search 指定搜索目录')
        # 去重
        seen, uniq = set(), []
        for d, tag in sources:
            if d not in seen:
                seen.add(d)
                uniq.append((d, tag))
        sources = uniq
        print('[search] {}'.format('  '.join(d for d, _ in sources)))
        os.makedirs(opt.save_path, exist_ok=True)

        dscs, ious, maes = [], [], []
        for stem in opt.names:
            img_path = msk_path = src = None
            for d, tag in sources:
                i, m = find_pair_in(d, stem)
                if i:
                    img_path, msk_path, src = i, m, tag
                    break
            if not img_path:
                print('  [MISS] {:<12s} 没找到'.format(stem))
                continue
            dice, iou, mae, out_path = run_one(model, img_path, msk_path, device, opt,
                                               opt.save_path)
            dscs.append(dice); ious.append(iou); maes.append(mae)
            print('  [{:<16s}] {:<14s} Dice={:.4f} IoU={:.4f} MAE={:.4f}'.format(
                src or '', os.path.basename(img_path), dice, iou, mae))
        if dscs:
            print('  mean over {} imgs: Dice={:.4f} IoU={:.4f} MAE={:.4f}'.format(
                len(dscs), float(np.mean(dscs)), float(np.mean(ious)), float(np.mean(maes))))
        print('[done] 图片保存在 {}'.format(os.path.abspath(opt.save_path)))
        return

    # ---- 数据集模式 ----
    if not opt.test_path:
        raise SystemExit('[ERROR] 数据集模式需要 --test_path（或用 --image/--mask 跑单张）')
    test_root, datasets = discover_datasets(opt.test_path)
    if opt.dataset:
        datasets = [d for d in datasets if d == opt.dataset]
        if not datasets:
            raise SystemExit('[ERROR] {} 里没有名为 {} 的数据集'.format(test_root, opt.dataset))
    if not datasets:
        raise SystemExit('[ERROR] {} 下没找到 {{images,masks}} 成对的目录'.format(test_root))
    print('[test] {}'.format(', '.join(datasets)))

    for ds in datasets:
        image_root = os.path.join(test_root, ds, 'images')
        gt_root = os.path.join(test_root, ds, 'masks')
        loader = test_dataset(image_root, gt_root, opt.testsize)
        out_dir = opt.save_path if opt.flat else os.path.join(opt.save_path, ds)
        os.makedirs(out_dir, exist_ok=True)

        n = min(loader.size, opt.start + max(opt.num, 0))
        dscs, ious, maes = [], [], []
        for i in range(opt.start, n):
            img_path, msk_path = loader.pairs[i]
            dice, iou, mae, out_path = run_one(model, img_path, msk_path, device, opt, out_dir)
            dscs.append(dice); ious.append(iou); maes.append(mae)
            print('  [{:<16s}] {:<14s} Dice={:.4f} IoU={:.4f} MAE={:.4f}'.format(
                ds, os.path.basename(img_path), dice, iou, mae))

        if dscs:
            print('  [{:<16s}] mean over {} imgs: Dice={:.4f} IoU={:.4f} MAE={:.4f}'.format(
                ds, len(dscs), float(np.mean(dscs)), float(np.mean(ious)), float(np.mean(maes))))
        print('  -> {}'.format(out_dir))

    print('[done] 图片保存在 {}'.format(os.path.abspath(opt.save_path)))


if __name__ == '__main__':
    main()
