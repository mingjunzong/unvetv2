"""
U-Net v2 —— PolypSeg 测试/推理脚本

原版 Test.py 已经没法用了：
  * 顶部 `from scipy import misc` 在 SciPy >= 1.12 被删除，直接 ImportError
  * 里面用的模型是 PolypPVT，权重默认路径 ./model_pth/PolypPVT.pth（不存在）
  * 到处都是 .cuda()
这里按 UNetV2 重写，并复用 Train.py 里的指标计算。

用法见同目录 RUN.md
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate

from unet_v2.UNet_v2 import UNetV2
from utils.dataloader import test_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_TEST_SETS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']


def calcuate_score(gt, pred):
    mae = np.mean(np.abs(gt - pred))
    intersection = np.sum((pred == 1) & (gt == 1))
    num_gt = np.sum(gt)
    num_pred = np.sum(pred)
    smooth = 1e-5
    iou = (intersection + smooth) / (num_gt + num_pred - intersection + smooth)
    dice = (2 * intersection + smooth) / (num_gt + num_pred + smooth)
    return dice, iou, mae


def discover_test_datasets(test_path):
    """同 Train.py，支持 test_path/<name>/{images,masks} 或 test_path/{images,masks}"""
    if not test_path:
        return None, []
    test_path = os.path.abspath(test_path)
    if not os.path.isdir(test_path):
        return test_path, []

    def has_pair(d):
        return os.path.isdir(os.path.join(d, 'images')) and os.path.isdir(os.path.join(d, 'masks'))

    if has_pair(test_path):
        return os.path.dirname(test_path), [os.path.basename(test_path)]

    names = [n for n in CANDIDATE_TEST_SETS if has_pair(os.path.join(test_path, n))]
    for n in sorted(os.listdir(test_path)):
        if n not in names and has_pair(os.path.join(test_path, n)):
            names.append(n)
    return test_path, names


def resolve_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


@torch.no_grad()
def run_dataset(model, test_root, dataset, opt, device):
    image_root = os.path.join(test_root, dataset, 'images')
    gt_root = os.path.join(test_root, dataset, 'masks')
    loader = test_dataset(image_root, gt_root, opt.testsize)

    save_dir = os.path.join(opt.save_path, dataset)
    if opt.save_masks:
        os.makedirs(save_dir, exist_ok=True)

    dscs, ious, maes = [], [], []
    for i in range(loader.size):
        image, gt, name = loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.to(device, non_blocking=True)

        y = model(image)
        res = F.interpolate(y, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().float().cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        if opt.save_masks:
            cv2.imwrite(os.path.join(save_dir, name), (res * 255).astype(np.uint8))

        pred = np.reshape(res, (-1)) >= 0.5
        target = np.reshape(np.array(gt), (-1))
        dice, iou, mae = calcuate_score(target, pred)
        dscs.append(dice)
        ious.append(iou)
        maes.append(mae)

    return float(np.mean(dscs)), float(np.mean(ious)), float(np.mean(maes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True,
                        help='测试集根目录，内含 CVC-ClinicDB/ 等子目录（各自有 images/ masks/）')
    parser.add_argument('--pth_path', type=str, default=None,
                        help='模型权重，默认 <train_save>/UNetV2/best/best.pth')
    parser.add_argument('--save_path', type=str, default=os.path.join(HERE, 'result_map', 'UNetV2'),
                        help='预测图保存目录')
    parser.add_argument('--testsize', type=int, default=352)
    parser.add_argument('--n_classes', type=int, default=1)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save_masks', action='store_true', help='把预测 mask 存成 png')
    opt = parser.parse_args()

    device = resolve_device(opt.device)
    print('[device] {}'.format(device))

    if opt.pth_path is None:
        raise SystemExit('[ERROR] 请用 --pth_path 指定权重，例如 '
                         '--pth_path ./checkpoints/UNetV2/best/best.pth')
    if not os.path.isfile(opt.pth_path):
        raise SystemExit('[ERROR] 找不到权重文件: {}'.format(opt.pth_path))

    model = UNetV2(n_classes=opt.n_classes, deep_supervision=False, pretrained_path=None)
    state = torch.load(opt.pth_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print('[warn] 缺少的权重 key: {}'.format(missing[:5]))
    if unexpected:
        print('[warn] 多余的权重 key: {}'.format(unexpected[:5]))
    model.to(device)
    model.eval()
    print('[ckpt] loaded {}'.format(opt.pth_path))

    if not os.path.isdir(opt.test_path):
        raise SystemExit('[ERROR] --test_path 不存在: {}'.format(opt.test_path))

    test_root, datasets = discover_test_datasets(opt.test_path)
    if not datasets:
        raise SystemExit('[ERROR] {} 下没有找到 {images,masks} 成对的测试数据集'.format(test_root))
    print('[test] {}'.format(', '.join(datasets)))

    results, dices, ious, maes = [], [], [], []
    for dataset in datasets:
        dsc, iou, mae = run_dataset(model, test_root, dataset, opt, device)
        dices.append(dsc)
        ious.append(iou)
        maes.append(mae)
        results.append([dataset, dsc, iou, mae])
        print('  [{:<20s}] dsc={:.4f} iou={:.4f} mae={:.4f}'.format(dataset, dsc, iou, mae))

    results.append(['mean', float(np.mean(dices)), float(np.mean(ious)), float(np.mean(maes))])
    tab = tabulate(results, headers=['dataset', 'dsc', 'iou', 'mae'], floatfmt='.3f')
    print(tab)

    os.makedirs(opt.save_path, exist_ok=True)
    with open(os.path.join(opt.save_path, 'test_results.txt'), 'w', encoding='utf-8') as f:
        f.write('ckpt: {}\n\n{}\n'.format(opt.pth_path, tab))
    print('[done] 结果已写入 {}'.format(os.path.join(opt.save_path, 'test_results.txt')))


if __name__ == '__main__':
    main()
