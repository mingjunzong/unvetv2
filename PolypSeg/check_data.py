"""数据体检：定位 "dsc=0.00 / iou=0.00 / mae 恒为前景占比" 这类问题。

这种三个数字同时出现，意味着模型对每张图都输出"全背景"。
本脚本用来区分到底是下面哪种：

  A. mask 文件本身是空的 / 值域不对（0-1 而不是 0-255）
  B. 图片和 mask 配错对了
  C. 图片或 mask 路径指错了（比如 junction 指到了别的目录）
  D. 数据没问题，是训练侧的问题（loss 没降、输出饱和）

用法（在 PolypSeg 目录下，用 learn 环境的 python）：
    python check_data.py
    python check_data.py --train_path ...\\TrainDataset --test_path ...\\TestDataset
"""
import argparse
import os

import numpy as np
from PIL import Image

TRAIN_DEFAULT = r'C:\Users\ming\Desktop\learn_deep\unetv2_data\TrainDataset'
TEST_DEFAULT = r'C:\Users\ming\Desktop\learn_deep\unetv2_data\TestDataset'


def list_files(folder):
    if not os.path.isdir(folder):
        return None
    return sorted(f for f in os.listdir(folder)
                  if os.path.splitext(f)[1].lower() in
                  ('.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'))


def sample_folder(folder, title, n=5):
    print('  [{}]'.format(title))
    print('    {}'.format(folder))
    fs = list_files(folder)
    if fs is None:
        print('    !! 目录不存在')
        return []
    print('    {} 个文件'.format(len(fs)))
    for f in fs[:n]:
        p = os.path.join(folder, f)
        im = Image.open(p)
        a = np.array(im.convert('L'))
        u = np.unique(a)
        print('      {:<14s} mode={:<6s} size={:<12s} min={:<4d} max={:<4d} mean={:<7.2f} 取值={}{}'.format(
            f, im.mode, str(im.size), int(a.min()), int(a.max()), float(a.mean()),
            u[:4].tolist(), '...' if len(u) > 4 else ''))
    return fs


def mask_fg(folder, f):
    a = np.array(Image.open(os.path.join(folder, f)).convert('L'))
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_path', default=TRAIN_DEFAULT)
    ap.add_argument('--test_path', default=TEST_DEFAULT)
    ap.add_argument('--trainsize', type=int, default=352)
    ap.add_argument('--batchsize', type=int, default=4)
    ap.add_argument('--ckpt', default=None,
                    help='训练好的 best.pth；给了就顺带看模型原始 logits 统计，判断是不是输出饱和')
    opt = ap.parse_args()

    tr_img = os.path.join(opt.train_path, 'images')
    tr_msk = os.path.join(opt.train_path, 'masks')
    va_img = os.path.join(opt.test_path, 'CVC-ClinicDB', 'images')
    va_msk = os.path.join(opt.test_path, 'CVC-ClinicDB', 'masks')

    print('=' * 74)
    print('第 1 步  磁盘上的文件本身')
    print('=' * 74)
    fi = sample_folder(tr_img, 'train images')
    fm = sample_folder(tr_msk, 'train masks')
    sample_folder(va_img, 'val images')
    sample_folder(va_msk, 'val masks')

    print()
    print('=' * 74)
    print('第 2 步  整目录的 mask 统计（是不是普遍是空的）')
    print('=' * 74)
    for title, folder in [('train masks', tr_msk), ('val masks', va_msk)]:
        fs = list_files(folder)
        if not fs:
            print('  [{}] 目录不存在或为空'.format(title))
            continue
        fracs = []
        maxes = []
        for f in fs:
            a = mask_fg(folder, f)
            fracs.append(float((a > 127).mean()))
            maxes.append(int(a.max()))
        fracs = np.array(fracs)
        empty = int((fracs == 0).sum())
        print('  [{}] {} 张：前景占比 mean={:.4f} median={:.4f} min={:.4f} max={:.4f}'.format(
            title, len(fs), fracs.mean(), np.median(fracs), fracs.min(), fracs.max()))
        print('        灰度最大值 min={} max={}  完全空白(前景=0)的图: {} 张'.format(
            min(maxes), max(maxes), empty))
        if max(maxes) == 1:
            print('        !! mask 的最大灰度是 1，说明它被存成了 0/1 而不是 0/255')
            print('           ToTensor() 会把它变成 0/0.0039 —— 目标几乎全 0，模型必然学成输出全黑')

    print()
    print('=' * 74)
    print('第 3 步  图文是否配对（息肉区域的颜色应该和背景明显不同）')
    print('=' * 74)
    if fi and fm:
        common = sorted(set(os.path.splitext(x)[0] for x in fi) &
                        set(os.path.splitext(x)[0] for x in fm))
        print('  文件名能配上的对数: {}（images {} / masks {}）'.format(len(common), len(fi), len(fm)))
        import random
        random.Random(0).shuffle(common)
        diffs = []
        for stem in common[:20]:
            ip = [x for x in fi if os.path.splitext(x)[0] == stem][0]
            mp = [x for x in fm if os.path.splitext(x)[0] == stem][0]
            im = np.array(Image.open(os.path.join(tr_img, ip)).convert('RGB').resize((256, 256)),
                          np.float32)
            mk = np.array(Image.open(os.path.join(tr_msk, mp)).convert('L').resize((256, 256)),
                          np.float32) > 127
            if mk.sum() == 0:
                continue
            d = float(np.abs(im[mk].mean(0) - im[~mk].mean(0)).mean())
            diffs.append(d)
        if diffs:
            print('  抽样 {} 对，mask 内 vs mask 外 的平均颜色差 = {:.2f}（0~255 尺度）'.format(
                len(diffs), float(np.mean(diffs))))
            print('  参考：配对正确时通常 > 10；接近 0~3 说明图文基本无关，配错对了')

    print()
    print('=' * 74)
    print('第 4 步  走真实 DataLoader，看训练时喂进去的 gt')
    print('=' * 74)
    try:
        from utils.dataloader import get_loader
        loader = get_loader(tr_img, tr_msk, batchsize=opt.batchsize,
                            trainsize=opt.trainsize, num_workers=0, shuffle=False)
        images, gts = next(iter(loader))
        print('  images : shape={} dtype={} min={:.3f} max={:.3f} mean={:.3f}'.format(
            tuple(images.shape), images.dtype,
            float(images.min()), float(images.max()), float(images.mean())))
        print('  gts    : shape={} dtype={} min={:.3f} max={:.3f} mean={:.5f}'.format(
            tuple(gts.shape), gts.dtype,
            float(gts.min()), float(gts.max()), float(gts.mean())))
        print('  每张图前景占比: {}'.format(
            [round(float((g > 0.5).float().mean()), 4) for g in gts]))

        print()
        print('-' * 74)
        if float(gts.max()) == 0:
            print('结论：gt 全为 0 —— 模型学到的目标就是"全背景"，')
            print('      所以验证 dsc=0、mae=前景占比。问题在数据（第 1~2 步的输出里找）。')
        elif float(gts.max()) < 0.01:
            print('结论：gt 的取值非常小（max={:.5f}）—— mask 被存成了 0/1，')
            print('      ToTensor() 后目标几乎全是 0，模型必然输出全黑。')
            print('      修法：把 mask 重新存成 0/255，或在 dataloader 里乘 255。'.format(float(gts.max())))
        elif float(gts.mean()) < 0.005:
            print('结论：gt 前景占比只有 {:.5f}，几乎全背景。检查 mask。'.format(float(gts.mean())))
        else:
            print('结论：数据没问题（前景占比 {:.4f}，max={:.2f}）。'.format(
                float(gts.mean()), float(gts.max())))
            print('      => 问题在训练侧：去看训练时打印的 loss 有没有从 ~3.1 往下降。')
            print('         如果 loss 一直不动，重点查 lr / 梯度裁剪 / 优化器。')
    except Exception as e:
        print('  DataLoader 检查失败: {}: {}'.format(type(e).__name__, e))
        print('  （可能是 --train_path 指向的目录结构不对）')

    if opt.ckpt:
        print()
        print('=' * 74)
        print('第 5 步  加载权重，看模型对验证图的原始输出')
        print('=' * 74)
        if not os.path.isfile(opt.ckpt):
            print('  找不到权重文件: {}'.format(opt.ckpt))
        else:
            try:
                import torch
                from unet_v2.UNet_v2 import UNetV2
                from utils.dataloader import test_dataset

                dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = UNetV2(n_classes=1, deep_supervision=True, pretrained_path=None).to(dev)
                model.load_state_dict(torch.load(opt.ckpt, map_location='cpu'), strict=False)
                model.eval()
                td = test_dataset(va_img, va_msk, opt.trainsize)
                print('  权重: {}  (device={})'.format(opt.ckpt, dev))
                with torch.no_grad():
                    for k in range(min(3, td.size)):
                        image, gt, name = td.load_data()
                        y = model(image.to(dev))
                        if isinstance(y, (list, tuple)):
                            y = y[0]
                        logit = y.float().cpu().numpy().squeeze()
                        prob = 1.0 / (1.0 + np.exp(-logit))
                        print('    {:<12s} logit[min={:7.2f} max={:7.2f} mean={:7.2f}]  '
                              'prob[mean={:.4f}]  prob>0.5 占比={:.4f}'.format(
                                  name, logit.min(), logit.max(), logit.mean(),
                                  prob.mean(), float((prob > 0.5).mean())))
                print()
                print('  怎么看：')
                print('    logit max 是很负的数（比如 < -10）、prob 几乎全 0')
                print('        -> 模型被训练成"输出全背景"了，训练侧塌了（或目标全是 0）')
                print('    prob mean 接近 0.5、logit 都在 0 附近')
                print('        -> 模型基本没被训练过 / 权重没加载对')
                print('    prob 在息肉位置明显偏高')
                print('        -> 模型其实是好的，问题出在 test() 的 min-max 归一化 + 0.5 阈值上')
            except Exception as e:
                print('  加载权重失败: {}: {}'.format(type(e).__name__, e))


if __name__ == '__main__':
    main()
