"""从已有数据集里挑出指定的若干张，组成一个小数据集。

典型用途：
  1. 过拟合自检 —— 只喂几张图训到 loss≈0 / dice≈1，验证代码实现没问题
  2. 快速实验 —— 改一行代码，几秒钟就能看出有没有报错
  3. 微调 —— 挑一些难例，从已训好的权重接着练

用法：
    python make_subset.py --names 117 10 130 12 110 129 107 126 \
        --dst /data/pc/unetv2_data/subset8

    # 默认会在 /data/pc/unetv2_data/TrainDataset 和 TestDataset 两个地方找
    # 也可以自己指定搜索目录：
    python make_subset.py --names 117 10 --search /path/A /path/B --dst /path/out

输出结构（和标准数据集一致，可直接当 --train_path / --test_path 用）：
    <dst>/images/*.png
    <dst>/masks/*.png
默认用软链接，零拷贝；加 --copy 则真拷贝。
"""
import argparse
import os
import shutil

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def find_pair(search_dirs, stem):
    """在若干数据集目录里找 <dir>/images/<stem>.* 和 <dir>/masks/<stem>.*"""
    for d in search_dirs:
        img_dir = os.path.join(d, 'images')
        msk_dir = os.path.join(d, 'masks')
        if not (os.path.isdir(img_dir) and os.path.isdir(msk_dir)):
            continue
        img = msk = None
        for f in os.listdir(img_dir):
            if os.path.splitext(f)[0] == stem and os.path.splitext(f)[1].lower() in IMG_EXT:
                img = os.path.join(img_dir, f)
                break
        for f in os.listdir(msk_dir):
            if os.path.splitext(f)[0] == stem and os.path.splitext(f)[1].lower() in IMG_EXT:
                msk = os.path.join(msk_dir, f)
                break
        if img and msk:
            return img, msk, d
    return None, None, None


def link_or_copy(src, dst, copy=False):
    if os.path.exists(dst):
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', nargs='+', required=True,
                    help='要挑出来的文件名（不带扩展名），例如 117 10 130')
    ap.add_argument('--dst', required=True, help='输出目录')
    ap.add_argument('--search', nargs='+', default=None,
                    help='在哪些数据集目录里找，默认两个常用位置')
    ap.add_argument('--copy', action='store_true', help='真拷贝（默认软链接）')
    opt = ap.parse_args()

    search = opt.search or [
        '/data/pc/unetv2_data/TrainDataset',
        '/data/pc/unetv2_data/TestDataset/CVC-ClinicDB',
        '/data/pc/unetv2_data/TestDataset',
    ]
    search = [d for d in search if os.path.isdir(d)]
    if not search:
        raise SystemExit('[ERROR] 没有可用的搜索目录，请用 --search 指定')
    print('[search] {}'.format('  '.join(search)))

    out_img = os.path.join(opt.dst, 'images')
    out_msk = os.path.join(opt.dst, 'masks')
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_msk, exist_ok=True)

    found, missing = [], []
    for stem in opt.names:
        img, msk, where = find_pair(search, stem)
        if not img:
            missing.append(stem)
            print('  [MISS] {:<10s} 在搜索目录里没找到'.format(stem))
            continue
        link_or_copy(img, os.path.join(out_img, os.path.basename(img)), opt.copy)
        link_or_copy(msk, os.path.join(out_msk, os.path.basename(msk)), opt.copy)
        found.append(stem)
        print('  [OK]   {:<10s} <- {}'.format(stem, where))

    print()
    print('[done] 找到 {} 张 / 共要 {} 张，输出到 {}'.format(len(found), len(opt.names), opt.dst))
    if missing:
        print('[warn] 没找到: {}'.format(', '.join(missing)))
    print()
    print('训练命令（过拟合自检，测试集就用它自己）：')
    print('  python /data/pc/PolypSeg/Train.py \\')
    print('    --train_path {0} --test_path {0} \\'.format(opt.dst))
    print('    --train_save <输出目录> --epoch 200 --batchsize 4 --val_every 20')


if __name__ == '__main__':
    main()
