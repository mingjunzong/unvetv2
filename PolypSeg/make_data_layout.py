"""把数据集整理成训练脚本需要的目录结构（新机器上跑一次即可）。

训练脚本要求的结构：
    <train>\\images\\*                      <train>\\masks\\*
    <test>\\CVC-ClinicDB\\images\\*          <test>\\CVC-ClinicDB\\masks\\*

本脚本支持两种来源划分：
    A) <src>\\train_images + <src>\\train_labels  (+ 可选 val_images / val_labels)
    B) <src>\\images + <src>\\masks
（src 直接就是这两种目录之一也行）

两种/三种模式：
    --mode copy      真拷贝（默认）。612 张图约 50MB，跨盘跨系统都能用，最稳。
    --mode symlink   Linux 下的软链接，零拷贝、秒完成。
    --mode junction  Windows 目录联接，零拷贝。要求源和目标在同一个盘符。

用法（Linux）：
    python make_data_layout.py \\
        --src        /root/unetv2/CVC-ClinicDB \\
        --out_train  /root/unetv2/unetv2_data/TrainDataset \\
        --out_test   /root/unetv2/unetv2_data/TestDataset

用法（Windows）：
    python make_data_layout.py ^
        --src        D:\\data\\CVC-ClinicDB ^
        --out_train  D:\\unetv2\\unetv2_data\\TrainDataset ^
        --out_test   D:\\unetv2\\unetv2_data\\TestDataset
"""
import argparse
import os
import shutil
import subprocess

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

TRAIN_IMG_NAMES = ['train_images', 'images', 'train_image', 'image', 'imgs']
TRAIN_MSK_NAMES = ['train_labels', 'train_masks', 'masks', 'labels', 'mask', 'gts', 'ground_truth']
VAL_IMG_NAMES = ['val_images', 'test_images', 'val_image', 'val_imgs']
VAL_MSK_NAMES = ['val_labels', 'val_masks', 'test_labels', 'test_masks', 'val_masks']


def find_dir(root, names):
    for n in names:
        d = os.path.join(root, n)
        if os.path.isdir(d):
            return d
    return None


def stems(d):
    return set(os.path.splitext(f)[0] for f in os.listdir(d)
               if os.path.splitext(f)[1].lower() in IMG_EXT)


def link_or_copy(src, dst, mode):
    if os.path.exists(dst) or os.path.islink(dst):
        print('    [skip] 已存在: {}'.format(dst))
        return
    parent = os.path.dirname(dst)
    if parent:
        os.makedirs(parent, exist_ok=True)

    ok = False
    if mode == 'junction':
        if os.name != 'nt':
            print('    [warn] junction 是 Windows 专有；Linux 请用 --mode symlink。已退回拷贝')
        else:
            r = subprocess.run(['cmd', '/c', 'mklink', '/J', dst, os.path.abspath(src)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if r.returncode == 0:
                print('    [junction] {}  ->  {}'.format(src, dst))
                ok = True
            else:
                print('    [warn] junction 失败（多半是跨盘）: {}'.format((r.stdout or '').strip()))
    elif mode == 'symlink':
        try:
            os.symlink(os.path.abspath(src), dst, target_is_directory=True)
            print('    [symlink] {}  ->  {}'.format(src, dst))
            ok = True
        except OSError as e:
            print('    [warn] symlink 失败 ({}), 退回拷贝'.format(e))

    if not ok:
        shutil.copytree(src, dst)
        print('    [copy] {}  ->  {}'.format(src, dst))


def report(title, img_dir, msk_dir):
    if not (img_dir and msk_dir):
        return
    print('    {}: images {} 个 / masks {} 个 / 文件名能配上 {} 对'.format(
        title, len(os.listdir(img_dir)), len(os.listdir(msk_dir)),
        len(stems(img_dir) & stems(msk_dir))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='你现有的数据集根目录')
    ap.add_argument('--out_train', required=True, help='输出训练集目录')
    ap.add_argument('--out_test', required=True, help='输出测试集根目录')
    ap.add_argument('--mode', choices=['copy', 'symlink', 'junction'], default='copy')
    opt = ap.parse_args()

    src = os.path.abspath(opt.src)
    if not os.path.isdir(src):
        raise SystemExit('[ERROR] 找不到目录: {}'.format(src))

    ti = find_dir(src, TRAIN_IMG_NAMES)
    tm = find_dir(src, TRAIN_MSK_NAMES)
    vi = find_dir(src, VAL_IMG_NAMES)
    vm = find_dir(src, VAL_MSK_NAMES)

    print('[src] {}'.format(src))
    print('  train images : {}'.format(ti))
    print('  train masks  : {}'.format(tm))
    print('  val images   : {}'.format(vi))
    print('  val masks    : {}'.format(vm))

    if not ti or not tm:
        raise SystemExit(
            '[ERROR] 在 {} 下没找到训练图片/掩码目录。\n'
            '        期望有 train_images/ + train_labels/，或者 images/ + masks/'.format(src))
    if not (stems(ti) & stems(tm)):
        print('[warn] 训练图片和掩码的文件名完全没有交集，请先确认数据本身')

    print('\n[out] 训练集')
    link_or_copy(ti, os.path.join(opt.out_train, 'images'), opt.mode)
    link_or_copy(tm, os.path.join(opt.out_train, 'masks'), opt.mode)

    if vi and vm:
        print('\n[out] 测试集 -> CVC-ClinicDB')
        link_or_copy(vi, os.path.join(opt.out_test, 'CVC-ClinicDB', 'images'), opt.mode)
        link_or_copy(vm, os.path.join(opt.out_test, 'CVC-ClinicDB', 'masks'), opt.mode)
    else:
        print('\n[skip] 没找到 val_images/val_labels，跳过测试集（只训练不评估也行）')

    print('\n[check]')
    report('train', os.path.join(opt.out_train, 'images'), os.path.join(opt.out_train, 'masks'))
    report('val  ', os.path.join(opt.out_test, 'CVC-ClinicDB', 'images'),
           os.path.join(opt.out_test, 'CVC-ClinicDB', 'masks'))

    print('\n[done] 训练命令里用这两个路径：')
    print('    --train_path "{}"'.format(opt.out_train))
    print('    --test_path  "{}"'.format(opt.out_test))


if __name__ == '__main__':
    main()
