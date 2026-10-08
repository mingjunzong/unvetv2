"""把原始 CVC-ClinicDB 整理成训练脚本需要的目录结构（并顺手转成 PNG）。

为什么必须转格式
---------------
CVC-ClinicDB 官方发布的 `Original/*.tif` 里，TIFF 标签是自相矛盾的：
    PhotometricInterpretation = 1   (MinIsBlack, 灰度)
    SamplesPerPixel           = 3   (RGB)
PIL/Pillow 会因此拒绝打开（UnidentifiedImageError: cannot identify image file），
而 DataLoader 里用的正是 PIL —— 所以不转格式，训练会在第一个 batch 就崩。
OpenCV 的解码器容错，能正常读出来，所以这里用 cv2.imdecode 读、再存成 PNG。

另外：cv2.imread 在 Windows 上读不了含中文的路径，所以统一走"读字节 -> imdecode"
和"imencode -> 写字节"，跨平台都不会踩这个坑。

支持两种来源布局：
    A) <src>/Original/*.tif              + <src>/Ground Truth/*.tif   (CVC 官网原始包)
    B) <src>/images/*                    + <src>/masks/*

输出：
    <out_train>/images/*.png                <out_train>/masks/*.png
    <out_test>/CVC-ClinicDB/images/*.png    <out_test>/CVC-ClinicDB/masks/*.png

划分：默认 550/62（PraNet / Polyp-PVT 系列的通用比例），固定 seed 可复现。
要和论文数字严格逐张对齐，请用作者发布的官方 split；本脚本给的是等价可用的划分。

用法（Linux）：
    python prepare_cvc_clinicdb.py \\
        --src       /root/unetv2/CVC-ClinicDB \\
        --out_train /root/unetv2/unetv2_data/TrainDataset \\
        --out_test  /root/unetv2/unetv2_data/TestDataset
"""
import argparse
import os
import random
import struct

import cv2
import numpy as np
from PIL import Image

IMG_EXT = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')

IMG_DIR_CANDIDATES = ['Original', 'original', 'images', 'image', 'Image', 'Images', 'frames']
MASK_DIR_CANDIDATES = ['Ground Truth', 'ground truth', 'GroundTruth', 'masks', 'mask',
                       'Mask', 'Masks', 'GT', 'gt']


def find_dir(root, candidates):
    for c in candidates:
        d = os.path.join(root, c)
        if os.path.isdir(d):
            return d
    return None


def index_by_stem(folder):
    out = {}
    for f in os.listdir(folder):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMG_EXT:
            out[stem] = os.path.join(folder, f)
    return out


_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}


def _tiff_tags(data):
    """极简 TIFF 目录解析，返回 (字节序, {tag: 值}) 或 None"""
    if len(data) < 8:
        return None
    if data[:2] == b'II':
        e = '<'
    elif data[:2] == b'MM':
        e = '>'
    else:
        return None
    if struct.unpack_from(e + 'H', data, 2)[0] != 42:
        return None
    ifd = struct.unpack_from(e + 'I', data, 4)[0]
    if ifd + 2 > len(data):
        return None
    n = struct.unpack_from(e + 'H', data, ifd)[0]

    tags = {}
    for i in range(n):
        off = ifd + 2 + i * 12
        if off + 12 > len(data):
            return None
        tag, typ, cnt = struct.unpack_from(e + 'HHI', data, off)
        size = _TYPE_SIZE.get(typ, 0)
        if size == 0 or cnt == 0:
            continue
        total = cnt * size
        if total <= 4:
            raw = data[off + 8:off + 8 + total]
        else:
            voff = struct.unpack_from(e + 'I', data, off + 8)[0]
            if voff + total > len(data):
                return None
            raw = data[voff:voff + total]
        if typ == 3:
            vals = list(struct.unpack(e + '%dH' % cnt, raw))
        elif typ == 4:
            vals = list(struct.unpack(e + '%dI' % cnt, raw))
        elif typ == 1:
            vals = list(raw)
        else:
            vals = raw
        tags[tag] = vals if cnt > 1 else (vals[0] if isinstance(vals, list) else vals)
    return e, tags


def load_tiff_raw(path):
    """自己按 strip 解未压缩的 8bit TIFF。

    为什么需要这个：CVC-ClinicDB 官方 Original/*.tif 的标签是矛盾的 ——
        PhotometricInterpretation = 1  (灰度)
        SamplesPerPixel           = 3  (三个通道)
    而且三个通道的数据**确实不同**（彩色图）。OpenCV / PIL 看到 photometric=1
    就只取一个通道，彩色图会被静默读成灰度图。
    这里直接按 StripOffsets/StripByteCounts 把原始字节拼起来，
    按 chunky(PlanarConfiguration=1) 还原成 (H, W, 3)。
    """
    with open(path, 'rb') as f:
        data = f.read()
    parsed = _tiff_tags(data)
    if parsed is None:
        return None
    e, tags = parsed

    w, h = tags.get(256), tags.get(257)
    spp = tags.get(277, 1)
    bps = tags.get(258, 8)
    comp = tags.get(259, 1)
    planar = tags.get(284, 1)
    strips, counts = tags.get(273), tags.get(279)
    if w is None or h is None or strips is None or counts is None:
        return None
    if comp != 1 or planar != 1:            # 只处理未压缩、chunky
        return None
    if isinstance(bps, list):
        if any(b != 8 for b in bps):
            return None
    elif bps != 8:
        return None
    if not isinstance(strips, list):
        strips = [strips]
    if not isinstance(counts, list):
        counts = [counts]
    if len(strips) != len(counts):
        return None

    need = w * h * spp
    buf = bytearray()
    for o, c in zip(strips, counts):
        buf += data[o:o + c]
        if len(buf) >= need:
            break
    if len(buf) < need:
        return None
    arr = np.frombuffer(bytes(buf[:need]), np.uint8)
    if spp == 3:
        return arr.reshape(h, w, 3)          # 原始顺序就是 R,G,B
    if spp == 1:
        return arr.reshape(h, w)
    return None


def load_any(path):
    """读图。TIFF 优先用自带的解析器（见 load_tiff_raw 的说明），
    其它格式或解析失败时退回 OpenCV，最后再退回 PIL。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        arr = load_tiff_raw(path)
        if arr is not None:
            return arr

    with open(path, 'rb') as f:
        buf = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        img = np.array(Image.open(path))
    if img is None:
        raise RuntimeError('无法解码: {}'.format(path))
    return img


def save_png(img, dst):
    """用 PIL 写 PNG：既不受 Windows 中文路径影响，也天然按 RGB 顺序处理"""
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    Image.fromarray(img).save(dst)


def convert_pairs(pairs, out_img_dir, out_msk_dir, tag):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)
    done = 0
    n_color = 0
    first_rgb = None
    mask_max = 0
    for src_img, src_msk in pairs:
        stem_i = os.path.splitext(os.path.basename(src_img))[0]
        stem_m = os.path.splitext(os.path.basename(src_msk))[0]
        img = load_any(src_img)
        msk = load_any(src_msk)
        if msk.ndim == 3:                 # 万一 mask 是 RGB/RGBA，取第一通道
            msk = msk[:, :, 0]
        if img.ndim == 3:
            n_color += 1
            if first_rgb is None:
                first_rgb = [float(img[:, :, c].mean()) for c in range(img.shape[2])]
        mask_max = max(mask_max, int(msk.max()))
        save_png(img, os.path.join(out_img_dir, stem_i + '.png'))
        save_png(msk, os.path.join(out_msk_dir, stem_m + '.png'))
        done += 1
        if done % 200 == 0:
            print('    ... {} 已转换 {} 张'.format(tag, done))
    print('    [{}] 转换完成 {} 张'.format(tag, done))
    print('         images -> {}'.format(out_img_dir))
    print('         masks  -> {}'.format(out_msk_dir))
    if done:
        print('         彩色图 {}/{} 张'.format(n_color, done), end='')
        if first_rgb:
            print('，首张 R/G/B 均值 = {:.1f}/{:.1f}/{:.1f}（不等即说明保住了颜色）'.format(*first_rgb))
        else:
            print('  <-- 全是单通道，检查数据')
        print('         mask 最大灰度 = {}'.format(mask_max))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='原始 CVC-ClinicDB 根目录')
    ap.add_argument('--out_train', required=True, help='输出训练集目录')
    ap.add_argument('--out_test', default=None, help='输出测试集根目录')
    ap.add_argument('--n_train', type=int, default=550,
                    help='训练集张数，PraNet 惯例 550（总共 612）')
    ap.add_argument('--seed', type=int, default=1234)
    opt = ap.parse_args()

    src = os.path.abspath(opt.src)
    if not os.path.isdir(src):
        raise SystemExit('[ERROR] 找不到目录: {}'.format(src))

    img_root = find_dir(src, IMG_DIR_CANDIDATES) or src
    mask_root = find_dir(src, MASK_DIR_CANDIDATES) or src
    print('[scan] images: {}'.format(img_root))
    print('[scan] masks : {}'.format(mask_root))

    images = index_by_stem(img_root)
    masks = index_by_stem(mask_root)
    stems = sorted(set(images) & set(masks))
    if not stems:
        raise SystemExit('[ERROR] 图片和 mask 文件名没有交集，请检查 --src 的目录结构')
    print('[scan] 找到 {} 对（images {} / masks {}）'.format(len(stems), len(images), len(masks)))

    rng = random.Random(opt.seed)
    rng.shuffle(stems)
    if opt.n_train > len(stems):
        raise SystemExit('[ERROR] --n_train={} 超过总对数 {}'.format(opt.n_train, len(stems)))
    train_stems = sorted(stems[:opt.n_train])
    test_stems = sorted(stems[opt.n_train:])

    train_pairs = [(images[s], masks[s]) for s in train_stems]
    test_pairs = [(images[s], masks[s]) for s in test_stems]

    print('\n[out] 训练集 {} 张'.format(len(train_pairs)))
    convert_pairs(train_pairs,
                  os.path.join(opt.out_train, 'images'),
                  os.path.join(opt.out_train, 'masks'), 'train')

    if opt.out_test:
        print('\n[out] 测试集 {} 张'.format(len(test_pairs)))
        base = os.path.join(opt.out_test, 'CVC-ClinicDB')
        convert_pairs(test_pairs,
                      os.path.join(base, 'images'),
                      os.path.join(base, 'masks'), 'test')

    print('\n[done] 总对数 {}（seed={}，{}/{} 划分）'.format(
        len(stems), opt.seed, opt.n_train, len(stems) - opt.n_train))
    print('       训练命令用这两个路径：')
    print('           --train_path "{}"'.format(opt.out_train))
    if opt.out_test:
        print('           --test_path  "{}"'.format(opt.out_test))


if __name__ == '__main__':
    main()
