"""
U-Net v2 —— PolypSeg 训练脚本（已去掉原仓库里作者自己集群的硬编码）

相对原版的改动：
  1. 删除硬编码的 /afs/crc.nd.edu/... 测试集路径，改用 --test_path
  2. 所有 .cuda() 改成按 --device 自动选择（默认有 GPU 用 GPU，没有也能 CPU 跑）
  3. PVT 预训练权重改为相对本脚本定位，不再依赖 os.getcwd()（原版从别的目录启动会静默随机初始化）
  4. wandb 改为可选（默认关闭），不再使用作者硬编码的 API key；评估表格本地 print + 落盘
  5. 测试集自动发现：--test_path 下有哪些子数据集就测哪些，缺的不再让训练崩掉
  6. reduce='none' -> reduction='none'（新版 PyTorch 兼容）
  7. 每 epoch 不再无条件存一份 checkpoint（原版 100 epoch 会占 ~11GB）

用法见同目录 RUN.md
"""
import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tabulate import tabulate

from unet_v2.UNet_v2 import UNetV2
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, AvgMeter

try:  # wandb 可选
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRETRAINED = os.path.join(HERE, 'pvt_pth', 'pvt_v2_b2.pth')
CANDIDATE_TEST_SETS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']

best = 0.0


class _Tee(object):
    """把输出同时写到多个流（终端 + 日志文件）"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def fmt_time(sec):
    """把秒数格式化成 1:23:45 / 12:34"""
    sec = int(max(sec, 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return '{:d}:{:02d}:{:02d}'.format(h, m, s) if h else '{:02d}:{:02d}'.format(m, s)


def set_lr(optimizer, base_lr, epoch, opt):
    """按 --scheduler 设置本轮学习率。

    注意：这里是"直接赋值"，不是像原版 adjust_lr 那样在当前 lr 上乘系数
    （原版那样写，一旦 epoch 超过 decay_epoch 就会每个 epoch 再乘一次，lr 会指数崩掉）。
    """
    e = epoch - 1                      # 第 1 个 epoch 用满初始 lr
    if opt.scheduler == 'poly':
        # 论文：lr = base_lr * (1 - e / max_epoch) ** 0.9
        factor = max(0.0, 1.0 - float(e) / float(max(opt.epoch, 1))) ** opt.poly_power
    elif opt.scheduler == 'step':
        factor = opt.decay_rate ** (e // max(opt.decay_epoch, 1))
    else:
        factor = 1.0
    lr = base_lr * factor
    for group in optimizer.param_groups:
        group['lr'] = lr
    return lr


# --------------------------------------------------------------------------- #
# 损失 / 指标
# --------------------------------------------------------------------------- #
def structure_loss(pred, mask):
    if mask.shape[-1] != pred.shape[-1]:
        pred = F.interpolate(pred, size=mask.shape[-2:], mode='bilinear', align_corners=False)

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def calcuate_score(gt, pred):
    mae = np.mean(np.abs(gt - pred))

    intersection = np.sum((pred == 1) & (gt == 1))
    num_gt = np.sum(gt)
    num_pred = np.sum(pred)

    smooth = 1e-5
    iou = (intersection + smooth) / (num_gt + num_pred - intersection + smooth)
    dice = (2 * intersection + smooth) / (num_gt + num_pred + smooth)

    return dice, iou, mae


# --------------------------------------------------------------------------- #
# 测试
# --------------------------------------------------------------------------- #
@torch.no_grad()
def test(model, data_path, dataset, device, trainsize=352, verbose=False):
    image_root = os.path.join(data_path, dataset, 'images')
    gt_root = os.path.join(data_path, dataset, 'masks')
    model.eval()
    loader = test_dataset(image_root, gt_root, trainsize)

    dscs, ious, maes = [], [], []
    for i in range(loader.size):
        image, gt, name = loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.to(device, non_blocking=True)

        res, res1 = model(image)
        res1 = F.interpolate(res1, res.shape[-2:], mode='bilinear', align_corners=False)
        # 评估时用两个深监督输出相加再上采样到原图尺寸
        res = F.interpolate(res + res1, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().float().cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        pred = np.reshape(res, (-1)) >= 0.5
        target = np.reshape(np.array(gt), (-1))
        dice, iou, mae = calcuate_score(target, pred)
        dscs.append(dice)
        ious.append(iou)
        maes.append(mae)
        if verbose:
            print('  {:>4d}/{:d} {} dice={:.4f}'.format(i + 1, loader.size, name, dice))

    model.train()
    return float(np.mean(dscs)), float(np.mean(ious)), float(np.mean(maes))


def discover_test_datasets(test_path):
    """返回 (test_root, [dataset_name, ...])

    支持两种布局：
      1) test_path/<name>/{images,masks}   -> 多数据集（README 里那份 TestDataset）
      2) test_path/{images,masks}          -> 单数据集，名字取目录名
    找不到就返回空列表，训练照常进行，只是不做评估。
    """
    if not test_path:
        return None, []
    test_path = os.path.abspath(test_path)
    if not os.path.isdir(test_path):
        print('[warn] --test_path 不存在，跳过评估: {}'.format(test_path))
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


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def train(train_loader, model, optimizer, epoch, opt, device, test_root, test_datasets, save_path,
          scaler=None, amp_enabled=False):
    global best
    model.train()
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    total_step = len(train_loader)
    last_step = total_step if not opt.max_steps else min(total_step, opt.max_steps)

    t0 = time.time()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).to(device, non_blocking=True)
            gts = Variable(gts).to(device, non_blocking=True)
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize),
                                       mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize),
                                    mode='bilinear', align_corners=True)
            # ---- forward ----
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp_enabled):
                p1, p2 = model(images)
                # ---- loss ----
                loss = structure_loss(p1, gts) + structure_loss(p2, gts)
            # ---- backward ----
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)   # 先反缩放再裁剪，否则裁的是放大后的梯度
                clip_gradient(optimizer, opt.clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_gradient(optimizer, opt.clip)
                optimizer.step()
            # ---- recording ----
            if rate == 1:
                loss_record.update(loss.data.item(), opt.batchsize)

        if i % opt.print_every == 0 or i == last_step:
            elapsed = time.time() - t0
            eta = elapsed / i * (last_step - i)
            print('[E{:03d}/{:03d}] step {:4d}/{:4d} ({:5.1f}%)  loss {:0.4f}  avg {:0.4f}  '
                  'lr {:.2e}  elapsed {}  eta {}'.format(
                      epoch, opt.epoch, i, last_step, 100.0 * i / last_step,
                      loss_record.val, loss_record.show(),
                      optimizer.param_groups[0]['lr'],
                      fmt_time(elapsed), fmt_time(eta)), flush=True)
        if i >= last_step:
            break

    # ---- save checkpoint ----
    latest_path = os.path.join(save_path, 'latest')
    best_path = os.path.join(save_path, 'best')
    os.makedirs(latest_path, exist_ok=True)
    os.makedirs(best_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(latest_path, 'latest.pth'))
    if epoch % opt.save_every == 0 or epoch == opt.epoch:
        torch.save(model.state_dict(), os.path.join(latest_path, 'epoch_{}.pth'.format(epoch)))

    if not test_datasets or epoch % opt.val_every != 0:
        print('[E{:03d}/{:03d}] 训练完成，用时 {}，跳过验证'.format(
            epoch, opt.epoch, fmt_time(time.time() - t0)), flush=True)
        return None

    # ---- validation ----
    print('[E{:03d}/{:03d}] 训练完成，用时 {}，开始验证...'.format(
        epoch, opt.epoch, fmt_time(time.time() - t0)), flush=True)
    tv0 = time.time()
    results, dices, ious, maes = [], [], [], []
    for dataset in test_datasets:
        dsc, iou, mae = test(model, test_root, dataset, device, opt.trainsize)
        dices.append(dsc)
        ious.append(iou)
        maes.append(mae)
        results.append([dataset, dsc, iou, mae])
        print('  [{:<20s}] dsc={:.4f} iou={:.4f} mae={:.4f}'.format(dataset, dsc, iou, mae),
              flush=True)
        if wandb is not None and wandb.run is not None:
            wandb.log({'dsc/{}'.format(dataset): dsc,
                       'iou/{}'.format(dataset): iou,
                       'mae/{}'.format(dataset): mae}, step=epoch)

    mean_dice = float(np.mean(dices))
    results.append(['mean', mean_dice, float(np.mean(ious)), float(np.mean(maes))])
    tab = tabulate(results, headers=['dataset', 'dsc', 'iou', 'mae'], floatfmt='.3f')
    print(tab)
    with open(os.path.join(save_path, 'results_log.txt'), 'a', encoding='utf-8') as f:
        f.write('epoch {}\n{}\n\n'.format(epoch, tab))
    if wandb is not None and wandb.run is not None:
        wandb.log({'mean_dsc': mean_dice}, step=epoch)

    if mean_dice > best:
        best = mean_dice
        torch.save(model.state_dict(), os.path.join(best_path, 'best.pth'))
        torch.save(model.state_dict(), os.path.join(best_path, 'best_epoch_{}.pth'.format(epoch)))
        print('got best dice {:.4f} at epoch {}'.format(best, epoch).center(70, '='), flush=True)

    print('[E{:03d}/{:03d}] 本轮总计 {}（验证 {}），best mean dice = {:.4f}'.format(
        epoch, opt.epoch, fmt_time(time.time() - t0), fmt_time(time.time() - tv0), best),
        flush=True)

    model.train()
    return mean_dice


# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=300,
                        help='epoch number（论文用 300）')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='初始学习率（论文用 1e-3）')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Adam / AdamW / SGD（论文用 Adam）')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减（论文没写，沿用原代码的 1e-4）')
    parser.add_argument('--scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'none'],
                        help='poly=论文的多项式衰减(power 0.9)；step=原代码分段衰减；none=恒定')
    parser.add_argument('--poly_power', type=float, default=0.9,
                        help='poly 衰减的幂次（论文用 0.9）')
    parser.add_argument('--augmentation', action='store_true',
                        help='开启随机旋转/翻转（原版默认关闭；你只有 492 张图，建议开启）')
    parser.add_argument('--batchsize', type=int, default=16,
                        help='training batch size（3090 24G 用 16，和论文一致）')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='step 衰减的倍率')
    parser.add_argument('--decay_epoch', type=int, default=100, help='step 衰减的周期')
    parser.add_argument('--val_every', type=int, default=5, help='every n epochs run evaluation')
    parser.add_argument('--max_steps', type=int, default=0,
                        help='每个 epoch 最多跑几个 iteration（0=不限制）。用于快速干跑验证整条流水线')
    parser.add_argument('--print_every', type=int, default=5,
                        help='每几个 iteration 打印一次进度（默认 5）')
    parser.add_argument('--save_every', type=int, default=10,
                        help='every n epochs save an epoch_XXX.pth (latest.pth 每个 epoch 都存)')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=1234, help='random seed, <0 表示不设置')
    parser.add_argument('--n_classes', type=int, default=1, help='output channels of the model')

    parser.add_argument('--train_path', type=str, required=True,
                        help='训练集目录，内含 images/ 和 masks/')
    parser.add_argument('--test_path', type=str, default=None,
                        help='测试集根目录，内含 CVC-ClinicDB/ 等子目录（各自有 images/ masks/）')
    parser.add_argument('--train_save', type=str, required=True, help='checkpoint 保存根目录')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='PVTv2-b2 预训练权重，默认用 PolypSeg/pvt_pth/pvt_v2_b2.pth')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='不使用 PVT 预训练权重，整个网络从随机初始化开始训练（预期会明显掉点）')
    parser.add_argument('--init_from', type=str, default=None,
                        help='从已有 checkpoint 初始化整个模型（微调/过拟合自检），例如 runs/UNetV2/best/best.pth')
    parser.add_argument('--device', type=str, default='auto',
                        help="auto / cuda / cuda:0 / cpu")
    parser.add_argument('--amp', action='store_true',
                        help='fp16 混合精度。3090 用 fp32(batch 16) 就够，一般不用开')
    parser.add_argument('--no_tf32', action='store_true',
                        help='关闭 TF32。Ampere 及以后的卡（3090）默认开 TF32，fp32 训练能快接近一倍，'
                             '精度基本无损')

    parser.add_argument('--wandb', action='store_true', help='开启 wandb（默认关闭）')
    parser.add_argument('--wandb_project', type=str, default='Polyp_UNetV2')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_key', type=str, default=None,
                        help='不填则读取环境变量 WANDB_API_KEY / 已有的 ~/.netrc')

    return parser.parse_args()


def resolve_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def main():
    global best
    # 让 print 立刻刷出来（重定向到文件 / 被别的程序捕获时，Python 默认是块缓冲，会看不到实时输出）
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    opt = parse_args()

    if opt.seed is not None and opt.seed >= 0:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)

    device = resolve_device(opt.device)
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        print('[device] {} ({}, {:.0f} MB, sm_{}{})'.format(
            device, torch.cuda.get_device_name(device),
            props.total_memory / 1024 ** 2, props.major, props.minor))
        torch.backends.cudnn.benchmark = True
        # Ampere 及以后（3090/4090/A100…）默认允许 TF32：fp32 的卷积和矩阵乘会走
        # TF32 张量核，速度接近翻倍，精度损失基本可忽略。老卡上这些开关是无害的。
        if not opt.no_tf32:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print('[tf32] 已开启（Ampere+ 上 fp32 训练提速明显；要关加 --no_tf32）')
            except Exception:
                pass
    else:
        print('[device] {}  <-- 没有 GPU 会很慢，建议换有 CUDA 的机器'.format(device))

    # ---------------- 数据检查 ----------------
    image_root = os.path.join(opt.train_path, 'images')
    gt_root = os.path.join(opt.train_path, 'masks')
    for d in (image_root, gt_root):
        if not os.path.isdir(d):
            raise SystemExit('[ERROR] 找不到训练集目录: {}\n'
                             '        期望结构: {}/images/* 和 {}/masks/*'.format(
                                 d, opt.train_path, opt.train_path))

    test_root, test_datasets = discover_test_datasets(opt.test_path)
    if test_datasets:
        print('[test] 将评估以下数据集: {}'.format(', '.join(test_datasets)))
    else:
        print('[test] 没有找到可评估的测试集，只训练不评估')

    # ---------------- 模型 ----------------
    if opt.no_pretrained:
        pretrained = None
        print('[pretrained] 已按 --no_pretrained 关闭：整个网络从随机初始化开始训练')
    else:
        pretrained = opt.pretrained_path or DEFAULT_PRETRAINED
        if not os.path.isfile(pretrained):
            print('[warn] 找不到 PVT 预训练权重 {}，将使用随机初始化（强烈建议补上）'.format(pretrained))
            pretrained = None
        else:
            print('[pretrained] {}'.format(pretrained))

    model = UNetV2(n_classes=opt.n_classes, deep_supervision=True,
                   pretrained_path=pretrained).to(device)
    model_name = model.__class__.__name__
    print(model_name.center(50, '='))

    # 从已有 checkpoint 初始化整个模型（微调 / 过拟合自检用）
    if opt.init_from:
        if not os.path.isfile(opt.init_from):
            raise SystemExit('[ERROR] --init_from 找不到文件: {}'.format(opt.init_from))
        sd = torch.load(opt.init_from, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        miss, unexp = model.load_state_dict(sd, strict=False)
        print('[init_from] {}   (missing {} keys / unexpected {} keys)'.format(
            opt.init_from, len(miss), len(unexp)))

    save_path = os.path.join(opt.train_save, model_name)
    os.makedirs(save_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(save_path, 'train_log.log'),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    print('[save] checkpoint -> {}'.format(save_path))

    # 把 stdout/stderr 同时写进 <save_path>/train.log
    # 这样不管前台跑还是 nohup 跑，plot_curve.py 都能拿到 loss / lr 曲线
    try:
        _log_fp = open(os.path.join(save_path, 'train.log'), 'a',
                       encoding='utf-8', buffering=1)
        sys.stdout = _Tee(sys.__stdout__, _log_fp)
        sys.stderr = _Tee(sys.__stderr__, _log_fp)
        print('[log] 训练日志同步写入 {}'.format(os.path.join(save_path, 'train.log')))
    except Exception as _e:
        print('[warn] 无法写 train.log: {}'.format(_e))

    # ---------------- wandb ----------------
    if opt.wandb:
        if wandb is None:
            print('[warn] 没装 wandb，已忽略 --wandb')
        else:
            if opt.wandb_key:
                wandb.login(key=opt.wandb_key)
            wandb.init(project=opt.wandb_project,
                       name=opt.wandb_name or model_name,
                       dir=save_path,
                       config=vars(opt))

    best = 0.0
    params = model.parameters()
    opt_name = opt.optimizer.lower()
    if opt_name == 'adam':
        optimizer = torch.optim.Adam(params, opt.lr, betas=(0.9, 0.999),
                                     weight_decay=opt.weight_decay)
    elif opt_name == 'adamw':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=opt.weight_decay)
    elif opt_name == 'sgd':
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=opt.weight_decay, momentum=0.9)
    else:
        raise SystemExit('[ERROR] 不认识的 --optimizer {}（可选 Adam / AdamW / SGD）'.format(opt.optimizer))
    print(optimizer)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              augmentation=opt.augmentation,
                              num_workers=opt.num_workers)

    # ---------------- AMP ----------------
    amp_enabled = bool(opt.amp) and device.type == 'cuda'
    scaler = None
    if amp_enabled:
        try:
            scaler = torch.amp.GradScaler('cuda', enabled=True)      # torch >= 2.4
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=True)         # 老版本兜底
        print('[amp] 已开启混合精度 fp16')
    elif opt.amp:
        print('[amp] 当前 device 不是 cuda，自动关闭 AMP')

    print('# ' * 10, 'Start Training', ' #' * 10)
    print('[config] optimizer={} lr={} wd={} scheduler={}{} epoch={} batch={} size={} '
          'amp={} workers={} augmentation={} seed={}'.format(
              opt.optimizer, opt.lr, opt.weight_decay, opt.scheduler,
              '(power={})'.format(opt.poly_power) if opt.scheduler == 'poly' else '',
              opt.epoch, opt.batchsize, opt.trainsize,
              'on' if amp_enabled else 'off(fp32)', opt.num_workers,
              'on' if opt.augmentation else 'off', opt.seed))
    print('[info] {} iterations / epoch, {} epochs'.format(len(train_loader), opt.epoch))

    for epoch in range(1, opt.epoch + 1):
        cur_lr = set_lr(optimizer, opt.lr, epoch, opt)
        if epoch == 1 or epoch % 10 == 0 or epoch == opt.epoch:
            print('[lr] epoch {} -> {:.3e}'.format(epoch, cur_lr), flush=True)
        try:
            train(train_loader, model, optimizer, epoch, opt, device,
                  test_root, test_datasets, save_path, scaler, amp_enabled)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                raise SystemExit(
                    '[ERROR] 显存不足（epoch {}）。把 --batchsize 调小（16 -> 8 -> 4 -> 2）'
                    '并确认加了 --amp；GTX 1660 Ti 只有 6G。'.format(epoch))
            raise

    print('Training done. best mean dice = {:.4f}'.format(best))
    print('best checkpoint: {}'.format(os.path.join(save_path, 'best', 'best.pth')))


if __name__ == '__main__':
    main()
