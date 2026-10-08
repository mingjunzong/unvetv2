import torch
import numpy as np
# 注意: thop 只被 CalParams 用到, 改成函数内延迟导入,
# 这样没装 thop 也能正常训练/测试。


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    """原版的分段衰减。注意它有 bug：decay 是乘在"当前 lr"上的，
    一旦 epoch 超过 decay_epoch，每个 epoch 都会再乘一次，lr 会指数崩掉。
    Train.py 已经不用它了（改用 set_lr 的多项式衰减），这里保留但改成基于 init_lr 计算。
    """
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        # 原实现是 torch.mean(torch.stack(...))，要求 losses 里全是 Tensor；
        # 这里改成对 float 也成立，返回 python float。
        return float(np.mean(self.losses[max(len(self.losses) - self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    from thop import profile, clever_format
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))