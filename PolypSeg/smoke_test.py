"""最小自检：不依赖数据集，验证 timm/PVT 权重加载/前向/反向是否都正常。

    python smoke_test.py                # 自动选设备
    python smoke_test.py --device cpu
"""
import argparse
import os

import torch

from unet_v2.UNet_v2 import UNetV2

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='auto')
    ap.add_argument('--size', type=int, default=352)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--pretrained_path', default=os.path.join(HERE, 'pvt_pth', 'pvt_v2_b2.pth'))
    opt = ap.parse_args()

    if opt.device == 'auto':
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(opt.device)

    print('[env] python', '.'.join(map(str, __import__('sys').version_info[:3])))
    print('[env] torch', torch.__version__, '| cuda', torch.version.cuda,
          '| available', torch.cuda.is_available())
    if dev.type == 'cuda':
        print('[env] gpu', torch.cuda.get_device_name(dev),
              '| {:.0f} MB'.format(torch.cuda.get_device_properties(dev).total_memory / 1024 ** 2))
    print('[env] device', dev)

    import timm
    print('[env] timm', timm.__version__)

    if not os.path.isfile(opt.pretrained_path):
        raise SystemExit('[ERROR] 找不到 PVT 预训练权重: {}'.format(opt.pretrained_path))

    model = UNetV2(n_classes=1, deep_supervision=True,
                   pretrained_path=opt.pretrained_path).to(dev)
    n_param = sum(p.numel() for p in model.parameters())
    print('[model] UNetV2  参数量 {:.2f} M'.format(n_param / 1e6))

    x = torch.rand(opt.batch, 3, opt.size, opt.size, device=dev)

    model.train()
    ys = model(x)
    print('[fwd] deep_supervision=True  ->', [tuple(y.shape) for y in ys])
    assert isinstance(ys, list) and len(ys) == 2, '深监督训练时应返回 2 个输出'
    assert tuple(ys[0].shape) == (opt.batch, 1, opt.size, opt.size), tuple(ys[0].shape)

    loss = sum(y.float().mean() for y in ys)
    loss.backward()
    print('[bwd] loss.backward() OK')

    model.eval()
    model.deep_supervision = False
    with torch.no_grad():
        y = model(x)
    print('[fwd] deep_supervision=False ->', tuple(y.shape))
    assert tuple(y.shape) == (opt.batch, 1, opt.size, opt.size), tuple(y.shape)

    if dev.type == 'cuda':
        print('[mem] peak {:.0f} MB / total {:.0f} MB'.format(
            torch.cuda.max_memory_allocated(dev) / 1024 ** 2,
            torch.cuda.get_device_properties(dev).total_memory / 1024 ** 2))

    print('\nSMOKE TEST PASSED')


if __name__ == '__main__':
    main()
