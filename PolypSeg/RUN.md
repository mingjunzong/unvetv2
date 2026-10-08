# U-Net v2 / PolypSeg 运行说明（改造版）

原仓库里 `PolypSeg/Train.py` 是从作者自己的 ND 集群上直接拷出来的，硬编码了 `/afs/crc.nd.edu/...`、
作者本人的 wandb key 和 `.cuda()`。这份改造把这些问题都处理掉了，你可以直接在有 GPU 的机器上跑。

## 1. 环境（建议 Python 3.10）

`requirements.txt` 是作者从自己环境里 `pip freeze` 出来的，里面有冲突条目
（`Pillow` 9.3.0/10.0.0、`PyYAML` 三个版本、`SimpleITK` 两个版本），
而且 `mmcv_full==1.2.7 + mmdet==3.1.0 + mmseg==1.3.0` 互相不兼容、在 Windows 上基本编不过。
**这三个 mm 系列的包只有 `nnunetv2/training/network/model/poolformer.py` 用到，跟 U-Net v2 无关，不要装。**

```bash
conda create -n unetv2 python=3.10 -y
conda activate unetv2

# 按你的 CUDA 版本选一个（cu118 最通用）
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

pip install timm==0.9.5 numpy==1.23.0 opencv-python==4.5.3.56 Pillow==9.3.0 \
            scipy==1.11.2 tabulate==0.8.9 tqdm==4.62.3 matplotlib==3.4.3
```

要点：

* **`timm` 必须是 0.9.x**。`unet_v2/pvtv2.py` 里有 `from timm.models.vision_transformer import _cfg`，
  timm 1.0 已经删掉了 `_cfg`，装最新版会直接 ImportError。
* `thop` 不用装（改造后是延迟导入，只有 `CalParams` 会用到）。
* `wandb` 不用装（改造后默认关闭；装了也不影响）。
* `opencv-python` 只在 `Test.py` 存预测图时用到。

## 2. 数据集

`utils/dataloader.py` 期望的结构是：

```
TrainDataset/
├── images/           # 训练图
└── masks/            # 对应 mask，和图片同名（扩展名可以不同）

TestDataset/
├── CVC-ClinicDB/{images,masks}
├── CVC-300/{images,masks}          # 这几个都是可选的
├── Kvasir/{images,masks}
├── CVC-ColonDB/{images,masks}
└── ETIS-LaribPolypDB/{images,masks}
```

* 如果你下的是**作者 Google Drive 里那份** `TrainDataset`/`TestDataset`，结构就是这个，直接用。
* 如果你只有**原始的 CVC-ClinicDB（612 张）**，用附带的脚本整理 + 划分：

```bash
python prepare_cvc_clinicdb.py \
  --raw /data/CVC-ClinicDB \
  --out_train /data/polyp/TrainDataset \
  --out_test  /data/polyp/TestDataset \
  --n_train 550 --seed 1234
```

  550/62 是 PraNet / Polyp-PVT 这一系列的通用划分比例（出处见
  [MedSeg-7D 的划分说明](https://huggingface.co/datasets/MaybeRichard/MedSeg-7D/commit/7da79a9e7a082af9a044f9388eddd397074cc440)），
  但**具体哪 550 张是随机划分的**，本脚本只是给你一个可复现的划分；
  要和论文数字逐张对齐，得用作者发布的那份 split。

`--test_path` 下**有哪些子数据集就评估哪些**，缺的不会报错（原版会硬编码去读 5 个数据集，
你只有 CVC-ClinicDB 的话第 1 个 epoch 训完就会崩）。

## 3. 训练

```bash
# 方式一：脚本
bash train.sh /data/polyp/TrainDataset /data/polyp/TestDataset ./checkpoints

# 方式二：直接调
python Train.py \
  --train_path /data/polyp/TrainDataset \
  --test_path  /data/polyp/TestDataset \
  --train_save ./checkpoints \
  --epoch 100 --batchsize 16 --lr 1e-4 --trainsize 352 \
  --device auto --num_workers 4
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--device` | `auto`（默认）/ `cuda` / `cuda:1` / `cpu` |
| `--batchsize` | 原版 16。显存不够就降到 8、4（352×352 下 Polyp-PVT 系列一般 16 需要 ~16G） |
| `--augmentation` | 加上这个 flag 才开随机旋转/翻转（原版默认关，且传参方式有 bug 导致永远关着） |
| `--val_every` | 每几轮评估一次，默认 1 |
| `--save_every` | 每几轮存一个 `epoch_XXX.pth`，默认 10；`latest.pth` 每轮都覆盖 |
| `--pretrained_path` | 默认用 `PolypSeg/pvt_pth/pvt_v2_b2.pth`（仓库里已带，101MB） |
| `--wandb` | 加上才开；默认关闭，评估表格直接 print 并写 `results_log.txt` |

输出目录：

```
checkpoints/UNetV2/
├── latest/latest.pth              # 每轮覆盖
├── latest/epoch_XXX.pth           # 每 save_every 轮
├── best/best.pth                  # 平均 Dice 最好的那轮（覆盖式）
├── best/best_epoch_XXX.pth
├── results_log.txt
└── train_log.log
```

## 4. 测试

```bash
python Test.py --test_path /data/polyp/TestDataset \
               --pth_path ./checkpoints/UNetV2/best/best.pth \
               --save_masks          # 可选，把预测图存成 png
```

会打印每个数据集的 Dice / IoU / MAE 表格，并存到 `<save_path>/test_results.txt`。

> 原来的 `Test.py` 已经不能用了：顶部 `from scipy import misc` 在 SciPy ≥ 1.12 被删除，
> 而且它是为 `PolypPVT` 写的。这里按 `UNetV2` 重写了。

## 5. 这次改了哪些文件

| 文件 | 改动 |
|---|---|
| `PolypSeg/Train.py` | 重写：路径参数化、device 自适应、wandb 可选、测试集自动发现、`reduce=` → `reduction=`、checkpoint 不再每轮无限堆、**新增 `--amp` 混合精度和 `--max_steps` 干跑参数**、OOM 友好报错 |
| `PolypSeg/Test.py` | 重写：改用 `UNetV2`、去掉失效的 `scipy.misc`、支持 GPU/CPU、输出指标表 |
| `PolypSeg/utils/dataloader.py` | mask 不再只认 `.png`（原来 `.tif` 会被静默丢弃）；按文件名配对而不是靠下标对齐；修 `augmentation` 判断 bug；`RandomRotation(resample=)` 改 `interpolation=`；路径/文件名跨平台 |
| `PolypSeg/utils/utils.py` | `thop` 改为函数内延迟导入；`AvgMeter.show()` 原来 `torch.stack` 只接受 Tensor，改成对 float 也成立（原版靠 `loss.data` 传张量才没暴露） |
| `PolypSeg/unet_v2/pvtv2.py` 等 3 处 | timm 导入容错：兼容 timm 1.x（`timm.layers` / `_cfg` 被移除），否则装新 timm 直接 ImportError |
| `PolypSeg/unet_v2/UNet_v2.py`、`unet_v2/UNet_v2.py` | `torch.load` 加 `map_location` + `weights_only` 兜底（兼容 torch ≥ 2.6 / 2.12） |
| `PolypSeg/train.sh`、`PolypSeg/test.sh` | 补上必需参数（原来只有一行 `python Train.py`，argparse 会直接退出） |
| `PolypSeg/prepare_cvc_clinicdb.py` | 新增：整理/划分原始 CVC-ClinicDB |
| `PolypSeg/smoke_test.py` | 新增：不依赖数据集的自检（环境→权重加载→前后向→推理） |
| `PolypSeg/RUN.md` | 本文件 |

## 6. 一个最小自检（不依赖数据集）

```bash
python smoke_test.py            # 自动选设备
python smoke_test.py --device cpu
```

会检查：环境/timm 版本 → PVT 权重加载 → 深监督前向（2 个输出）→ 反向传播 → 推理前向，
最后打印 `SMOKE TEST PASSED`。显存不够会在这里就先暴露出来。

## 7. 远程机器（rlaptop / 10.128.137.111）上的具体路径

那台 **天选笔记本（Windows）** 上的实际情况：

| 项 | 值 |
|---|---|
| GPU | **RTX 3090，24GB**（之前在 GTX 1660 Ti 6GB 上验证过） |
| conda 环境 | `C:\Users\ming\anaconda3\envs\learn`（Python 3.10.20，torch 2.12.1+cu126 ✅） |
| 环境里的 python | `C:\Users\ming\anaconda3\envs\learn\python.exe` |
| 代码位置 | `C:\Users\ming\Desktop\learn_deep\U-Net_v2\`（已部署） |
| 数据 junction | `C:\Users\ming\Desktop\learn_deep\unetv2_data\`（指向你原有的 492/120 划分） |
| 已装依赖 | timm 0.9.5、tabulate（装进 `learn` 环境） |

**已验证可用**（在 GTX 1660 Ti 那台上实跑过；3090 上参数更宽裕，只会更容易）：

* `smoke_test.py` → `SMOKE TEST PASSED`，UNetV2 参数量 25.15M，batch 2 前向+反向峰值显存 **993 MB**
* 深监督训练时返回 `[(B,1,352,352), (B,1,176,176)]`（第二个输出是半分辨率，代码里有插值兜底）
* 干跑 `--epoch 1 --max_steps 3 --batchsize 4 --amp` → 训练 + 120 张验证 + 出表 + 存 checkpoint 全通，exit 0
* 干跑 `--batchsize 8 --amp --num_workers 4` → **也没 OOM**，Windows 多进程 DataLoader 正常

**双击运行（推荐）**：

| 脚本 | 作用 |
|---|---|
| `START_VERIFY.bat` | **先跑这个**：5 个 epoch 快速确认 loss 在降、dsc 在涨 |
| `START_TRAIN.bat` | 正式训练：按论文配方 300 epoch（RTX 3090 档位） |
| `START_TEST.bat` | 用 `best.pth` 在测试集上出 dsc/iou/mae 表 |
| `START_SMOKE_TEST.bat` | 不依赖数据集的环境+模型自检 |

用记事本打开 `.bat` 就能改参数。

**论文配方 vs 代码原默认值**（这次已经按论文调好）：

| | 论文 §3.2 | 代码原默认（Polyp-PVT 那套） | 现在 |
|---|---|---|---|
| 优化器 | Adam | AdamW | **Adam** |
| 初始 lr | 1e-3 | 1e-4 | **1e-3** |
| LR 衰减 | 多项式 power 0.9 | `adjust_lr`，实际**全程不衰减** | **poly 0.9** |
| epoch | 300 | 100 | **300** |
| batch | — | 16 | **16** |
| c | 32 | 32 ✓ | 32 |

> 原来那个 `adjust_lr` 还有个 bug：它是把 decay 乘在"当前 lr"上，
> 一旦 epoch 超过 `decay_epoch` 就会**每个 epoch 再乘一次**，lr 会指数崩掉。
> 现在已经换成 `set_lr()` 的绝对赋值实现（`lr = base_lr * (1 - e/epoch)^0.9`）。

**命令行等价写法**：

```powershell
$py = "C:\Users\ming\anaconda3\envs\learn\python.exe"
cd "C:\Users\ming\Desktop\learn_deep\U-Net_v2\PolypSeg"

& $py Train.py `
  --train_path "C:\Users\ming\Desktop\learn_deep\unetv2_data\TrainDataset" `
  --test_path  "C:\Users\ming\Desktop\learn_deep\unetv2_data\TestDataset" `
  --train_save "C:\Users\ming\Desktop\learn_deep\unetv2_runs_3090" `
  --epoch 300 --batchsize 16 --lr 1e-3 --optimizer Adam --scheduler poly --poly_power 0.9 `
  --val_every 5 --num_workers 8 --device auto --seed 1234
```

只跑 5 轮验证一下（就是 `START_VERIFY.bat` 干的事）：

```powershell
& $py Train.py --train_path "...\unetv2_data\TrainDataset" --test_path "...\unetv2_data\TestDataset" `
  --train_save "$env:TEMP\unetv2_verify" --epoch 5 --batchsize 16 --lr 1e-3 `
  --optimizer Adam --scheduler poly --val_every 1 --num_workers 8
```

**RTX 3090（24G）的参数建议**：

| 参数 | 说明 |
|---|---|
| `--batchsize` | **16**，和论文一致；显存富余想更快可以试 24/32 |
| `--amp` | 一般**不用开**。fp32 + batch 16 在 24G 上很宽裕，而且论文就是 fp32 |
| TF32 | **默认已开**（Ampere 专属，fp32 走 TF32 张量核，提速接近一倍，精度影响可忽略）；要关加 `--no_tf32` |
| `--num_workers` | **8**（3090 太快，worker 少了 GPU 会等数据） |
| `--val_every` | 5，验证是纯 Python 单张循环，跟 GPU 无关，别每轮都跑 |
| `--scheduler` | `poly`（论文）/ `step`（原版分段）/ `none` |
| `--print_every` | 默认 5，显示进度百分比 + loss + 平均 loss + lr + 已用时间 + **ETA** |

耗时估算（492 张训练图、batch 16、300 epoch）：**每 epoch 约 30~50 秒，总计约 3~5 小时**。
第 1 个 epoch 结束会打印 `[E001/300] 本轮总计 ...`，拿那个数 ×300 就是准确值。

> 注意：492 张图比论文的训练集（Kvasir 900 + ClinicDB 548）小不少，
> 想要和论文数字可比，得把 Kvasir 也加进来。只有 CVC 这 492 张时，
> 建议加 `--augmentation`（随机旋转+翻转）抑制过拟合。

## 8. 关于 nnUNet 那条线（ISIC / Dataset123_Polyp）

`nnunetv2/` 是作者改过的 nnU-Net fork，走那半边的话：

* README 里的 `python /path/to/UNet_v2/run/run_training.py` 路径是错的，实际是
  `nnunetv2/run/run_training.py`。
* 没有 `setup.py`/`pyproject.toml`，不能 `pip install -e .`，作者当年是靠
  `export PYTHONPATH=<repo 根目录>:$PYTHONPATH` 跑的。
* README 命令里的 `--c` 是“续训”，全新训练要去掉。
* `ISICTrainer`/`PolypTrainer` 在 `--no-debug` 下会 `wandb.login(作者key)`，
  同样建议改成离线或去掉。
* 走的也是 `--device cpu` 没有意义，那半边基本必须用 GPU。

CVC-ClinicDB 对应的其实是 `Dataset123_Polyp` + `PolypTrainer`（README 没写），
不是 README 第 1 节的 ISIC。如果你想复现论文里的息肉数字，告诉我，
我把那半边的启动命令和 eval 脚本一起整理出来。
