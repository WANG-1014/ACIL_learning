# CODE_ANALYSIS.md — ACIL 项目代码分析

> **项目名称**: Analytic Class-Incremental Learning (ACIL)  
> **代码仓库**: `ACIL_learning`  
> **分析日期**: 2026-06-05  
> **分析方法**: ACIL / G-ACIL / DS-AL / GKEAL / AEF-OCL / AIR 等一系列基于解析解（闭式解）的增量学习算法  
> **备注**:ClaudeCode生成

---

## 目录

1. [项目概览](#1-项目概览)
2. [核心数学原理](#2-核心数学原理)
3. [文件结构](#3-文件结构)
4. [入口：main.py 详解](#4-入口mainpy-详解)
5. [配置层：config.py 详解](#5-配置层configpy-详解)
6. [算法层：analytic/ 详解](#6-算法层analytic-详解)
7. [数据集层：datasets/ 详解](#7-数据集层datasets-详解)
8. [模型层：models/ 详解](#8-模型层models-详解)
9. [工具层：utils/ 详解](#9-工具层utils-详解)
10. [数据流全景图](#10-数据流全景图)
11. [各方法对比总结](#11-各方法对比总结)

---

## 1. 项目概览

这是一个基于 **解析解（Analytic Solution）** 的类别增量学习（Class-Incremental Learning, CIL）研究代码库。与传统的基于梯度下降的增量学习方法不同，该项目的所有方法都使用 **递归最小二乘法（Recursive Least Squares, RLS）** 来解析地更新分类器权重，实现了：

- **绝对记忆（Absolute Memorization）**：数学上等价于对所有历史数据做岭回归，不会发生灾难性遗忘
- **无需存储旧样本（Exemplar-Free）**：不需要 replay buffer，保护隐私
- **单次前向更新**：新类别到达时，通过闭式解一步到位，无需多轮迭代

### 支持的方法

| 方法 | 论文 | 特点 |
|------|------|------|
| **ACIL** | NeurIPS 2022 | 基础解析增量学习 |
| **G-ACIL** | arXiv 2024 | ACIL 的泛化版本，支持 mini-batch |
| **DS-AL** | AAAI 2024 | 双流解析学习 + 过往标签清洗(PLC) |
| **GKEAL** | CVPR 2023 | 高斯核嵌入，支持小样本增量 |
| **AEF-OCL** | arXiv 2024 | 基于原型的在线增量学习，处理类别不平衡 |
| **AIR** | arXiv 2024 | 解析式不平衡校正器 |

---

## 2. 核心数学原理

### 2.1 岭回归的闭式解

传统分类问题中，给定特征矩阵 $X$ 和 one-hot 标签 $Y$，岭回归的最优权重为：

$$\hat{W} = (X^T X + \gamma I)^{-1} X^T Y$$

其中 $\gamma$ 是正则化系数。

### 2.2 递归更新（Woodbury 矩阵恒等式）

ACIL 的核心创新：当新数据 $(X_{new}, Y_{new})$ 到达时，不需要重新计算整个逆矩阵，而是通过递归公式更新。

定义 $R = (\gamma I + X_{all}^T X_{all})^{-1}$ 为正则化特征自相关矩阵的逆。

递归更新公式（来自 G-ACIL 论文的公式 9 和 10）：

```
步骤1:  K = (I + X_new @ R @ X_new^T)^{-1}          （核矩阵求逆，尺寸为 batch_size × batch_size）
步骤2:  R ← R - R @ X_new^T @ K @ X_new @ R         （Woodbury 更新）
步骤3:  W ← W + R @ X_new^T @ (Y_new - X_new @ W)   （权重更新）
```

**关键洞察**：
- $K$ 的尺寸是 `batch_size × batch_size`，而非特征维度 × 特征维度
- $R$ 是 `in_features × in_features` 的矩阵，编码了所有历史数据的二阶统计信息
- 旧数据不需要存储，因为其信息已完整保留在 $R$ 中

### 2.3 特征扩展（Random Buffer）

在实际应用中，backbone 特征维度（如 768）可能不够高，使得类间线性可分性不足。ACIL 引入了一个**固定的随机投影层**：

$$X_{expanded} = \text{ReLU}(W_{random} \cdot X_{backbone})$$

- $W_{random}$ 随机初始化后**永久冻结**（注册为 `buffer`，不参与训练）
- 投影维度（`buffer_size`，默认 8192）远大于输入维度
- 这与随机傅里叶特征（Random Fourier Features）和极限学习机（ELM）的思想一致

---

## 3. 文件结构

```
ACIL_learning/
├── main.py                  # 入口：编排整个训练流程
├── config.py                # CLI 参数解析 + 方法注册表
├── environment.yaml         # Conda 环境配置
├── README.md                # 用户自己的使用说明（中文）
├── ORIGIN_README.md         # 原始上游 README
├── ORIGIN_README_CN.md      # 原始上游 README（中文翻译）
│
├── analytic/                # 核心算法模块
│   ├── __init__.py           #   公开 API 导出
│   ├── ACIL.py               #   ACIL 模型 + ACILLearner 训练器
│   ├── DSAL.py               #   DS-AL 双流模型 + DSALLearner
│   ├── GKEAL.py              #   GKEAL 高斯核模型 + GKEALLearner
│   ├── AEFOCL.py             #   AEF-OCL 在线模型 + AEFOCLLearner
│   ├── AIR.py                #   AIR 不平衡校正模型 + AIRLearner
│   ├── AnalyticLinear.py     #   解析线性层：RecursiveLinear / GeneralizedARM
│   ├── Buffer.py             #   特征扩展层：RandomBuffer / GaussianKernel
│   └── Learner.py            #   抽象基类 Learner
│
├── datasets/                # 数据集加载与划分
│   ├── __init__.py           #   数据集注册表 + load_dataset()
│   ├── DatasetWrapper.py     #   多阶段类别增量划分核心
│   ├── CIFAR.py              #   CIFAR-10/CIFAR-100 包装器
│   ├── MNIST.py              #   MNIST 包装器
│   ├── ImageNet.py           #   ImageNet-1k 包装器
│   ├── UCMerced.py           #   UCMerced_LandUse 包装器
│   └── Features.py           #   缓存特征加载器（加速增量学习）
│
├── models/                  # 骨干网络注册
│   ├── __init__.py           #   models 字典 + load_backbone()
│   └── CifarResNet.py        #   CIFAR 专用 ResNet (20/32/44/56/110/1202)
│
├── utils/                   # 工具函数
│   ├── __init__.py           #   导出：validate / set_weight_decay / set_determinism / ClassificationMeter
│   ├── validate.py           #   验证循环
│   ├── metrics.py            #   分类指标计算（acc@1, acc@5, F1-micro, F1-macro）
│   ├── set_weight_decay.py  #   分层 weight decay 配置
│   └── set_determinism.py   #   随机种子固定（5层确定性）
│
├── backbones/               # 预训练 backbone 和缓存特征
├── my_dataset/              # UCMerced LandUse 数据集
├── saved_models/            # 训练输出（模型权重 + IL.csv）
└── figures/                 # 结果图表
```

---

## 4. 入口：main.py 详解

### 4.1 文件级导入

```python
# main.py:1-12
import torch                         # 深度学习框架
from os import path                  # 路径操作
from tqdm import tqdm                # 进度条
from config import load_args, ALL_METHODS  # CLI 参数解析 + 方法注册表
from models import load_backbone     # backbone 加载函数
from typing import Any, Dict, List, Tuple, Optional  # 类型注解
from datasets import Features, load_dataset    # 数据集加载
from utils import set_determinism, validate     # 确定性设置 + 验证函数
from torch._prims_common import DeviceLikeType  # 设备类型
from torch.utils.data import Dataset, DataLoader # PyTorch 数据工具
```

### 4.2 `make_dataloader()` — 数据加载器工厂（第15-41行）

```python
def make_dataloader(dataset, shuffle, batch_size, num_workers, device, persistent_workers=False):
```

**参数映射表**（以你的命令为例）：

| 参数 | 值 | 来源 |
|------|-----|------|
| `dataset` | Subset/DatasetWrapper | 数据集划分结果 |
| `shuffle` | True/False | 训练=True, 验证=False |
| `batch_size` | 16 (base) / 64 (IL) | `--batch-size` / `--IL-batch-size` |
| `num_workers` | 4 | `--num-workers` |
| `device` | `cuda:0` | 自动检测 |
| `pin_memory` | True (CUDA时) | 自动设置 |

**功能**：
1. 若 CUDA 可用，设置 `pin_memory=True`（加速 CPU→GPU 传输）
2. 尝试导入 `prefetch_generator.BackgroundGenerator`（异步预取，加速数据加载）
3. 若导入失败，退化为普通 `DataLoader`

### 4.3 `check_cache_features()` — 缓存检查（第44-49行）

```python
def check_cache_features(root: str) -> bool:
    files_list = ["X_train.pt", "y_train.pt", "X_test.pt", "y_test.pt"]
    for file in files_list:
        if not path.isfile(path.join(root, file)):
            return False
    return True
```

检查四个缓存文件是否都存在。若全部存在，跳过特征提取步骤。

### 4.4 `cache_features()` — 特征缓存（第52-66行）

```python
@torch.no_grad()
def cache_features(backbone, dataloader, device):
    backbone.eval()
    X_all, y_all = [], []
    for X, y in tqdm(dataloader, "Caching"):
        X = backbone(X.to(device))     # 仅跑 backbone 前向
        y = y.to(torch.int16)
        X_all.append(X.cpu())          # 搬回 CPU 节省显存
        y_all.append(y.cpu())
    return torch.cat(X_all), torch.cat(y_all)
```

**功能**：对整个数据集跑一次 backbone 前向传播，将输出特征和标签保存为 `.pt` 文件。后续增量学习阶段直接用缓存特征，**跳过 backbone 推理**，速度提升巨大。

### 4.5 `main()` — 主函数（第69-259行）

#### 阶段 A：设备选择（第70-83行）

```python
backbone_name = args["backbone"]  
# → 你的命令：backbone_name = "vit_b_16"
```

```python
# 设备选择的三分支逻辑：
if args["cpu_only"] or not torch.cuda.is_available():
    main_device = torch.device("cpu")          # 分支1: 强制CPU
    all_gpus = None
elif args["gpus"] is not None:
    gpus = args["gpus"]                        # 分支2: 指定GPU列表
    main_device = torch.device(f"cuda:{gpus[0]}")
    all_gpus = [torch.device(f"cuda:{gpu}") for gpu in gpus]
else:
    main_device = torch.device("cuda:0")       # 分支3: 默认单GPU
    all_gpus = None
# → 你的命令：走分支3，main_device = cuda:0, all_gpus = None
```

#### 阶段 B：确定性设置（第85-86行）

```python
if args["seed"] is not None:
    set_determinism(args["seed"])
# → 你的命令：seed=None，跳过。若指定 --seed 42，则执行 set_determinism(42)
```

`set_determinism()` 详解见 [§9.4](#94-set_determinismpy--五层随机种子锁定)。

#### 阶段 C：Backbone 加载（第88-103行）

有两种途径加载 backbone：

**途径 1：从缓存加载**（指定了 `--cache-path`）
```python
if "backbone_path" in args:                   # False（你的命令未指定）
    preload_backbone = True
    backbone, _, feature_size = torch.load(
        args["backbone_path"], map_location=main_device, weights_only=False
    )
```

**途径 2：从 torchvision 加载预训练模型**（你的命令走这条）
```python
else:
    preload_backbone = False
    load_pretrain = (args["base_ratio"] == 0) or ("ImageNet" not in args["dataset"])
    # base_ratio=0.5238>0 → False
    # "ImageNet" not in "UCMerced_LandUse" → True
    # → load_pretrain = True

    backbone, _, feature_size = load_backbone("vit_b_16", pretrain=True)
```

进入 [models/__init__.py:102-118](models/__init__.py#L102-L118) `load_backbone()`：

```python
# models 字典查找：
# "vit_b_16" → (384, vit_b_16, ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
input_img_size, model, weights = models[name]
# → input_img_size = 384  (ViT-B/16 输入分辨率)
# → model = vit_b_16       (torchvision 构造函数)
# → weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1  (SWAG 微调权重)

backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)

# 切掉分类头，获取特征维度：
# isinstance(backbone, VisionTransformer) → True
feature_size = backbone.heads[-1].in_features  # = 768  (ViT-B/16 的隐藏维度)
backbone.heads = torch.nn.Identity()           # 替换分类头为恒等映射
# → 现在 backbone(x) 输出 768 维特征向量

return backbone, 384, 768
```

回到 main：
```python
backbone = backbone.to("cuda:0", non_blocking=True)
```

| 变量 | 值 | 含义 |
|------|-----|------|
| `backbone_name` | `"vit_b_16"` | 骨架网络名称 |
| `backbone` | ViT-B/16 (无头) | 实际模型对象 |
| `feature_size` | `768` | backbone 输出特征维度 |
| `input_img_size` | `384` | 输入图片分辨率 |

#### 阶段 D：数据集加载（第105-113行）

```python
dataset_args = {
    "name": "UCMerced_LandUse",
    "root": "./my_dataset/UCMerced_LandUse",   # load_args 中 data_root + dataset 已拼接
    "base_ratio": 0.5238095238,
    "num_phases": 10,
    "shuffle_seed": None,
}
dataset_train = load_dataset(train=True,  augment=True,  **dataset_args)
dataset_test  = load_dataset(train=False, augment=False, **dataset_args)
```

进入 [datasets/__init__.py:33-55](datasets/__init__.py#L33-L55) `load_dataset()` → `UCMerced_LandUse_(...)`：

**数据集内部构造**（[datasets/UCMerced.py](datasets/UCMerced.py)）：
```
UCMerced_LandUse: 21类 × 100张 = 2100张 .tif 图片 (256×256)
├── 训练集: 每类前 80 张 → 21×80 = 1680 张
└── 测试集: 每类后 20 张 → 21×20 = 420 张
```

**DatasetWrapper 初始化**（[datasets/DatasetWrapper.py:19-49](datasets/DatasetWrapper.py#L19-L49)）：
```python
self.num_classes = 21
self.base_size = int(21 * 0.5238095238) = 11   # 基类: 11 个类（类 0~10）
self.incremental_size = 21 - 11 = 10            # 增量类: 10 个类
self.phase_size = 10 // 10 = 1                  # 每个 phase: 1 个新类
```

**类别划分示意**：
```
Phase 0 (基类):  | 类0 | 类1 | 类2 | ... | 类10 |         ← 11个类, SGD训练
Phase 1 (增量):  | 类11 |                                  ← 1个新类
Phase 2 (增量):  | 类12 |                                  ← 1个新类
...
Phase 10 (增量): | 类20 |                                  ← 1个新类
```

#### 阶段 E：创建 Learner（第116-119行）

```python
assert args["method"] in ALL_METHODS  # "ACIL" in dict → 通过
learner = ALL_METHODS["ACIL"](args, backbone, 768, "cuda:0", all_devices=None)
# → learner = ACILLearner(args, backbone, 768, cuda:0, None)
```

进入 [ACILLearner.__init__](analytic/ACIL.py#L74-L88)：
```python
self.learning_rate = 0.001
self.buffer_size = 2048      # 随机投影目标维度
self.gamma = 0.1             # 正则化系数
self.base_epochs = 300       # 基类训练总 epoch
self.warmup_epochs = 10      # 学习率 warmup 轮数
self.make_model()            # 立即创建 ACIL 模型
```

`make_model()` 创建 [ACIL](analytic/ACIL.py#L28-L46)：
```python
self.model = ACIL(
    backbone_output = 768,
    backbone = backbone,          # ViT-B/16 (单GPU, 不包 DataParallel)
    buffer_size = 2048,           # 随机投影到 2048 维
    gamma = 0.1,                  # 正则化系数 (R = I/γ = 10*I)
    device = "cuda:0",
    dtype = torch.double,        # float64 高精度
    linear = RecursiveLinear,     # 递归最小二乘解析器
)
```

ACIL 模型内部结构：
```
ACIL(
  (backbone): ViT-B/16 → 768维
  (buffer): RandomBuffer(768 → 2048, relu)
    └── weight: 随机初始化 [2048, 768], register_buffer (不训练)
    └── bias: None
  (analytic_linear): RecursiveLinear(2048, gamma=0.1)
    └── weight: [2048, 0] (空矩阵, 等 fit 时动态扩展)
    └── R: I/0.1 = 10*I [2048, 2048] (正则化特征自相关矩阵的逆)
)
```

#### 阶段 F：Base Training（第122-143行）

```python
if args["base_ratio"] > 0 and not preload_backbone:
    # 0.5238>0 → True, preload_backbone=False → True → 进入
```

**F1. 数据划分**：
```python
train_subset = dataset_train.subset_at_phase(0)
# → 仅类 0~10 (11个类) 的训练数据, ~880张图
test_subset = dataset_test.subset_at_phase(0)
# → 仅类 0~10 (11个类) 的测试数据, ~220张图

train_loader = make_dataloader(train_subset, True,  16, 4, device="cuda:0")
test_loader  = make_dataloader(test_subset,  False, 16, 4, device="cuda:0")
```

**F2. 进入 `learner.base_training()`**（[analytic/ACIL.py:90-216](analytic/ACIL.py#L90-L216)）：

```python
# 构造训练网络
model = torch.nn.Sequential(
    self.backbone,                        # ViT-B/16 → 768维
    torch.nn.Linear(768, 11),             # 11分类线性头
).to("cuda:0")

# 优化器
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,          # --learning-rate
    momentum=0.9,      # 默认值
    weight_decay=5e-4, # --weight-decay
)

# 学习率调度器 (warmup + cosine)
# epoch 0~9:   lr: 1e-6 → 0.001    (线性 warmup)
# epoch 10~299: lr: 0.001 → 1e-6    (cosine 退火)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    [
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=10
        ),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=290, eta_min=1e-6
        ),
    ],
    [10],  # 第 10 个 epoch 后切换
)

# 损失函数
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05).to("cuda:0")
```

**F3. 训练循环**：

```python
best_acc = 0.0
for epoch in range(300 + 1):   # epoch = 0, 1, 2, ..., 300
    if epoch != 0:  # epoch=0 仅验证
        model.train()
        for X, y in train_loader:       # X: (16, 3, 384, 384)
            X = X.to("cuda:0")
            y = y.to("cuda:0")
            assert y.max() < 11         # 安全检查

            optimizer.zero_grad()       # 清零梯度
            logits = model(X)           # 前向: (16, 768) → (16, 11)
            loss = criterion(logits, y) # 带 label_smoothing 的 CE Loss
            loss.backward()             # 反向传播
            optimizer.step()            # SGD 更新
        scheduler.step()                # 更新学习率

    # 验证（每个 epoch 都做）
    model.eval()
    train_meter = validate(model, train_loader, 11)
    val_meter   = validate(model, test_loader,  11)

    if val_meter.accuracy > best_acc:
        best_acc = val_meter.accuracy
        if epoch != 0:
            # 保存最佳 backbone, 输入尺寸, 特征维度
            self.save_object((backbone, 384, 768), "backbone.pth")
```

**F4. 训练完成**：
```python
self.backbone.eval()   # 冻结 backbone
self.make_model()      # 重新创建 ACIL 模型（使用训练好的 backbone）
```

#### 阶段 G：Cache Features（第146-200行）

由于你指定了 `--cache-features`：

```python
if args["cache_features"]:  # True
    if "cache_path" not in args or args["cache_path"] is None:
        args["cache_path"] = args["saving_root"]
        # → cache_path = "saved_models/vit_b_16_UCMerced_LandUse_.../ACIL/.../"

    if not check_cache_features(args["cache_path"]):
        # 首次运行 → 提取特征
```

**G1. 特征提取**：
```python
# 重新加载全量数据集（base_ratio=1 包含所有 21 类）
backbone = learner.backbone.eval()
dataset_train = load_dataset(..., base_ratio=1, num_phases=0, augment=False)
dataset_test  = load_dataset(..., base_ratio=1, num_phases=0, augment=False)

# 单次前向，提取所有 ViT 特征
X_train, y_train = cache_features(backbone, train_loader)
# → X_train: (1680, 768) 特征矩阵
# → y_train: (1680,)     标签向量

X_test, y_test = cache_features(backbone, test_loader)
# → X_test:  (420,  768)
# → y_test:  (420,)

# 保存到磁盘
torch.save(X_train, ".../X_train.pt")
torch.save(y_train, ".../y_train.pt")
torch.save(X_test,  ".../X_test.pt")
torch.save(y_test,  ".../y_test.pt")
```

**G2. 切换到缓存模式**：
```python
dataset_train = Features(cache_path, train=True,  base_ratio=0.5238, num_phases=10)
dataset_test  = Features(cache_path, train=False, base_ratio=0.5238, num_phases=10)

# Features 类（datasets/Features.py）：
#   从 X_train.pt / y_train.pt 加载数据 → TensorDataset
#   继承 DatasetWrapper → 同样的 11 base + 10 phase 划分
#   每次 getitem 返回 (768维特征向量, 标签)

# 关键：替换 backbone 为 Identity（特征已缓存，无需再跑 ViT）
learner.backbone = torch.nn.Identity()
learner.model.backbone = torch.nn.Identity()
```

**为什么这样做**：增量学习阶段 backbone 是冻结的，每次都跑 ViT 太浪费。提前提取后，增量阶段只需 `RandomBuffer → RecursiveLinear`，计算量降为原来的 ~1%。

#### 阶段 H：Incremental Learning（第202-259行）

```python
sum_acc = 0
log_file = open(".../IL.csv", "w")
print("phase", "acc@avg", "acc@1", "acc@5", "f1-micro", "loss", file=log_file, sep=",")
```

**H1. Phase 0 — 基类对齐（Re-align）**：

```python
for phase in range(0, 10 + 1):     # phase = 0, 1, ..., 10
    # --- 数据准备 ---
    train_subset = dataset_train.subset_at_phase(phase)
    test_subset  = dataset_test.subset_until_phase(phase)

    # phase=0:
    #   train_subset: 仅类 0~10（11 类）
    #   test_subset:  仅类 0~10（11 类）

    # phase=1:
    #   train_subset: 仅类 11（1 个新类）
    #   test_subset:  类 0~11（12 类，所有见过的类）

    train_loader = make_dataloader(train_subset, True,  64, 4, device="cuda:0")
    test_loader  = make_dataloader(test_subset,  False, 64, 4, device="cuda:0")

    # --- 学习 ---
    if phase == 0:
        learner.learn(train_loader, 11, "Re-align")
    else:
        learner.learn(train_loader, 1)   # 每次 1 个新类
```

**H2. `learner.learn()` 内部**（[analytic/ACIL.py:229-240](analytic/ACIL.py#L229-L240)）：

```python
def learn(self, data_loader, incremental_size, desc="Incremental Learning"):
    self.model.eval()
    for X, y in tqdm(data_loader, desc=desc):
        X = X.to(self.device)  # X: (64, 768) 缓存特征
        y = y.to(self.device)  # y: (64,)
        self.model.fit(X, y, increase_size=incremental_size)
```

**H3. `ACIL.fit()` → `RecursiveLinear.fit()`** 核心算法（[analytic/AnalyticLinear.py:100-128](analytic/AnalyticLinear.py#L100-L128)）：

```python
def fit(self, X, Y):
    # X: (batch_size, 2048) — 经过 RandomBuffer 扩展
    # Y: (batch_size, num_classes) — one-hot 标签

    # 步骤1: 自动扩展权重矩阵（新类尾部补零）
    num_targets = Y.shape[1]
    if num_targets > self.out_features:
        increment_size = num_targets - self.out_features
        tail = torch.zeros((2048, increment_size))
        self.weight = torch.cat((self.weight, tail), dim=1)
        # phase 0: weight (2048, 0) → (2048, 11)
        # phase 1: weight (2048, 11) → (2048, 12)
        # ...
        # phase 10: weight (2048, 20) → (2048, 21)

    # 步骤2: 核矩阵求逆
    K = torch.inverse(torch.eye(batch_size) + X @ self.R @ X.T)
    # K: (batch_size, batch_size) — 大小与 batch 有关，与特征维度无关！

    # 步骤3: Woodbury 更新 R 矩阵 (公式10)
    self.R -= self.R @ X.T @ K @ X @ self.R
    # self.R 始终保持 (2048, 2048)，编码所有历史数据信息

    # 步骤4: 更新权重矩阵 (公式9)
    self.weight += self.R @ X.T @ (Y - X @ self.weight)
    # Y - X @ self.weight: 当前模型在新数据上的残差
    # R @ X.T @ residual: 最优权重校正量
```
![alt text](/figures/MY_README_img6.png)

**递归最小二乘的直观理解**：
- `R` 是全体历史数据协方差矩阵的逆 → 编码了"数据分布"
- `K` 是新 batch 的核矩阵的逆 → 衡量新数据的"信息量"
- `Y - X @ W` 是残差 → "当前模型哪里预测错了"
- 三步结合在一起 → 等价于对所有数据做岭回归的闭式解

**H4. 验证**：

```python
learner.before_validation()   # → model.update() → 数值稳定性检查

val_meter = validate(learner, test_loader, 21)
# validate() 内部:
#   model.eval()
#   for X, y in test_loader:
#       logits = model(X)              # Identity → RandomBuffer → RecursiveLinear
#       meter.record(y, logits)        # 累积预测结果
#   return meter                       # 包含 acc@1, acc@5, F1 等
```

**H5. 日志记录**：

```python
sum_acc += val_meter.accuracy
print(
    f"loss: {val_meter.loss:.4f}",
    f"acc@1: {val_meter.accuracy * 100:.3f}%",
    f"acc@5: {val_meter.accuracy5 * 100:.3f}%",
    f"f1-micro: {val_meter.f1_micro * 100:.3f}%",
    f"acc@avg: {sum_acc / (phase + 1) * 100:.3f}%",  # 所有 phase 的平均准确率
)
print(phase, sum_acc/(phase+1), val_meter.accuracy, ..., file=log_file, sep=",")
```

#### 阶段 I：结束

```python
log_file.close()
# → IL.csv 包含 11 行数据（phase 0~10），每行记录各指标
```

---

### 4.6 程序入口（第262-263行）

```python
if __name__ == "__main__":
    main(load_args())
```

执行顺序：
1. `load_args()` — 解析 CLI 参数 → 返回 `Dict[str, Any]`
2. `main(args)` — 运行完整训练流程

---

## 5. 配置层：config.py 详解

### 5.1 方法注册表（第21-29行）

```python
ALL_METHODS: dict[str, type[Learner]] = {
    "ACIL": ACILLearner,
    "G-ACIL": ACILLearner,       # G-ACIL 等价于 ACIL（ACILLearner 本身就是 G-ACIL 实现）
    "DS-AL": DSALLearner,
    "GKEAL": GKEALLearner,
    "AEF-OCL": AEFOCLLearner,
    "AIR": AIRLearner,
    "G-AIR": GeneralizedAIRLearner,
}
```

`ALL_METHODS` 是一个从方法名字符串到 Learner 类的映射字典。main.py 通过它动态创建 learner。

### 5.2 参数解析器（第33-246行）

核心参数分组：

| 组 | 参数 | 默认值 | 说明 |
|------|------|------|------|
| **方法** | `method` | (必选) | 位置参数，必须是 ALL_METHODS 中的 key |
| **数据集** | `--dataset` / `-d` | `"CIFAR-100"` | 数据集名称 |
| | `--data-root` | `"~/dataset"` | 数据集根目录 |
| | `--num-workers` / `-j` | `8` | DataLoader 工作进程数 |
| | `--base-ratio` | `0.5` | 基类占总类别的比例 |
| | `--phases` | `10` | 增量阶段数 |
| | `--batch-size` / `-b` | `256` | 基类训练的 batch size |
| | `--cache-features` | `False` | 是否缓存 backbone 特征 |
| **模型** | `--backbone` / `-a` | `"resnet32"` | 骨干网络名称 |
| | `--cache-path` | `None` | 预训练 backbone 路径 |
| | `--seed` | `None` | 随机种子 |
| | `--dataset-seed` | `None` | 数据集打乱种子 |
| **基类训练** | `--base-epochs` | `300` | 基类训练总 epoch |
| | `--warmup-epochs` | `10` | 学习率 warmup 轮数 |
| | `--learning-rate` / `-lr` | `0.5` | 初始学习率 |
| | `--momentum` | `0.9` | SGD 动量 |
| | `--weight-decay` / `--wd` | `5e-4` | 权重衰减 |
| | `--separate-decay` | `False` | 分层 weight decay |
| | `--label-smoothing` | `0.05` | 标签平滑系数 |
| **增量学习** | `--IL-batch-size` | `batch_size` | 增量学习 batch size |
| | `--gamma` | `0.1` | 解析线性层的正则化系数 |
| | `--buffer-size` | `8192` | 随机投影缓冲区大小 |
| | `--gamma-comp` | `0.1` | 补偿流的正则化（仅 DS-AL） |
| | `--sigma` | `10` | 高斯核宽度（仅 GKEAL） |
| | `--compensation-ratio` / `-C` | `1` | 补偿流权重（仅 DS-AL） |

### 5.3 `load_args()` 函数（第249-279行）

```python
def load_args() -> Dict[str, Any]:
    global _parser
    args = vars(_parser.parse_args())                    # 1. 解析命令行

    args["data_root"] = path.expanduser(args["data_root"])  # 2. 展开 ~

    if args["cache_path"] is not None:                   # 3. cache_path → backbone_path
        args["backbone_path"] = path.join(args["cache_path"], "backbone.pth")

    # 4. 构造 saving_root
    saving_root = path.join(
        "saved_models",
        f"{args['backbone']}_{args['dataset']}_{args['base_ratio']}_{args['dataset_seed']}",
    )
    if args["exp_name"].strip() == "":
        args["exp_name"] = args["method"]
    saving_root = path.join(saving_root, args["exp_name"])

    if args["IL_batch_size"] is None:                    # 5. IL_batch_size 默认值
        args["IL_batch_size"] = args["batch_size"]

    # 6. 添加时间戳子目录
    current_time = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    saving_root = path.join(saving_root, current_time)
    args["saving_root"] = saving_root
    args["argv"] = str(argv)

    makedirs(saving_root, exist_ok=True)                 # 7. 创建输出目录
    with open(path.join(saving_root, "args.yaml"), "w") as yaml_file:
        yaml.safe_dump(args, yaml_file)                  # 8. 保存参数

    args["data_root"] = path.join(args["data_root"], args["dataset"])  # 9. 拼接数据集路径
    return args
```

**你的命令生成的文件路径**：
```
saved_models/
└── vit_b_16_UCMerced_LandUse_0.5238095238_None/
    └── ACIL/
        └── 2026-06-05Txx-xx-xx/
            ├── args.yaml         ← 保存所有参数
            ├── base_training.csv ← base training 每轮指标
            ├── backbone.pth      ← 最佳 backbone 权重
            ├── X_train.pt        ← 缓存特征 (若 --cache-features)
            ├── y_train.pt
            ├── X_test.pt
            ├── y_test.pt
            └── IL.csv            ← 增量学习每 phase 指标
```

---

## 6. 算法层：analytic/ 详解

### 6.1 `__init__.py` — 公开 API

所有算法组件通过此文件统一导出。

### 6.2 `Learner.py` — 抽象基类

```python
class Learner(metaclass=ABCMeta):
    def __init__(self, args, backbone, backbone_output, device, all_devices):
        self.args = args
        self.backbone = backbone
        self.backbone_output = backbone_output
        self.device = device
        self.all_devices = all_devices
        self.model: torch.nn.Module

    @abstractmethod
    def base_training(self, train_loader, val_loader, baseset_size): ...
    @abstractmethod
    def learn(self, data_loader, incremental_size, desc): ...
    @abstractmethod
    def before_validation(self): ...
    @abstractmethod
    def inference(self, X): ...

    def save_object(self, model, file_name):
        torch.save(model, path.join(self.args["saving_root"], file_name))

    def __call__(self, X):
        return self.inference(X)  # 让 Learner 实例可被当作函数调用
```

**接口约定**：
- `base_training()` — 在基类数据上用 SGD 训练 backbone
- `learn()` — 增量学习新类（解析更新，不训练 backbone）
- `before_validation()` — 验证前钩子（如数值检查、数据增强）
- `inference()` — 前向推理
- `save_object()` — 保存模型到 `saving_root`

### 6.3 `AnalyticLinear.py` — 解析线性层

#### `AnalyticLinear` — 抽象基类

```python
class AnalyticLinear(torch.nn.Linear, metaclass=ABCMeta):
    def __init__(self, in_features, gamma, bias, device, dtype):
        # 跳过 nn.Linear 的原始 __init__
        super(torch.nn.Linear, self).__init__()
        self.gamma = gamma
        # weight 注册为 buffer（非参数），形状为 (in_features, 0)
        # 初始为空，随着新类别到达动态扩展列
        weight = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("weight", weight)
```

**设计要点**：
- 继承 `nn.Linear` 但不使用其参数机制
- `weight` 是 `buffer`，不会出现在 `parameters()` 中
- `out_features` 随 `fit()` 调用动态增长（新类 → 补零扩展）

#### `RecursiveLinear` — ACIL 的核心引擎

```python
class RecursiveLinear(AnalyticLinear):
    def __init__(self, in_features, gamma, ...):
        super().__init__(in_features, gamma, ...)
        # R: 正则化特征自相关矩阵的逆，初始化为 I/γ
        R = torch.eye(in_features) / gamma
        self.register_buffer("R", R)

    def fit(self, X, Y):
        # 动态扩展权重矩阵（处理新类）
        if num_targets > self.out_features:
            tail = torch.zeros((in_features, increment_size))
            self.weight = torch.cat((self.weight, tail), dim=1)

        # 核心三步更新
        K = torch.inverse(I + X @ self.R @ X.T)           # (1) 核矩阵求逆
        self.R -= self.R @ X.T @ K @ X @ self.R           # (2) Woodbury 更新
        self.weight += self.R @ X.T @ (Y - X @ self.weight) # (3) 权重校正
```

**数学正确性**：以上三步等价于对历史所有 batch 的并集做一次岭回归。

#### `GeneralizedARM` — 解析式重加权模块（AIR 用）

```python
class GeneralizedARM(AnalyticLinear):
    def __init__(self, ...):
        # A: 每类的特征自相关矩阵 (num_classes, in_features, in_features)
        # C: 累积互协方差矩阵 (in_features, num_classes)
        # cnt: 每个类别的样本计数

    def fit(self, X, y):
        # 按类别累积统计量
        self.C += X.T @ Y           # 累积互协方差
        for i in range(num_targets):
            self.A[i] += X[y==i].T @ X[y==i]  # 每类自相关矩阵
        self.cnt[y_labels] += label_cnt       # 类别计数

    def update(self):
        # 根据类别频率进行逆频率加权，然后解闭式
        cnt_inv = 1 / self.cnt       # 逆频率权重
        weighted_A = sum(cnt_inv[i] * A[i])  # 加权自相关
        A = weighted_A + gamma * I
        C = C * cnt_inv               # 加权互协方差
        self.weight = inverse(A) @ C  # 闭式解
```

**目的**：通过逆频率加权，消除类别不平衡对分类器的影响。

### 6.4 `Buffer.py` — 特征扩展层

#### `RandomBuffer` — 随机投影缓冲

```python
class RandomBuffer(torch.nn.Linear, Buffer):
    def __init__(self, in_features, out_features, activation=torch.relu_):
        # 权重随机初始化，注册为 buffer → 不参与梯度
        W = torch.empty((out_features, in_features))
        self.register_buffer("weight", W)
        self.activation = activation if activation else nn.Identity()

    def forward(self, X):
        X = X.to(self.weight)
        return self.activation(super().forward(X))  # ReLU(W @ X)
```

**参数**：
- `in_features`: backbone 特征维度（如 768）
- `out_features`: 随机投影目标维度（`buffer_size`，如 2048）
- `activation`: 默认 ReLU

**为什么要随机投影**：
1. 升维 → 增强类间线性可分性
2. ReLU 非线性 → 引入非线性变换
3. 随机固定 → 保持解析解的闭合性（可微但不需要可微）

#### `GaussianKernel` — 高斯核缓冲（GKEAL 用）

```python
class GaussianKernel(Buffer):
    def __init__(self, mean, sigma=1):
        # mean: 从训练数据中采样的中心向量
        # beta = 1 / (2σ²)
        self.register_buffer("mean", mean)
        self.register_buffer("beta", 1/(2*sigma²))

    def forward(self, X):
        # 计算每个样本到每个中心的 RBF 相似度
        X = torch.cdist(X, self.mean)       # 欧氏距离
        X = X.square_().mul_(-self.beta)    # -||x-μ||²/(2σ²)
        return torch.exp_(X)                 # exp(-||x-μ||²/(2σ²))
```

### 6.5 `ACIL.py` — ACIL/G-ACIL 实现

详见 [ACIL 类详解对话](#)。核心架构：

```
输入 X → backbone → RandomBuffer → RecursiveLinear → logits
         (冻结)     (随机投影+ReLU)  (递归最小二乘)
```

### 6.6 `DSAL.py` — DS-AL 双流解析学习

```python
class DSAL(torch.nn.Module):
    def __init__(self, ...):
        self.buffer = RandomBuffer(...)
        self.activation_main = torch.relu_
        self.activation_comp = torch.tanh_     # 不同的激活函数
        self.linear_main = RecursiveLinear(...)  # gamma_main
        self.linear_comp = RecursiveLinear(...)  # gamma_comp
        self.compensation_ratio = C              # 补偿流权重

    def forward(self, X):
        X = self.buffer(self.backbone(X))
        main = self.linear_main(self.activation_main(X))
        comp = self.linear_comp(self.activation_comp(X))
        return main + self.compensation_ratio * comp
```

**设计思想**：
- **主流（Main Stream）**：ReLU + RecursiveLinear，负责主要分类
- **补偿流（Compensation Stream）**：Tanh + RecursiveLinear，负责修正残差
- **过往标签清洗（PLC）**：补偿流训练时，旧类的残差目标被置零，使补偿流专注于新类

### 6.7 `GKEAL.py` — 高斯核嵌入解析学习

将 `RandomBuffer` 替换为 `GaussianKernel`。在首次 `learn()` 时：
1. 提取所有 backbone 特征
2. 初始化高斯核中心（从训练数据中采样或生成）
3. 后续增量学习与 ACIL 相同

**适用场景**：小样本增量学习（Few-Shot CIL），高斯核提供了更强的非线性表达能力。

### 6.8 `AEFOCL.py` — 在线不平衡增量学习

```python
class AEFOCL(ACIL):
    def __init__(self, ...):
        self.register_buffer("ex", ...)   # 每类特征均值 (原型)
        self.register_buffer("ex2", ...)  # 每类特征平方均值 (用于计算方差)
        self.register_buffer("cnt", ...)  # 每类样本计数
        self.noise = 0.05                 # 合成数据的噪声水平

    def update(self):
        # 对每个少数类:
        #   1. 估计分布 N(mean, noise*std)
        #   2. 从分布中采样合成特征
        #   3. 用合成特征重新 fit，平衡各类别样本数
```

**目的**：在在线（单样本/小批次）增量学习场景下，通过原型增强克服类别不平衡。

### 6.9 `AIR.py` — 解析式不平衡校正器

使用 `GeneralizedARM` 替代 `RecursiveLinear`。通过按类别累积统计量并在最后做逆频率加权，消除类别不平衡。

---

## 7. 数据集层：datasets/ 详解

### 7.1 `DatasetWrapper.py` — 多阶段划分核心

```python
class DatasetWrapper(Dataset, metaclass=ABCMeta):
    def __init__(self, labels, base_ratio, num_phases, augment, inplace_repeat, shuffle_seed):
        self.num_classes = ...                  # 总类别数
        self.base_size = int(num_classes * base_ratio)  # 基类数
        self.incremental_size = num_classes - base_size # 增量类总数
        self.phase_size = incremental_size // num_phases # 每阶段新增类数

        # 按类别分组索引
        self.class_indices: list[list[int]] = [[] for _ in range(num_classes)]
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)

        # 可选：打乱类别顺序
        if shuffle_seed is not None:
            Random(shuffle_seed).shuffle(self.real_labels)
            Random(shuffle_seed).shuffle(self.class_indices)
```

**三个关键方法**：

```python
def _subset(self, label_begin, label_end):
    # 取类别 [label_begin, label_end) 的所有样本索引
    sub_ids = chain.from_iterable(self.class_indices[label_begin:label_end])
    return Subset(self, sub_ids)

def subset_at_phase(self, phase):
    # phase=0 → 基类 (0 ~ base_size-1)
    # phase=k → 第 k 批增量类
    if phase == 0:
        return self._subset(0, self.base_size)
    return self._subset(
        self.base_size + (phase-1) * self.phase_size,
        self.base_size + phase * self.phase_size,
    )

def subset_until_phase(self, phase):
    # 所有已见的类 (0 ~ base_size + phase*phase_size - 1)
    return self._subset(0, self.base_size + phase * self.phase_size)
```

**UCMerced (21类) 划分示例**：
| Phase | `subset_at_phase` (训练) | `subset_until_phase` (测试) |
|-------|--------------------------|----------------------------|
| 0 | 类 0-10 (11类) | 类 0-10 (11类) |
| 1 | 类 11 (1类) | 类 0-11 (12类) |
| 2 | 类 12 (1类) | 类 0-12 (13类) |
| 10 | 类 20 (1类) | 类 0-20 (21类) |

### 7.2 各数据集包装器

| 文件 | 类 | 类别数 | 数据增强 |
|------|------|------|------|
| `MNIST.py` | `MNIST_` | 10 | 无 (与 basic_transform 相同) |
| `CIFAR.py` | `CIFAR10_` | 10 | RandomCrop+Flip+TrivialAugmentWide |
| `CIFAR.py` | `CIFAR100_` | 100 | RandomCrop+Flip+TrivialAugmentWide |
| `ImageNet.py` | `ImageNet_` | 1000 | RandomResizedCrop(176)+Flip |
| `UCMerced.py` | `UCMerced_LandUse_` | 21 | RandomResizedCrop(384)+Flip+TrivialAugmentWide |

### 7.3 `Features.py` — 缓存特征数据集

当启用 `--cache-features` 时使用。从 `.pt` 文件加载预先提取的 backbone 特征，跳过图像加载和 backbone 推理。

```python
class Features(DatasetWrapper):
    def __init__(self, root, train, ...):
        X = torch.load("X_train.pt" if train else "X_test.pt")  # 加载缓存
        y = torch.load("y_train.pt" if train else "y_test.pt")
        self.dataset = TensorDataset(X, y)                       # 包装为 TensorDataset
        super().__init__(labels, base_ratio, num_phases, False, ...)  # 复用划分逻辑
```

---

## 8. 模型层：models/ 详解

### 8.1 `__init__.py` — Backbone 注册表

```python
models: Dict[str, Tuple[int, Callable, Optional[WeightsEnum]]] = {
    # 格式: "名称": (输入尺寸, 构造函数, 预训练权重)

    # CIFAR ResNet
    "resnet32":   (32, resnet32,   None),
    "resnet110":  (32, resnet110,  None),

    # ImageNet ResNet
    "resnet18":   (224, resnet18,  ResNet18_Weights.DEFAULT),
    "resnet50":   (224, resnet50,  ResNet50_Weights.DEFAULT),

    # Vision Transformer
    "vit_b_16":   (384, vit_b_16,  ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1),
    "vit_l_16":   (512, vit_l_16,  ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1),
    "vit_h_14":   (518, vit_h_14,  ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1),

    # 无 backbone（展平输入，用于 MNIST）
    "Flatten":    (28, Flatten,    None),
}
```

**`load_backbone()` 函数**：
```python
def load_backbone(name, pretrain=False):
    input_img_size, model_fn, weights = models[name]

    # pretrain=True 时加载预训练权重
    if pretrain and weights is not None:
        kwargs["weights"] = weights

    backbone = model_fn(**kwargs)

    # 切掉分类头，替换为 Identity
    if isinstance(backbone, VisionTransformer):
        feature_size = backbone.heads[-1].in_features  # 768/1024/1280
        backbone.heads = nn.Identity()
    elif isinstance(backbone, (ResNet, CifarResNet)):
        feature_size = backbone.fc.in_features         # 64/128/256/512/2048
        backbone.fc = nn.Identity()

    return backbone, input_img_size, feature_size
```

### 8.2 `CifarResNet.py` — CIFAR 专用 ResNet

标准实现（基于 Yerlan Idelbayev 版本），专为 32×32 输入设计：
- 3 个 stage，输出通道 [16, 32, 64]
- 使用 Option A shortcut（零填充降采样）
- 第一层是 `3×3 conv`（非 7×7），无 maxpool

---

## 9. 工具层：utils/ 详解

### 9.1 `validate.py` — 验证循环

```python
@torch.no_grad()
def validate(model, data_loader, num_classes, desc=None):
    model.eval()
    meter = ClassificationMeter(num_classes)

    for X, y in tqdm(data_loader, desc=desc):
        X = X.to(device)
        y = y.to(device)
        logits = model(X)           # 前向推理
        meter.record(y, logits)     # 累积预测

    return meter  # 包含所有聚合指标
```

### 9.2 `metrics.py` — 分类指标

```python
class ClassificationMeter:
    def record(self, y_true, logits):
        # 累积预测概率和标签
        self.probas.append(F.softmax(logits, dim=-1))
        self.targets.append(y_true)

    # 属性（延迟计算）：
    @property
    def accuracy(self): ...    # Top-1 准确率
    @property
    def accuracy5(self): ...   # Top-5 准确率
    @property
    def balanced_accuracy(self): ...   # 平衡准确率
    @property
    def f1_micro(self): ...    # Micro F1 分数
    @property
    def f1_macro(self): ...    # Macro F1 分数
    @property
    def loss(self): ...        # 交叉熵损失
```

### 9.3 `set_weight_decay.py` — 分层权重衰减

```python
def set_weight_decay(model, weight_decay):
    # 将参数分为两组：
    # - norm/bias/position_embedding/class_token: weight_decay=0
    # - 其他参数: weight_decay=指定值
    return [
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": decay_params,    "weight_decay": weight_decay},
    ]
```

**目的**：对归一化层和偏置不施加 weight decay，这是现代训练的标准做法。

### 9.4 `set_determinism.py` — 五层随机种子锁定

```python
def set_determinism(seed: int) -> None:
    # 第1层：cuBLAS 工作空间
    environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # → 设置 cuBLAS 使用确定性工作空间大小（4096KB），
    #   避免因工作空间不足而回退到非确定性算法

    # 第2层：PyTorch 全局
    torch.use_deterministic_algorithms(True)
    # → 强制所有 CUDA 操作使用确定性实现
    #   影响：torch.bmm, torch.index_add, torch.scatter 等
    #   副作用：可能稍慢，某些操作会报错（如 torch.Tensor.normal_）

    # 第3层：Python 生态
    torch.manual_seed(seed)      # PyTorch CPU 随机数
    numpy.random.seed(seed)      # NumPy 随机数
    random.seed(seed)            # Python 标准库随机数

    # 第4层：CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        # → 关闭 cuDNN auto-tuner（它会测试多种算法选最快的，
        #   每次运行选择的算法可能不同 → 非确定性）

        torch.backends.cudnn.deterministic = True
        # → 强制 cuDNN 使用确定性卷积算法

    # 第5层：多 GPU
        torch.cuda.manual_seed_all(seed)
        # → 为所有 GPU 设备设置种子
```

**确定性层级图**：
```
set_determinism(42)
    │
    ├─ 环境变量 ─── CUBLAS_WORKSPACE_CONFIG=":4096:8"
    ├─ PyTorch ──── use_deterministic_algorithms(True)
    ├─ Python ───── random.seed(42)
    ├─ NumPy ────── numpy.random.seed(42)
    ├─ PyTorch CPU ─ torch.manual_seed(42)
    └─ CUDA ─────── cudnn.benchmark=False
                    cudnn.deterministic=True
                    cuda.manual_seed_all(42)
```

---

## 10. 数据流全景图

```
                                CLI 参数
                                   │
                          ┌────────▼────────┐
                          │  config.py       │
                          │  load_args()     │
                          │  → Dict args     │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  main.py:main() │
                          └────────┬────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
   ┌──────▼──────┐         ┌──────▼──────┐         ┌──────▼──────┐
   │ 设备选择     │         │ backbone    │         │ 数据集       │
   │ cuda:0      │         │ vit_b_16    │         │ UCMerced     │
   │ all_gpus=None│        │ → 768 维    │         │ 21类 1680/420│
   └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                          ┌────────▼────────┐
                          │  Learner 创建    │
                          │  ACILLearner(   │
                          │    backbone,    │
                          │    buffer=2048, │
                          │    gamma=0.1    │
                          │  )              │
                          └────────┬────────┘
                                   │
                    ╔══════════════╩══════════════╗
                    ║   Base Training (SGD)        ║
                    ║   ┌──────────────────────┐  ║
                    ║   │ ViT + Linear(768→11) │  ║
                    ║   │ CE Loss + SGD        │  ║
                    ║   │ 300 epochs           │  ║
                    ║   │ → backbone.pth       │  ║
                    ║   └──────────────────────┘  ║
                    ╚══════════════╦══════════════╝
                                   │
                    ╔══════════════╩══════════════╗
                    ║   Cache Features              ║
                    ║   ┌──────────────────────┐  ║
                    ║   │ ViT(全量) → .pt 文件  │  ║
                    ║   │ backbone=Identity()  │  ║
                    ║   └──────────────────────┘  ║
                    ╚══════════════╦══════════════╝
                                   │
         ╔═════════════════════════╩═════════════════════════╗
         ║   Incremental Learning (11 phases)                ║
         ║                                                   ║
         ║   Phase 0:  X(类0-10) → Buffer → RecursiveLinear  ║
         ║             weight (2048,0)→(2048,11)             ║
         ║             R 矩阵编码11类信息                     ║
         ║                     │                              ║
         ║   Phase 1:  X(类11) → Buffer → RecursiveLinear    ║
         ║             weight (2048,11)→(2048,12)            ║
         ║             Woodbury 更新 R (不遗忘旧类)          ║
         ║                     │                              ║
         ║   Phase 2:  X(类12) → ... → weight (2048,13)     ║
         ║   ...                                             ║
         ║   Phase 10: X(类20) → ... → weight (2048,21)     ║
         ║                                                   ║
         ║   每次验证: 在所有已见类上评估                     ║
         ╚══════════════════════════╦════════════════════════╝
                                    │
                           ┌────────▼────────┐
                           │  IL.csv          │
                           │  acc@1/acc@5/   │
                           │  F1/loss         │
                           └─────────────────┘
```

---

## 11. 各方法对比总结

| 特性 | ACIL | G-ACIL | DS-AL | GKEAL | AEF-OCL | AIR |
|------|------|--------|-------|-------|---------|-----|
| **论文发表** | NeurIPS '22 | arXiv '24 | AAAI '24 | CVPR '23 | arXiv '24 | arXiv '24 |
| **CLI 名称** | `ACIL` | `G-ACIL` | `DS-AL` | `GKEAL` | `AEF-OCL` | `AIR` |
| **Learner 类** | `ACILLearner` | `ACILLearner` | `DSALLearner` | `GKEALLearner` | `AEFOCLLearner` | `AIRLearner` |
| **线性引擎** | `RecursiveLinear` | `RecursiveLinear` | 2×`RecursiveLinear` | `RecursiveLinear` | `RecursiveLinear` | `GeneralizedARM` |
| **缓冲层** | `RandomBuffer` | `RandomBuffer` | `RandomBuffer` | `GaussianKernel` | `RandomBuffer` | `RandomBuffer` |
| **激活函数** | ReLU | ReLU | ReLU + Tanh | RBF | ReLU | ReLU |
| **核心创新** | 递归最小二乘 CIL | mini-batch 支持 | 双流+PLC 残差补偿 | 高斯核嵌入 | 原型增强+不平衡 | 逆频率加权 |
| **适用场景** | 标准 CIL | 泛化 CIL | 标准 CIL | 小样本 CIL | 在线+不平衡 | 不平衡 CIL |
| **Exemplar-Free** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **隐私保护** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### 算法选择指南

```
你的任务是什么？

├── 标准类别增量学习
│   ├── 批量数据 → ACIL
│   └── 小批次数据 → G-ACIL
│
├── 需要更高准确率 → DS-AL（双流补偿）
│
├── 小样本增量学习（每类只有少量样本）→ GKEAL
│
├── 在线学习（逐样本到达）→ AEF-OCL
│
└── 类别不平衡严重 → AIR / G-AIR
```

---

## 附录：关键文件位置索引

| 功能 | 文件 | 行号 |
|------|------|------|
| 程序入口 | `main.py` | 262-263 |
| main 函数 | `main.py` | 69-259 |
| 设备选择 | `main.py` | 73-82 |
| backbone 加载 | `main.py` | 88-103 |
| 数据集加载 | `main.py` | 105-113 |
| Learner 创建 | `main.py` | 116-119 |
| Base Training | `main.py` | 122-143 |
| Cache Features | `main.py` | 146-200 |
| Incremental Learning | `main.py` | 210-258 |
| 参数解析 | `config.py` | 249-279 |
| ALL_METHODS | `config.py` | 21-29 |
| ACIL 模型 | `analytic/ACIL.py` | 28-65 |
| ACILLearner | `analytic/ACIL.py` | 67-252 |
| RecursiveLinear | `analytic/AnalyticLinear.py` | 83-128 |
| GeneralizedARM | `analytic/AnalyticLinear.py` | 131-211 |
| RandomBuffer | `analytic/Buffer.py` | 37-68 |
| GaussianKernel | `analytic/Buffer.py` | 71-102 |
| DatasetWrapper | `datasets/DatasetWrapper.py` | 15-70 |
| Features (缓存) | `datasets/Features.py` | 10-44 |
| load_backbone | `models/__init__.py` | 102-118 |
| set_determinism | `utils/set_determinism.py` | 9-18 |
| validate | `utils/validate.py` | 8-29 |
| ClassificationMeter | `utils/metrics.py` | — |
| make_dataloader | `main.py` | 15-41 |
| cache_features | `main.py` | 52-66 |
