# 手搓 ViT

这是我学习 Vision Transformer（ViT）时整理的中文笔记和练习代码，主要参考 Daniel Bourke 的 [Learn PyTorch 第 08 章：PyTorch Paper Replicating](https://www.learnpytorch.io/08_pytorch_paper_replicating/)。**这是个人学习记录，并非原创教程或原创模型。** 教程思路、示例和配图来自原教程及其引用资料，我在学习过程中进行了中文整理、代码练习和模块拆分。

原始资料：

- [Learn PyTorch 原教程](https://www.learnpytorch.io/08_pytorch_paper_replicating/)
- [原教程代码仓库：mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning)
- [ViT 论文：An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

## 从哪里开始

建议先阅读 **[main.ipynb](main.ipynb)**。为了便于理解，数据准备、patch embedding、class token、position embedding、注意力模块、Transformer Encoder、ViT 组装、训练和单张图片预测，都按学习顺序集中放在这个 Notebook 中。

Notebook 仍会导入本仓库的 `utils.py`、`dataset.py` 和 `train.py`，因此需要保留完整项目。它包含两个教学实验：先从零训练手写 ViT，再使用 torchvision 预训练 ViT 进行迁移学习。

模块化脚本整理了同一套学习内容。**当前 `train.py` 直接使用 `vit.py` 中的手写 ViT，加载 torchvision 提供的 ImageNet 预训练主干权重，再训练当前任务的分类头。** 这让手写结构也能使用预训练参数，而不必从随机权重开始训练。

这些小数据集实验用于理解模型与训练流程，不等同于复现论文的大规模训练结果。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `main.ipynb` | 分步骤讲解与实验 |
| `vit.py` | 手写 ViT，以及 torchvision 预训练权重到手写模型的映射与校验 |
| `dataset.py` | 图片数据集、RGB 转换、统一类别编号和 DataLoader |
| `train.py` | 通用训练／评估函数；手写 ViT 加载预训练权重的训练入口 |
| `test.py` | 加载手写 ViT 的任务权重，预测并显示一张图片 |
| `utils.py` | 配置读取、可靠下载、随机种子、计时与训练曲线 |
| `config.toml` | 模型结构、训练参数、数据目录和预测图片配置 |
| `network.py` | 早期 FashionMNIST 练习模型，不属于 ViT 主流程 |
| `losses.py` | 空的预留文件，当前未使用 |
| `PROJECT_REVIEW.md` | 问题修复状态、验证结果与保留事项 |

## 阅读 Notebook

在项目根目录执行：

```bash
python -m jupyterlab main.ipynb
```

选择安装了依赖的内核，按顺序执行单元格。后面的代码依赖前面创建的模型和变量。代码修正后，相关单元格的旧输出已经清空；保留下来的设备、路径和下载日志属于历史环境记录。

Notebook 保留从零训练手写 ViT 和 torchvision 迁移学习两个实验，方便对照理解。手写 ViT 默认规模较大，训练建议使用 GPU。

## 环境准备

使用 Python 3.11 或以上版本。安装相互兼容的 PyTorch 和 torchvision；如需 GPU 加速，使用适合本机环境的 PyTorch 安装包。

```bash
python -m pip install torch torchvision matplotlib pillow requests tqdm torchinfo jupyterlab
```

依赖尚未锁定版本。首次训练需要联网下载数据集和预训练权重；有缓存时可以复用。数据、训练权重和临时文件不随仓库上传。


## 用预训练权重训练手写 ViT

在项目根目录执行：

```bash
python train.py
```

训练过程依次为：

1. 读取 `config.toml`，准备图片数据和预训练权重对应的预处理。
2. 创建 `vit.ViT` 手写模型。
3. 将 torchvision ViT-B/16 的主干权重映射到手写模型，完整检查参数名称、层数和形状。
4. 保留为当前类别数新建的分类头。默认冻结主干，只训练分类头；也可以选择微调整个模型。
5. 每轮评估，在准确率提高时保存 `best.pt`，训练结束后显示曲线。

核心调用位于 `train.py`：

```python
model = build_model(config["model"], num_classes=len(class_names))
model.load_pretrained_weights(weights.get_state_dict(progress=True, check_hash=True))
```

映射覆盖 patch 投影、class token、position embedding、各层注意力、MLP 和最后的 LayerNorm，兼容官方权重的新旧 MLP 参数名称。ImageNet 的原始 1000 类分类头不会覆盖新的任务分类头。

预训练权重要求主干结构与 ViT-B/16 一致：输入 224×224、patch 为 16、embedding 维度 768、12 层、12 个注意力头、MLP 维度 3072。入口会拒绝不兼容的结构配置。手写类本身仍可用于构建其他尺寸的实验模型。

默认配置为 10 轮、batch size 32、学习率 `1e-3`。默认权重保存在：

```text
runs/handwritten_pretrained_vit/best.pt
```

## 预测图片

完成训练后执行：

```bash
python test.py
```

脚本从配置指定的训练目录加载 `best.pt`，对 `test.image_path` 指定的图片进行预测。默认使用 `04-pizza-dad.jpeg`，不存在或损坏时会按 `test.image_url` 重新下载。它使用固定图片，不是随机抽取测试集。

训练入口、预测入口和 Notebook 最后的预训练模型预测，都使用权重配套的预处理：短边缩放到 256，再中心裁剪为 224×224，随后进行归一化。图片读取时统一转换为 RGB，预测显式使用选定的设备。

预测时沿用训练数据的类别编号，因此类别目录及其名称需要与训练时保持一致。

## 修改配置

训练、预测和数据集示例都读取 `config.toml`。相对路径以配置文件所在目录为基准解析。

| 配置项 | 作用 |
| --- | --- |
| `model` | 手写模型的结构与 dropout；预训练主干的结构参数需符合 ViT-B/16 |
| `train.batch_size`、`train.num_workers` | 数据加载参数 |
| `train.epochs`、`train.learning_rate` | 训练轮次与学习率 |
| `train.seed` | 随机种子 |
| `train.device` | `auto` 自动选择 CUDA／CPU，也可指定 `cpu` 或 `cuda` |
| `train.freeze_backbone` | `true` 只训练分类头；`false` 微调整个模型 |
| `train.save_dir` | 模型权重保存目录，预测也从这里读取 |
| `data.data_path`、`data.dataset_name` | 数据根目录与数据集目录名 |
| `data.source` | 数据集 ZIP 下载地址 |
| `test.image_path`、`test.image_url` | 预测图片路径及缺失时的下载地址 |

也可以选择另一份配置文件：

```bash
python train.py --config path/to/config.toml
python test.py --config path/to/config.toml
```

## 三个实验的权重目录

| 实验 | 默认目录 | 模型 |
| --- | --- | --- |
| Notebook：从零训练 | `runs/custom_vit/` | Notebook 中定义的手写 ViT |
| Notebook：迁移学习 | `runs/pretrained_vit/` | torchvision ViT |
| 模块化训练入口 | `runs/handwritten_pretrained_vit/` | `vit.py` 中的手写 ViT，加载预训练主干 |

三个目录分开保存，避免不同实验的权重相互覆盖。手写模型和 torchvision 模型的参数名称不同，不能直接把另一个实验的文件交给 `load_state_dict()`。旧版 `runs/best.pt` 也不会被当前预测入口自动使用。

Notebook 最后会加载其对应实验的最佳权重进行预测。比较不同入口结果时，需要先对齐模型、权重、图片、预处理和设备。


## 注意事项 : 全程零准确率时不写入新的最佳权重

`train()` 中 `best_acc=0`、仅在准确率提高时保存的行为保留。

如果所有轮次准确率均为零，没有新的 `best.pt`；若目录里已有旧文件，它仍会保留。后续可将初值设为 `float("-inf")`，保证首轮写入。