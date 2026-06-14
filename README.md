# 数字图像处理课程论文 —— Swin-Unet 复现

**Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation** (ECCVW 2022) 论文复现项目。

> 课程：数字图像处理 | 论文原文：[arXiv:2105.05537](https://arxiv.org/abs/2105.05537)

---

## 仓库结构

```
.
├── Swin-Unet/                  # 主项目（代码、模型、实验输出）
│   ├── train.py / test.py      # 训练 & 推理入口
│   ├── trainer.py              # 训练循环
│   ├── utils.py                # 损失函数 & 评估指标
│   ├── config.py               # 配置管理
│   ├── networks/               # SwinUnet 网络定义
│   ├── datasets/               # 数据加载器
│   ├── configs/                # YAML 配置文件
│   ├── lists/                  # 数据划分列表
│   ├── model_out/              # 训练输出（权重、日志、预测、结果图）
│   ├── manuscript.md           # 课程论文手稿（Markdown）
│   ├── manuscript.docx / .pdf  # 编译后的论文
│   ├── README.md               # 项目详细文档
│   └── CLAUDE.md               # Claude Code 指导文件
├── 数图.pdf                    # 数字图像处理教材
├── 数图_中文翻译.md             # Swin-Unet 论文中文翻译
├── 名词解释.md                  # 术语解释（U-Net、Transformer 等）
├── Fig1.png / Fig2.png / Fig3.png  # 论文插图
└── README.md                   # 本文件
```

## 论文简介

Swin-Unet 提出了一种**纯 Transformer 的 U 型编码器-解码器网络**用于医学图像分割。与传统的基于 CNN 的 U-Net 不同，Swin-Unet 使用 Swin Transformer 作为编码器和解码器的基本构建块，通过移位窗口自注意力机制在保持线性计算复杂度的同时建模全局上下文信息。

**核心创新点：**
- 首个完全基于 Transformer 的 U 型医学图像分割网络（无 CNN 组件）
- 使用分块合并层（Patch Merging）进行下采样，分块扩张层（Patch Expanding）进行上采样
- 解码器通过镜像映射策略复用编码器的 ImageNet 预训练权重

## 实验结果

在 Synapse 多器官 CT 数据集上训练 150 轮的结果：

| 指标 | Aorta | Gallbladder | Kidney(L) | Kidney(R) | Liver | Pancreas | Spleen | Stomach | **平均** |
|:-----|:-----:|:-----------:|:---------:|:---------:|:-----:|:--------:|:------:|:-------:|:--------:|
| DSC | 0.832 | 0.611 | 0.807 | 0.755 | 0.929 | 0.541 | 0.876 | 0.738 | **0.761** |
| HD95 | 21.97 | 42.11 | 22.41 | 24.83 | 18.95 | 53.77 | 14.28 | 27.10 | **26.93** |

> 原论文 DSC 0.791，HD95 25.53。本复现因 GPU 显存限制（8 GB，批大小 12 vs 原论文 24），结果略低约 3%。

## 快速开始

### 环境配置

```bash
conda create -n swin_unet python=3.7 -y
conda activate swin_unet
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge simpleitk -y
pip install -r Swin-Unet/requirements.txt
```

### 训练

```bash
cd Swin-Unet
sh train.sh
# 或自定义参数
python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
    --root_path project_transunet/project_TransUNet/data/Synapse \
    --max_epochs 150 --output_dir ./model_out/Synapse --img_size 224 \
    --base_lr 0.05 --batch_size 12 --n_class 9 --num_workers 0
```

### 测试 & 生成结果图

```bash
sh test.sh                          # 推理并保存 NIfTI 预测
sh generate_figures.sh              # 生成论文结果图
python plot_training_curves.py      # 绘制训练曲线
tensorboard --logdir=./model_out/Synapse --port=6006  # 查看训练日志
```

更多细节（数据集准备、预训练权重下载、ACDC 数据集训练等）请参阅 [Swin-Unet/README.md](Swin-Unet/README.md)。

## 环境要求

| 组件 | 版本/说明 |
|:-----|:---------|
| Python | 3.7 |
| PyTorch | 1.13.1+cu117 |
| GPU | NVIDIA RTX 4060 Laptop (8 GB) |
| CUDA | 11.7 |
| OS | Windows 11 |

## 参考文献

1. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI, 2015.
2. Chen J, et al. TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation. arXiv, 2021.
3. Liu Z, et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV, 2021.
4. Cao H, et al. Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation. ECCVW, 2022.

```bibtex
@InProceedings{swinunet,
  author = {Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang
            and Xiaopeng Zhang and Qi Tian and Manning Wang},
  title = {Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation},
  booktitle = {Proceedings of the European Conference on Computer Vision Workshops (ECCVW)},
  year = {2022}
}
```
