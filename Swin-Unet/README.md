# Swin-Unet

[ECCVW 2022] Reproduction of **"Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation"** [[arXiv](https://arxiv.org/abs/2105.05537)]

> Paper accepted by ECCV 2022 Medical Computer Vision Workshop ([MCV](https://mcv-workshop.github.io/))

---

## 项目结构

```
Swin-Unet/
├── train.py                     # 训练入口
├── test.py                      # 测试/推理入口
├── trainer.py                   # 训练循环 + TensorBoard 日志
├── utils.py                     # Dice Loss、测试指标计算
├── config.py                    # 配置管理 (yacs)
├── visualize_results.py         # 论文结果图生成
├── plot_training_curves.py      # 训练曲线绘制
├── make_dataset_txt.py          # 生成数据列表工具
├── train.sh / test.sh           # 便捷启动脚本
├── generate_figures.sh          # 结果图生成脚本
├── requirements.txt             # Python 依赖
├── manuscript.md                # 课程论文手稿 (Markdown)
├── manuscript.docx              # 编译后的论文
├── README.md                    # 本文件
├── CLAUDE.md                    # Claude Code 指导
├── configs/
│   └── swin_tiny_patch4_window7_224_lite.yaml  # 模型配置
├── networks/
│   ├── vision_transformer.py                    # SwinUnet 模型定义
│   └── swin_transformer_unet_skip_expand_decoder_sys.py  # Swin Transformer 骨干
├── datasets/
│   └── dataset_synapse.py       # Synapse 数据集加载
├── lists/
│   └── Synapse/                 # 训练/验证/测试 列表
├── pretrained_ckpt/             # 预训练 Swin-T 权重 (需下载)
│   └── swin_tiny_patch4_window7_224.pth
└── model_out/Synapse/           # 训练输出
    ├── best_model.pth           # 最佳模型检查点
    ├── log/                     # 训练 TensorBoard 日志
    ├── log_test/                # 测试 TensorBoard 日志
    └── result_figures/          # 结果图 (逐案例 + 汇总)
```

---

## 1. 环境配置

### Conda 环境 (推荐)

```bash
conda create -n swin_unet python=3.7 -y
conda activate swin_unet

# PyTorch + CUDA
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# 其他依赖 (SimpleITK 用 conda 预编译版本更快)
conda install -c conda-forge simpleitk -y
pip install -r requirements.txt
```

### Pip 环境

```bash
pip install -r requirements.txt
pip install SimpleITK==2.2.1  # Windows 推荐此版本
```

### 关键依赖版本

| Package | Version |
|:--------|:--------|
| Python | 3.7 |
| PyTorch | 1.13.1+cu117 |
| timm | 0.6.12 |
| einops | 0.6.1 |
| SimpleITK | 2.2.1 |
| medpy | 0.4.0 |

---

## 2. 预训练模型

下载 Swin-T 预训练权重并放入 `pretrained_ckpt/`：

- [Google Drive](https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY)

```bash
mkdir -p pretrained_ckpt
mv swin_tiny_patch4_window7_224.pth pretrained_ckpt/
```

---

## 3. 数据集

使用 TransUNet 作者提供的预处理数据：

| 数据集 | 类别 | 格式 | 链接 |
|:-------|:----:|:----:|:-----|
| Synapse (BTCV) | 9 (8器官+背景) | .npz / .npy.h5 | [Google Drive](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd) |

> 本复现仅使用 Synapse 数据集。ACDC 数据集压缩包保留在 `TransUNet_ACDC_code_data/` 中备查。

### 数据目录结构 (Synapse)

```
data/Synapse/
├── train_npz/                  # 2211 个训练切片 (.npz)
│   ├── case0005_slice000.npz
│   └── ...
└── test_vol_h5/                # 12 个测试体数据 (.npy.h5)
    ├── case0001.npy.h5
    └── ...
```

### 数据列表

`lists/Synapse/` 目录包含数据划分文件（遵循原论文：18 例训练，12 例测试）：

```
lists/Synapse/
├── train.txt                   # 训练样本列表 (18 cases, 2211 slices)
├── val.txt                     # 与 train.txt 相同，仅用于训练进度监控
└── test_vol.txt                # 测试体数据列表 (12 cases)
```

---

## 4. 训练

### 快速启动

```bash
sh train.sh
```

### 完整命令

```bash
python train.py \
    --dataset Synapse \
    --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
    --root_path data/Synapse \
    --max_epochs 150 \
    --output_dir ./model_out/Synapse \
    --img_size 224 \
    --base_lr 0.05 \
    --batch_size 12 \
    --n_class 9 \
    --list_dir ./lists/Synapse \
    --num_workers 0
```

### 参数说明

| 参数 | 默认值 | 说明 |
|:-----|:------|:-----|
| `--root_path` | - | Synapse 数据根目录（代码自动追加 `train_npz`） |
| `--max_epochs` | 150 | 训练轮数 |
| `--batch_size` | 12 | 批大小（8GB 显存推荐 6-12） |
| `--base_lr` | 0.05 | 初始学习率（batch≠24 时自动线性缩放） |
| `--n_class` | 9 | 类别数（Synapse=9, ACDC=4） |
| `--img_size` | 224 | 输入图像尺寸 |
| `--num_workers` | 0 | 数据加载线程（Windows 建议 0） |

### 学习率缩放规则

```python
if batch_size != 24 and batch_size % 6 == 0:
    base_lr *= batch_size / 24
```

### TensorBoard

训练过程中自动记录到 `model_out/Synapse/log/`：

```bash
tensorboard --logdir=./model_out/Synapse --port=6006
```

记录的指标：`train/lr`, `train/total_loss`, `train/loss_ce`, `train/loss_dice`, `epoch/train_loss`, `epoch/val_loss`, `epoch/best_loss`，以及训练图像。

---

## 5. 测试/推理

### 快速启动

```bash
sh test.sh
```

### 完整命令

```bash
python test.py \
    --dataset Synapse \
    --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
    --root_path data/Synapse \
    --output_dir ./model_out/Synapse \
    --list_dir ./lists/Synapse \
    --n_class 9 \
    --img_size 224 \
    --split_name test_vol
```

### TensorBoard 测试日志

测试日志保存在 `model_out/Synapse/log_test/`，包含：

- `test_per_case/dice`, `test_per_case/hd95` — 每个测试体的指标
- `test_per_class/cls_N_dice`, `test_per_class/cls_N_hd95` — 各类别指标
- `test_summary/mean_dice`, `test_summary/mean_hd95` — 总体平均
- 模型计算图 (`SwinUnet` graph)

---

## 6. 评估指标

使用两个标准医学图像分割指标：

- **Dice Similarity Coefficient (DSC)**: 值越高越好（0~1）
- **Hausdorff Distance 95% (HD95)**: 值越低越好

论文在 Synapse 上的结果（150 epochs, Tesla V100）：

| 指标 | Aorta | Gallbladder | Kidney(L) | Kidney(R) | Liver | Pancreas | Spleen | Stomach | **Avg** |
|:-----|:-----:|:-----------:|:---------:|:---------:|:-----:|:--------:|:------:|:-------:|:-------:|
| DSC | 85.47 | 66.53 | 83.81 | 79.61 | 94.29 | 56.58 | 90.66 | 76.60 | **79.13** |
| HD95 | 17.34 | 28.80 | 17.85 | 38.60 | 12.52 | 50.83 | 13.72 | 24.58 | **25.53** |

---

## 7. 其他数据集

> 本复现仅使用 Synapse 数据集。ACDC 数据集原始压缩包保留在 `TransUNet_ACDC_code_data/` 中备查，不做实际训练使用。

### 自定义数据集

使用 `make_dataset_txt.py` 生成数据列表：

```bash
python make_dataset_txt.py --data .npz --name my_dataset
```

---

## 8. 复现注意事项

1. **预训练权重**: 论文对 encoder 和 decoder 均使用 Swin-T 预训练权重初始化，这对纯 Transformer 模型至关重要
2. **GPU 类型**: 不同 GPU 会产生不同结果（论文使用 Tesla V100）。代码固定了随机种子保证同一 GPU 上结果一致
3. **学习率**: 如果结果不理想，尝试调整学习率（`--base_lr`）
4. **Batch Size**: 推荐 24，显存不足可降至 12 或 6，学习率会自动缩放

---

## 9. Windows 平台说明

在 Windows 上运行时需注意：

- `num_workers` 设置为 0（防止 multiprocessing pickle 错误）
- 路径分隔符：代码已修复 `os.path.basename()` 替代 `.split('/')`

---

## References

- [TransUnet](https://github.com/Beckschen/TransUNet)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)

## Citation

```bibtex
@InProceedings{swinunet,
  author = {Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
  title = {Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation},
  booktitle = {Proceedings of the European Conference on Computer Vision Workshops (ECCVW)},
  year = {2022}
}
```
