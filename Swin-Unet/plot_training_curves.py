#!/usr/bin/env python
"""
读取 TensorBoard 训练日志，将全部 epoch 级曲线绘制到一张图上。
用法:
    python plot_training_curves.py --logdir ./model_out/Synapse/log
"""

import os
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np

# ── 7条 epoch 级曲线 ──
TAGS = [
    'epoch/train_loss',
    'epoch/train_loss_ce',
    'epoch/train_loss_dice',
    'epoch/val_loss',
    'epoch/val_loss_ce',
    'epoch/val_loss_dice',
]

TITLES = [
    'epoch/train_loss',
    'epoch/train_loss_ce',
    'epoch/train_loss_dice',
    'epoch/val_loss',
    'epoch/val_loss_ce',
    'epoch/val_loss_dice',
]

YLABELS = [
    'Total Loss', 'CE Loss', 'Dice Loss',
    'Total Loss', 'CE Loss', 'Dice Loss',
]

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='./model_out/Synapse/log')
    parser.add_argument('--output', type=str, default='./model_out/Synapse/result_figures/training_curves.png')
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()

    # 读取 TensorBoard event 文件
    ea = EventAccumulator(args.logdir)
    ea.Reload()

    # 提取每个 tag 的 epoch 数据
    data = {}
    for tag in TAGS:
        try:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[tag] = (steps, values)
            print(f"  {tag}: {len(events)} points")
        except KeyError:
            print(f"  [WARN] {tag} not found in logs, skipping")
            data[tag] = ([], [])

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # ── 绘图: 2行3列 ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    for idx, tag in enumerate(TAGS):
        ax = axes[idx]
        steps, values = data[tag]
        if len(steps) > 0:
            ax.plot(steps, values, color=COLORS[idx], linewidth=1.2)
        ax.set_title(TITLES[idx], fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel(YLABELS[idx], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # ── 标注：在 train_loss 图上标最低点 ──
    if len(data['epoch/train_loss'][0]) > 0:
        steps, values = data['epoch/train_loss']
        min_idx = np.argmin(values)
        axes[0].annotate(f"Min: {values[min_idx]:.4f}\nEpoch {steps[min_idx]}",
                         xy=(steps[min_idx], values[min_idx]),
                         xytext=(steps[min_idx] + 10, values[min_idx] + 0.1),
                         arrowprops=dict(arrowstyle='->', color='red'),
                         fontsize=8, color='red')

    fig.suptitle('Swin-UNet Training Curves (Synapse Dataset)', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()
