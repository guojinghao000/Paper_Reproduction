#!/usr/bin/env python
"""
对比两次训练的 epoch 级损失曲线。
用法:
    conda activate swin_unet
    python compare_training.py
"""

import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np

OLD_LOGDIR = 'Swin-Unet/model_out/Synapse_旧版_验证集bug/log'
NEW_LOGDIR = 'Swin-Unet/model_out/Synapse/log'
OUTPUT = 'Swin-Unet/model_out/Synapse/result_figures/training_comparison.png'

TAGS = [
    'epoch/train_loss', 'epoch/train_loss_ce', 'epoch/train_loss_dice',
    'epoch/val_loss', 'epoch/val_loss_ce', 'epoch/val_loss_dice',
]
TITLES = ['Train Total Loss', 'Train CE Loss', 'Train Dice Loss',
          'Val Total Loss', 'Val CE Loss', 'Val Dice Loss']
OLD_LABEL = 'Old (150ep, 18 cases)'
NEW_LABEL = 'New (300ep, 15+3 split)'


def load(logdir):
    ea = EventAccumulator(logdir)
    ea.Reload()
    d = {}
    for tag in TAGS:
        try:
            e = ea.Scalars(tag)
            d[tag] = ([s.step for s in e], [s.value for s in e])
            print(f'  {tag}: {len(e)} pts')
        except KeyError:
            d[tag] = ([], [])
    return d


def main():
    print('Loading old ...')
    old = load(OLD_LOGDIR)
    print('Loading new ...')
    new = load(NEW_LOGDIR)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, tag in enumerate(TAGS):
        ax = axes[idx // 3][idx % 3]
        os_, ov = old[tag]
        ns, nv = new[tag]
        if os_:
            ax.plot(os_, ov, color='#d62728', lw=1.0, alpha=0.7, label=OLD_LABEL)
        if ns:
            ax.plot(ns, nv, color='#1f77b4', lw=1.0, label=NEW_LABEL)
        if 'val' in tag and nv:
            i = np.argmin(nv)
            ax.scatter(ns[i], nv[i], c='#1f77b4', s=30, zorder=5)
            ax.annotate(f'{nv[i]:.4f} (ep {ns[i]})',
                        xy=(ns[i], nv[i]),
                        xytext=(ns[i] + 15, nv[i] + 0.04),
                        arrowprops=dict(arrowstyle='->', color='#1f77b4'),
                        fontsize=7, color='#1f77b4', fontweight='bold')
        ax.set_title(TITLES[idx], fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('Training Comparison: Old vs New', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {OUTPUT}')


if __name__ == '__main__':
    main()
