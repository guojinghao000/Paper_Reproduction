#!/usr/bin/env python
"""
Generate paper-style result visualizations for Swin-UNet reproduction.

Outputs:
  1. result_figures/          — Per-case segmentation comparison figures
  2. result_figures/summary.png  — Per-class Dice/HD95 summary table
  3. result_figures/results.csv  — Raw metrics CSV

Usage:
    conda activate swin_unet
    python visualize_results.py \
        --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
        --root_path project_transunet/project_TransUNet/data/Synapse \
        --output_dir ./model_out/Synapse \
        --list_dir ./lists/Synapse \
        --n_class 9 --img_size 224
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from datasets.dataset_synapse import Synapse_dataset
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import test_single_volume

# ── Organ labels for Synapse dataset ──
ORGAN_NAMES = ["Background", "Aorta", "Gallbladder", "Kidney(L)", "Kidney(R)",
               "Liver", "Pancreas", "Spleen", "Stomach"]
ORGAN_COLORS = [
    [0, 0, 0],         # 0: background (black)
    [1, 0.2, 0.2],     # 1: aorta (red)
    [0.2, 1, 0.2],     # 2: gallbladder (green)
    [0.2, 0.2, 1],     # 3: kidney left (blue)
    [0.8, 0.8, 0],     # 4: kidney right (yellow)
    [1, 0.6, 0],       # 5: liver (orange)
    [0.6, 0, 1],       # 6: pancreas (purple)
    [0, 0.8, 0.8],     # 7: spleen (cyan)
    [1, 0.4, 0.8],     # 8: stomach (pink)
]


def make_overlay(image_2d, mask_2d, alpha=0.4):
    """Create RGB overlay of segmentation mask on grayscale image."""
    # Normalize image to [0, 1]
    img = (image_2d - image_2d.min()) / (image_2d.max() - image_2d.min() + 1e-8)
    img_rgb = np.stack([img] * 3, axis=-1)

    overlay = np.zeros_like(img_rgb)
    for cls in range(1, len(ORGAN_COLORS)):
        overlay[mask_2d == cls] = ORGAN_COLORS[cls]

    blended = (1 - alpha) * img_rgb + alpha * overlay
    blended = np.clip(blended, 0, 1)
    return blended


def infer_volume(image, label, model, patch_size, num_classes):
    """Modified from test_single_volume: returns prediction array + metric list."""
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    if image.shape[0] == 1:
        image, label = image.squeeze(0), label.squeeze(0)

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slc = image[ind, :, :]
            x, y = slc.shape[0], slc.shape[1]
            in_ = slc
            if x != patch_size[0] or y != patch_size[1]:
                from scipy.ndimage import zoom
                in_ = zoom(slc, (patch_size[0] / x, patch_size[1] / y), order=3)
            inp = torch.from_numpy(in_).unsqueeze(0).unsqueeze(0).float().cuda()
            model.eval()
            with torch.no_grad():
                out = model(inp)
                out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    from scipy.ndimage import zoom
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        inp = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        model.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(model(inp), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    from utils import calculate_metric_percase
    metric_list = []
    for i in range(1, num_classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return prediction, metric_list


def create_figure(image, label, prediction, case_name, metrics, save_path, num_slices=3):
    """Create a single figure with original / ground-truth / prediction rows."""
    import matplotlib.pyplot as plt

    # Pick evenly spaced slices through the volume
    if len(image.shape) == 3:
        D = image.shape[0]
        indices = np.linspace(D * 0.25, D * 0.75, num_slices, dtype=int)
        indices = np.clip(indices, 0, D - 1)
        slices_img = [image[i] for i in indices]
        slices_gt = [label[i] for i in indices]
        slices_pred = [prediction[i] for i in indices]
    else:
        slices_img = [image]
        slices_gt = [label]
        slices_pred = [prediction]
        num_slices = 1

    fig, axes = plt.subplots(3, num_slices, figsize=(3 * num_slices, 9))

    if num_slices == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    row_labels = ["Original CT", "Ground Truth", "Swin-UNet Prediction"]
    for row in range(3):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=12, fontweight="bold")
        for col in range(num_slices):
            ax = axes[row, col]
            if row == 0:
                img = (slices_img[col] - slices_img[col].min()) / (slices_img[col].max() - slices_img[col].min() + 1e-8)
                ax.imshow(img, cmap="gray")
            elif row == 1:
                ax.imshow(make_overlay(slices_img[col], slices_gt[col]))
            else:
                ax.imshow(make_overlay(slices_img[col], slices_pred[col]))
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Slice {col+1}", fontsize=10)

    # Per-class metrics text
    dice_text = f"{case_name}\nMean Dice: {np.mean([m[0] for m in metrics]):.4f} | Mean HD95: {np.mean([m[1] for m in metrics]):.2f}"
    fig.suptitle(dice_text, fontsize=10, y=0.98)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def create_summary_figure(all_metrics, save_path):
    """Create a summary bar chart of per-class Dice across all cases."""
    import matplotlib.pyplot as plt

    # all_metrics: (num_cases, num_classes-1, 2)
    arr = np.array(all_metrics)  # (C, 8, 2)
    mean_dice = arr[:, :, 0].mean(axis=0)
    std_dice = arr[:, :, 0].std(axis=0)
    mean_hd95 = arr[:, :, 1].mean(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    names = ORGAN_NAMES[1:]  # skip background
    colors = [ORGAN_COLORS[i] for i in range(1, len(ORGAN_COLORS))]
    x = np.arange(len(names))

    # Dice bar chart
    bars = ax1.bar(x, mean_dice, yerr=std_dice, color=colors, capsize=5, edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
    ax1.set_ylabel("Dice Score", fontsize=13)
    ax1.set_title("Per-Class Dice Similarity Coefficient", fontsize=14)
    ax1.set_ylim(0, 1.05)
    for bar, val in zip(bars, mean_dice):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # HD95 bar chart
    bars = ax2.bar(x, mean_hd95, color=colors, capsize=5, edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
    ax2.set_ylabel("HD95 (mm)", fontsize=13)
    ax2.set_title("Per-Class Hausdorff Distance 95%", fontsize=14)
    for bar, val in zip(bars, mean_hd95):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Swin-UNet Synapse Results  |  Mean DSC: {mean_dice.mean():.4f}  |  Mean HD95: {mean_hd95.mean():.2f}",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--root_path', type=str,
                        default='data/Synapse')
    parser.add_argument('--output_dir', type=str, default='./model_out/Synapse')
    parser.add_argument('--list_dir', type=str, default='./lists/Synapse')
    parser.add_argument('--n_class', type=int, default=9)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--split_name', type=str, default='test_vol')
    parser.add_argument('--num_slices', type=int, default=3,
                        help='Number of slices to show per case')
    args = parser.parse_args()

    # Rest of args for get_config
    class FakeArgs:
        pass
    for k, v in vars(args).items():
        setattr(FakeArgs, k, v)
    for attr in ['opts', 'zip', 'cache_mode', 'resume', 'accumulation_steps',
                 'use_checkpoint', 'amp_opt_level', 'tag', 'eval', 'throughput', 'batch_size']:
        if not hasattr(FakeArgs, attr):
            setattr(FakeArgs, attr, None)

    config = get_config(FakeArgs)

    # Volume path
    volume_path = os.path.join(args.root_path, "test_vol_h5")

    # Load model
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.n_class).cuda()
    snapshot = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(snapshot):
        print(f"Error: {snapshot} not found! Train the model first.")
        sys.exit(1)
    net.load_state_dict(torch.load(snapshot))
    print(f"Loaded: {snapshot}")

    # Output directories
    fig_dir = os.path.join(args.output_dir, "result_figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Load test data
    db_test = Synapse_dataset(base_dir=volume_path, split=args.split_name, list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    print(f"{len(testloader)} test volumes to process")

    all_metrics = []
    for i_batch, sampled_batch in tqdm(enumerate(testloader), desc="Inference"):
        image, label = sampled_batch["image"], sampled_batch["label"]
        case_name = sampled_batch['case_name'][0].strip()

        prediction, metrics = infer_volume(
            image, label, net,
            patch_size=[args.img_size, args.img_size],
            num_classes=args.n_class)

        all_metrics.append(metrics)

        # Generate per-case figure
        save_path = os.path.join(fig_dir, f"{case_name}.png")
        create_figure(image.squeeze(0).cpu().numpy(),
                      label.squeeze(0).cpu().numpy(),
                      prediction, case_name, metrics,
                      save_path, num_slices=args.num_slices)

    # Generate summary figure
    summary_path = os.path.join(fig_dir, "summary.png")
    create_summary_figure(all_metrics, summary_path)

    # Save CSV
    import csv
    csv_path = os.path.join(fig_dir, "results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Case'] + [f"{name}_Dice" for name in ORGAN_NAMES[1:]]
                        + [f"{name}_HD95" for name in ORGAN_NAMES[1:]] + ['Mean_Dice', 'Mean_HD95'])
        db_test = Synapse_dataset(base_dir=volume_path, split=args.split_name, list_dir=args.list_dir)
        for i, (metrics, sampled_batch) in enumerate(zip(all_metrics, db_test)):
            case = sampled_batch['case_name'].strip() if isinstance(sampled_batch, dict) else f"case_{i}"
            dice_row = [m[0] for m in metrics]
            hd95_row = [m[1] for m in metrics]
            writer.writerow([case] + dice_row + hd95_row + [np.mean(dice_row), np.mean(hd95_row)])

    print(f"\nDone! Results saved to: {fig_dir}/")
    print(f"  Figures: {len(all_metrics)} cases")
    print(f"  Summary: summary.png")
    print(f"  Metrics: results.csv")


if __name__ == "__main__":
    main()
