# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reproduction of **Swin-Unet** (ECCVW 2022): a pure Transformer U-Net for medical image segmentation using Swin Transformer as both encoder and decoder. The model processes 2D slices of 3D CT/MRI volumes.

## Environment & Commands

**Conda:** `E:\IT\anaconda3` | **Env:** `swin_unet` (Python 3.7, PyTorch 1.13.1+cu117)  
**GPU:** RTX 4060 Laptop (8GB) → batch_size must be ≤12

```bash
# Activate
conda activate swin_unet

# Train (quick verification: reduce --max_epochs to 5)
python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
    --root_path data/Synapse \
    --max_epochs 150 --output_dir ./model_out/Synapse --img_size 224 \
    --base_lr 0.05 --batch_size 12 --n_class 9 --list_dir ./lists/Synapse --num_workers 0

# Test (uses best_model.pth from output_dir)
python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
    --root_path data/Synapse \
    --output_dir ./model_out/Synapse --list_dir ./lists/Synapse \
    --n_class 9 --img_size 224 --split_name test_vol

# TensorBoard
tensorboard --logdir=./model_out/Synapse --port=6006
```

## Architecture (big picture)

**Model** (`networks/vision_transformer.py` → `SwinUnet`):
- Wraps `SwinTransformerSys` (the actual encoder-decoder)
- If input has 1 channel (grayscale medical), repeats to 3 channels in forward()
- `load_from()`: pretrained Swin-T weights → maps encoder `layers.0-3` to both encoder and decoder `layers_up.3-0` (decoder uses mirrored pretrained weights)

**SwinTransformerSys** (`networks/swin_transformer_unet_skip_expand_decoder_sys.py`):
- Encoder: standard Swin-T with patch embedding + 4 stages of Swin blocks + patch merging
- Decoder: 4 stages of patch expanding + Swin blocks + skip connections from encoder
- Bottleneck: 2 Swin blocks between encoder/decoder
- Final output: patch expanding → linear projection → `num_classes` channels

**Config** (`config.py`): yacs-based, loaded from YAML + CLI overrides. Key overrides: `batch_size`, `resume`, `amp_opt_level`, `use_checkpoint`.

**Training** (`trainer.py`):
- Optimizer: SGD with momentum (hardcoded, NOT from config)
- Loss: `0.4 * CrossEntropyLoss + 0.6 * DiceLoss`
- LR schedule: polynomial decay `base_lr * (1 - iter/max_iter) ** 0.9`
- LR scales linearly with batch_size when `batch_size % 6 == 0`
- Train/val split: val_loader uses `db_train` (same as training data, per paper — Synapse has no dedicated validation set; 18 cases all for training)

**Data** (`datasets/dataset_synapse.py`):
- Training: `.npz` files with `image`/`label` keys; `RandomGenerator` does rot/flip/zoom → 224×224
- Testing: `.npy.h5` files (3D volumes) loaded via h5py; inference loops over slices
- List files in `lists/Synapse/` with one sample per line

**Metrics** (`utils.py`):
- `DiceLoss`: multi-class soft Dice loss
- `test_single_volume`: slice-by-slice inference for 3D volumes, returns per-class (Dice, HD95) tuples using `medpy.metric.binary`

## Known Issues & Fixes Applied

1. **Windows multiprocessing**: `num_workers=0` required; local `worker_init_fn` can't be pickled with spawn
2. **`test.py` bug**: `args.volume_path` referenced before assignment → fixed to use `args.root_path`
3. **`test.py` path separator**: `.split('/')[-1]` fails on Windows → fixed to `os.path.basename()`
4. **`utils.py` squeeze bug**: double `.squeeze(0)` fails on 3D volumes → conditional squeeze
5. **val loader by design**: Original paper uses all 18 Synapse cases for training, no validation split. `val_loader = DataLoader(db_train, ...)` is intentional — a training-progress monitor, not a real validation. This matches the paper's setup.
6. **list_dir override**: `dataset_config` hardcoded `./lists/{dataset}` ignoring CLI `--list_dir` → removed `dataset_config` dicts in `train.py` and `test.py`, CLI args now take effect directly
7. **optimizer/config mismatch**: `config.py` defaults to AdamW/cosine LR but `trainer.py` hardcodes SGD/polynomial LR — actual training uses SGD (matches paper), but config is misleading

## Shell Scripts

All shell scripts accept environment variable overrides instead of CLI args:

| Script | Env vars | Defaults | Purpose |
|--------|----------|----------|---------|
| `train.sh` | `$epoch_time`, `$out_dir`, `$cfg`, `$data_dir`, `$learning_rate`, `$img_size`, `$batch_size` | 150, `./model_out/Synapse`, default cfg, default data, 0.05, 224, 12 | Quick training launch |
| `test.sh` | `$out_dir`, `$cfg`, `$data_dir`, `$img_size` | `./model_out/Synapse`, default cfg, default data, 224 | Quick testing launch |
| `generate_figures.sh` | `$out_dir`, `$cfg`, `$data_dir`, `$img_size`, `$n_class`, `$num_slices` | `./model_out/Synapse`, default cfg, default data, 224, 9, 3 | Generate paper result figures |

Example: `out_dir=./model_out/ACDC n_class=4 sh test.sh`

## Visualization & Analysis

```bash
# Plot training curves from TensorBoard logs (6 subplots: train/val CE+Dice+total loss)
python plot_training_curves.py --logdir ./model_out/Synapse/log

# Generate paper-style figures (per-case overlays, summary bar chart, CSV)
# ~15 minutes for 12 test volumes with batched inference; requires best_model.pth to exist
sh generate_figures.sh
# Or directly:
python visualize_results.py \
    --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
    --root_path data/Synapse \
    --output_dir ./model_out/Synapse --list_dir ./lists/Synapse \
    --n_class 9 --img_size 224 --num_slices 3 --split_name test_vol
```

**Visualization outputs** (in `model_out/Synapse/result_figures/`):
- `case00XX.png` — per-case figures: 3 rows (original CT / ground truth overlay / prediction overlay) × N slices
- `summary.png` — per-class Dice + HD95 bar charts with error bars
- `results.csv` — raw per-case per-class metrics
- `training_curves.png` — 2×3 grid of epoch-level loss curves

## Manuscript

The course manuscript lives in `manuscript.md` (Chinese), with compiled `manuscript.docx` and `manuscript.pdf` outputs. Figures referenced by the manuscript are generated by `visualize_results.py` and `plot_training_curves.py`.

## Other Tools

```bash
# Generate data list files from .npz directory
python make_dataset_txt.py --data .npz --name my_dataset
```

## Output Directory Structure

```
model_out/Synapse/
├── best_model.pth              # Best checkpoint
├── last_model.pth              # Final epoch checkpoint
├── log.txt                     # Full training log
├── log/                        # TensorBoard training events
├── log_test/                   # TensorBoard test events
└── result_figures/             # Paper figures (per-case images, summary, CSV, training curves)
```

Archived experiments:
- `model_out/Synapse/` — 150 epochs, 18 cases, paper-consistent results (DSC 0.761)
- `model_out/Synapse_300ep_valsplit.zip` — 300 epochs, 15/3 split exploratory experiment (DSC 0.769), archived for reference

## Data Locations

- Synapse train: `data/Synapse/train_npz/` (2211 .npz)
- Synapse test: `data/Synapse/test_vol_h5/` (12 .npy.h5)
- ACDC: not used (archives kept in `TransUNet_ACDC_code_data/`)
- Pretrained: `pretrained_ckpt/swin_tiny_patch4_window7_224.pth`
