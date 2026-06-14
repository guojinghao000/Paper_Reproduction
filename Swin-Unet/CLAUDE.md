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
    --root_path project_transunet/project_TransUNet/data/Synapse \
    --max_epochs 150 --output_dir ./model_out/Synapse --img_size 224 \
    --base_lr 0.05 --batch_size 12 --n_class 9 --list_dir ./lists/Synapse --num_workers 0

# Test (uses best_model.pth from output_dir)
python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
    --is_savenii --root_path project_transunet/project_TransUNet/data/Synapse \
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
- Train/val split: same dataset (val_loader uses `db_train`, not `db_val`)

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
5. **list_dir override**: `dataset_config` hardcodes `./lists/{dataset}` regardless of CLI `--list_dir` → created `lists/Synapse/` symlink-style copy
6. **num_classes override**: `dataset_config` sets `num_classes = args.n_class`, so CLI `--n_class` takes priority over `--num_classes`

## Data Locations

- Synapse train: `project_transunet/project_TransUNet/data/Synapse/train_npz/` (2211 .npz)
- Synapse test: `project_transunet/project_TransUNet/data/Synapse/test_vol_h5/` (12 .npy.h5)
- ACDC: `TransUNet_ACDC_code_data/ACDC/` (slices + volumes)
- Pretrained: `pretrained_ckpt/swin_tiny_patch4_window7_224.pth`
