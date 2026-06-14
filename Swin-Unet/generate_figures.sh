#!/bin/bash
# ============================================================
# generate_figures.sh
# 生成 Swin-UNet 论文复现所需的全部结果图
# 输出: model_out/Synapse/result_figures/
# ============================================================

if [ $out_dir ]; then
    OUT_DIR=$out_dir
else
    OUT_DIR='./model_out/Synapse'
fi

if [ $cfg ]; then
    CFG=$cfg
else
    CFG='configs/swin_tiny_patch4_window7_224_lite.yaml'
fi

if [ $data_dir ]; then
    DATA_DIR=$data_dir
else
    DATA_DIR='project_transunet/project_TransUNet/data/Synapse'
fi

if [ $img_size ]; then
    IMG_SIZE=$img_size
else
    IMG_SIZE=224
fi

if [ $n_class ]; then
    N_CLASS=$n_class
else
    N_CLASS=9
fi

if [ $num_slices ]; then
    NUM_SLICES=$num_slices
else
    NUM_SLICES=3
fi

echo "============================================"
echo "  Swin-UNet 结果图生成"
echo "============================================"
echo "  Config:      $CFG"
echo "  Data:        $DATA_DIR"
echo "  Output:      $OUT_DIR/result_figures/"
echo "  Class num:   $N_CLASS"
echo "  Slices/case: $NUM_SLICES"
echo "  预计耗时:    ~80 分钟 (12个测试体)"
echo "============================================"
echo ""

python visualize_results.py \
    --cfg $CFG \
    --root_path $DATA_DIR \
    --output_dir $OUT_DIR \
    --list_dir ./lists/Synapse \
    --n_class $N_CLASS \
    --img_size $IMG_SIZE \
    --num_slices $NUM_SLICES \
    --split_name test_vol

echo ""
echo "============================================"
echo "  完成! 结果在: $OUT_DIR/result_figures/"
echo "============================================"
ls -la "$OUT_DIR/result_figures/"
