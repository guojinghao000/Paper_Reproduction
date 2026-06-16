#!/bin/bash
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
    DATA_DIR='data/Synapse'
fi

if [ $img_size ]; then
    IMG_SIZE=$img_size
else
    IMG_SIZE=224
fi

echo "start test model"
python test.py --dataset Synapse \
    --cfg $CFG --root_path $DATA_DIR \
    --output_dir $OUT_DIR \
    --img_size $IMG_SIZE --n_class 9 \
    --list_dir ./lists/Synapse --split_name test_vol
