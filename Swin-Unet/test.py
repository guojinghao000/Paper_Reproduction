from os.path import split
import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from config import get_config
from datasets.dataset_synapse import Synapse_dataset
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data/Synapse/test_vol_h5',
                    help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='datasets', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

parser.add_argument("--n_class", default=4, type=int)
parser.add_argument("--split_name", default="test", help="Directory of the input list")

args = parser.parse_args()

if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.root_path, "test_vol_h5")
else:
    args.volume_path = args.root_path
config = get_config(args)


def inference(args, model, test_save_path=None, writer=None):
    db_test = Synapse_dataset(base_dir=args.volume_path, split=args.split_name, list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        if args.dataset == "datasets":
            case_name = split(case_name.split(",")[0])[-1]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                      patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)

        mean_dice = np.mean(metric_i, axis=0)[0]
        mean_hd95 = np.mean(metric_i, axis=0)[1]
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
            i_batch, case_name, mean_dice, mean_hd95))

        # ---- TensorBoard: per-case metrics ----
        if writer is not None:
            writer.add_scalar('test_per_case/dice', mean_dice, i_batch)
            writer.add_scalar('test_per_case/hd95', mean_hd95, i_batch)
            for cls_idx in range(len(metric_i)):
                writer.add_scalar(f'test_per_class/cls_{cls_idx+1}_dice', metric_i[cls_idx][0], i_batch)
                writer.add_scalar(f'test_per_class/cls_{cls_idx+1}_hd95', metric_i[cls_idx][1], i_batch)

    metric_list = metric_list / len(db_test)

    # ---- TensorBoard: mean metrics across all cases ----
    mean_dice_all = np.mean(metric_list, axis=0)[0]
    mean_hd95_all = np.mean(metric_list, axis=0)[1]
    if writer is not None:
        writer.add_scalar('test_summary/mean_dice', mean_dice_all, 0)
        writer.add_scalar('test_summary/mean_hd95', mean_hd95_all, 0)

    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
        if writer is not None:
            writer.add_scalar(f'test_summary/cls_{i}_dice', metric_list[i - 1][0], 0)
            writer.add_scalar(f'test_summary/cls_{i}_hd95', metric_list[i - 1][1], 0)

    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    args.num_classes = args.n_class
    args.z_spacing = 1
    args.is_pretrain = True

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

    snapshot = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(snapshot):
        snapshot = snapshot.replace('best_model', 'epoch_' + str(args.max_epochs - 1))
    msg = net.load_state_dict(torch.load(snapshot))
    print("self trained swin unet", msg)
    snapshot_name = os.path.basename(snapshot)

    # ---- TensorBoard writer for testing ----
    tb_log_dir = os.path.join(args.output_dir, 'log_test')
    os.makedirs(tb_log_dir, exist_ok=True)
    test_writer = SummaryWriter(tb_log_dir)
    logging.info(f"TensorBoard test logs: {tb_log_dir}")

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    # Log model graph to TensorBoard
    dummy_input = torch.randn(1, 1, args.img_size, args.img_size).cuda()
    try:
        test_writer.add_graph(net, dummy_input)
    except Exception as e:
        logging.warning(f"Could not add model graph to TensorBoard: {e}")

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path, writer=test_writer)

    test_writer.close()
    logging.info(f"TensorBoard test logs saved to: {tb_log_dir}")

# python train.py --dataset Synapse --cfg $CFG --root_path $DATA_DIR --max_epochs $EPOCH_TIME --output_dir $OUT_DIR --img_size $IMG_SIZE --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE
# python train.py --output_dir './model_out/datasets' --dataset datasets --img_size 224 --batch_size 32 --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_preprocessed/Dataset001_mm/nnUNetPlans_2d_split
# python test.py --output_dir ./model_out/datasets --dataset datasets --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --root_path /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_preprocessed/Dataset001_mm/test --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
