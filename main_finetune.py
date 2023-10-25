# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import pandas as pd
import datetime
import json
import numpy as np
from utils import utils
import pandas as pd
import ast
import wfdb
import numpy as np

import os
import time
from pathlib import Path
import glob
import torch
from wfdb import processing
import re
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import timm
import ast
# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import utils.lr_decay as lrd
import utils.misc as misc
from utils.datasets import build_dataset
from utils.pos_embed import interpolate_pos_embed
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import wfdb
import vit_model
from utils.ecg_dataloader import CustomDataset
from engine_finetune import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for ECG classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_1dcnn', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    # * Data Set
    
    parser.add_argument('--val_start',type=int,  default= 1,
                        help='validation start')
    parser.add_argument('--val_end',type=int, default=30,
                        help='validation end')
    parser.add_argument('--train_start',type=int, default=31,
                        help='train start')
    parser.add_argument('--train_end',type=int, default=40,
                        help='train end')
    parser.add_argument('--data',type=str, default=" ",
                        help='Which dataset')
    parser.add_argument('--classf_type',type=str, default="multi_label",
                        help='Which Classification')
    parser.add_argument('--mode',type=str, default="finetune",
                        help='Which Classification')



    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/Users/parthagrawal02/Desktop/Carelog/ECG_CNN/physionet', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=10, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir_fin',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_fin',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', default=None,
                        help='url used to set up distributed training')
    parser.add_argument('--cuda', default=None,
                        help='url used to set up distributed training')
    parser.add_argument('--data_split', default=0.8, type= float,
                        help='url used to set up distributed training')


    return parser


# Gets data from Astart - Aend from the data_path, also converts Y(target) from SNOMED ids.

# def get_data(start, end):
#     classes = {
#     426177001: 1,
#     426783006: 2,
#     164889003: 3,
#     427084000: 4,
#     164890007: 5,
#     427393009: 6,
#     426761007: 7,
#     713422000: 8,
#     233896004: 9,
#     233897008: 0
#     }
#     dataset = []
#     y = []
#     for n in range(start, end):
#         for j in range(0, 10):
#             for filepath in glob.iglob(args.data_path + '/physionet/WFDBRecords/' + f"{n:02}" +  '/' + f"{n:02}" + str(j) +  '/*.hea'):
#                 try:
#                     ecg_record = wfdb.rdsamp(filepath[:-4])
#                 except Exception:
#                     continue
#                 # annots = wfdb.Annotation(filepath[:-4], 'hea')
#                 # print(ecg_record[0].transpose(1,0).shape)
#                 numbers = re.findall(r'\d+', ecg_record[1]['comments'][2])
#                 output_array = list(map(int, numbers))
#                 for j in output_array:
#                     if int(j) in classes:
#                         output_array = j
#                 if isinstance(output_array, list):
#                     continue
#                 y.append(output_array)
#                 lx = []
#                 for chan in range(ecg_record[0].shape[1]):
#                     resampled_x, _ = wfdb.processing.resample_sig(ecg_record[0][:, chan], 500, 100)
#                     lx.append(resampled_x)
#                 dataset.append(ecg_record[0])
#     dataset = np.array(dataset)
#     print(dataset.shape)
#     dataset = dataset.astype(np.double, copy=False)
#     print(dataset.shape)
#     X = torch.from_numpy(dataset[:, :, :])
#     print(X.shape)
#     for i in range(len(y)):
#         y[i] = classes[y[i]]
#     Y = torch.from_numpy(np.array(y))
#     X = X[:, None, :, :]
#     dataset_train = torch.utils.data.TensorDataset(X, Y)

#     return dataset_train


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.cuda is not None:
        cudnn.benchmark = True
    
    if args.data == "PTB":
        # def load_raw_data(df, sampling_rate, path):
        #     if(sampling_rate == 100):
        #         data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
        #     else:
        #         data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
        #     data = np.array([signal for signal, meta in data])
        #     return data
        # path = args.data_path
        # sampling_rate = 100

        # # load and convert annotation data
        # Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
        # Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # # Load raw signal data
        # X = load_raw_data(Y, sampling_rate, path)

        # # Load scp_statements.csv for diagnostic aggregation
        # agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
        # agg_df = agg_df[agg_df.diagnostic == 1]

        # def aggregate_diagnostic(y_dic):
        #     tmp = []
        #     for key in y_dic.keys():
        #         if key in agg_df.index:
        #             tmp.append(agg_df.loc[key].diagnostic_class)
        #     return list(set(tmp))

        # # Apply diagnostic superclass
        # Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

        # # Split data into train and test
        # test_fold = 10
        # # Train
        # X_train = X[np.where(Y.strat_fold != test_fold)]
        # y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
        # # Test
        # X_test = X[np.where(Y.strat_fold == test_fold)]
        # y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

        def load_raw_data(df, sampling_rate, path):
            if(sampling_rate == 100):
                data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
            else:
                data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
            data = np.array([signal for signal, meta in data])
            return data


        sampling_frequency=100
        datafolder=args.data_path
        task='superdiagnostic'
        outputfolder='/output/'

        # Load PTB-XL data
        raw_labels = pd.read_csv(datafolder+'ptbxl_database.csv', index_col='ecg_id')
        raw_labels.scp_codes = raw_labels.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        data = load_raw_data(raw_labels, sampling_frequency, datafolder)

        # data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)
        # Preprocess label data
        labels = utils.compute_label_aggregations(raw_labels, datafolder, task)
        # Select relevant data and convert to one-hot
        data, labels, Y, _ = utils.select_data(data, labels, task, min_samples=0, outputfolder=outputfolder)

        # 1-9 for training 
        X_train = data[labels.strat_fold < 10]
        y_train = Y[labels.strat_fold < 10]
        # 10 for validation
        X_test = data[labels.strat_fold == 10]
        y_test = Y[labels.strat_fold == 10]

        X_train = torch.tensor(X_train.transpose(0, 2, 1))
        mean = X_train.mean(dim=-1, keepdim=True)
        var = X_train.var(dim=-1, keepdim=True)
        X_train = (X_train - mean) / (var + 1.e-6)**.5
        X_test = torch.tensor(X_test.transpose(0, 2, 1))
        mean = X_test.mean(dim=-1, keepdim=True)
        var = X_test.var(dim=-1, keepdim=True)
        X_test = (X_test - mean) / (var + 1.e-6)**.5
        dataset_train = torch.utils.data.TensorDataset(torch.tensor(X_train[:, None, :, :]).double(), torch.tensor(y_train).double())
        dataset_val = torch.utils.data.TensorDataset(torch.tensor(X_test[:, None, :, :]).double(), torch.tensor(y_test).double())

    else:
        # dataset_train = build_dataset(is_train=True, args=args)
        # dataset_val = build_dataset(is_train=False, args=args)
        full_dataset = CustomDataset(args.data_path, args.train_start, args.train_end)    # Training Data -
        train_size = int(args.data_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        dataset_train, dataset_val = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    


    if args.distributed is not None:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    model = vit_model.__dict__[args.model](
        num_classes=args.nb_classes,
        # drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # print(checkpoint_model)
        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    if(args.mode == "linprobe"):
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True
    
    model = model.double()
    if args.cuda is not None:
        model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_without_ddp = model

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    # if mixup_fn is not None:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif args.smoothing > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    # else:
    if args.classf_type == "multi_label":
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    # ckpt_file = args.ckpt
    # state_dict = torch.load(ckpt_file, map_location="cpu")

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    # model.load_state_dict(state_dict, strict=True)
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, args = args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device, args = args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            # log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
