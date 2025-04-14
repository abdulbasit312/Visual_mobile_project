import argparse
import datetime
import time
import torch.backends.cudnn as cudnn
import json
import torch
import numpy as np
import os
import shutil
import torch.nn as nn
from torch.ao.quantization.fuser_method_mappings import (  # noqa: F401  # noqa: F401
    fuse_conv_bn,
    fuse_conv_bn_relu,
    get_fuser_method,
)

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer

from data.datasets import build_dataset
from engine import train_one_epoch, evaluate, eval_calibrate,evaluate_quantised

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils as utils

from model_quant import *
from data.samplers import MultiScaleSamplerDDP


# print("changedddddd>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.")
# from model import rcvit_quant
import torch.quantization


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def get_args_parser():
    parser = argparse.ArgumentParser('CAS-ViT training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=2, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='rcvit_xs', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9995, help='')  # TODO: MobileViT is using 0.9995
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=6e-3, metavar='LR',
                        help='learning rate (default: 6e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--warmup_start_lr', type=float, default=0, metavar='LR',
                        help='Starting LR for warmup (default 0)')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                        help='Random erase prob (default: 0.0)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=0.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data_path', default='datasets/imagenet_full', type=str,
                        help='dataset path (path to full imagenet)')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=200, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='image_folder', choices=['IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=43, type=int)

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--finetune', default='',
                        help='finetune the model')
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    # print("use_amp thissss changeeedddddddddddddd>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>........")
    parser.add_argument('--use_amp', type=str2bool, default=False,
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")

    # Weights and Biases arguments
    parser.add_argument('--enable_wandb', type=str2bool, default=False,
                        help="enable logging to Weights and Biases")
    parser.add_argument('--project', default='edgenext', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--wandb_ckpt', type=str2bool, default=False,
                        help="Save model checkpoints as W&B Artifacts.")
    parser.add_argument("--multi_scale_sampler", action="store_true", help="Either to use multi-scale sampler or not.")
    parser.add_argument('--min_crop_size_w', default=160, type=int)
    parser.add_argument('--max_crop_size_w', default=320, type=int)
    parser.add_argument('--min_crop_size_h', default=160, type=int)
    parser.add_argument('--max_crop_size_h', default=320, type=int)
    parser.add_argument("--find_unused_params", action="store_true",
                        help="Set this flag to enable unused parameters finding in DistributedDataParallel()")
    parser.add_argument("--three_aug", action="store_true",
                        help="Either to use three augments proposed by DeiT-III")
    parser.add_argument('--classifier_dropout', default=0.0, type=float)
    parser.add_argument('--usi_eval', type=str2bool, default=False,
                        help="Enable it when testing USI model.")

    return parser


def main(args):
    #utils.init_distributed_mode(args)
    args.distributed=False
    print(args)
    device = torch.device(args.device)
    logger_name = time.strftime('%Y%m%d_%H%M%S', time.localtime())


    if args.eval:
        if args.usi_eval:
            args.crop_pct = 0.95
            model_state_dict_name = 'state_dict'
        else:
            model_state_dict_name = 'model'
    else:
        model_state_dict_name = 'model'

    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True


    # print("is_train        this changeddddddddddd>>>>>>>>>>>>>>")
    dataset_train, args.nb_classes = build_dataset(is_train=False, args=args)

    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    if args.multi_scale_sampler:
        sampler_train = MultiScaleSamplerDDP(base_im_w=args.input_size, base_im_h=args.input_size,
                                             base_batch_size=args.batch_size, n_data_samples=len(dataset_train),
                                             is_training=True, distributed=args.distributed,
                                             min_crop_size_w=args.min_crop_size_w, max_crop_size_w=args.max_crop_size_w,
                                             min_crop_size_h=args.min_crop_size_h, max_crop_size_h=args.max_crop_size_h)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
        )
        print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None
    print("log writter dir: ", args.log_dir)

    if global_rank == 0 and args.enable_wandb:
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    if args.multi_scale_sampler:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_sampler=sampler_train,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    model = create_model(args.model,pretrained=True, num_classes=args.nb_classes,
        # pretrained_cfg_overlay=dict(file="/w/331/stutiwadhwa/Visual_mobile_project/classification/class_output_200classes/checkpoint-best_224.pth"),
        drop_path_rate=args.drop_path,
        layer_scale_init_value=args.layer_scale_init_value,
        head_init_scale=1.0,
        input_res=args.input_size,
        classifier_dropout=args.classifier_dropout,
        distillation=False,
        )
        # .to('cpu')

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        # Layer decay not supported
        raise NotImplementedError
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))


    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                            find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)

    loss_scaler = NativeScaler()  # if args.use_amp is False, this won't be used

    print("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        start_warmup_value=args.warmup_start_lr
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
    args=args, model=model, model_without_ddp=model_without_ddp,
    optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema, state_dict_name=model_state_dict_name)

    model.eval()

    fuse_custom_config_dict = {"additional_fuser_method_mapping": {(torch.nn.modules.conv.Conv2d, torch.nn.modules.batchnorm.BatchNorm2d): fuse_conv_bn}}
    model_fp32_fused = torch.ao.quantization.fuse_modules(model, [["patch_embed.0","patch_embed.1","patch_embed.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["patch_embed.3","patch_embed.4","patch_embed.5"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.0.0.local_perception.network.0","network.0.0.local_perception.network.1"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.0.0.attn.oper_q.0.block.0","network.0.0.attn.oper_q.0.block.1","network.0.0.attn.oper_q.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.0.0.attn.oper_k.0.block.0","network.0.0.attn.oper_k.0.block.1","network.0.0.attn.oper_k.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.0.1.local_perception.network.0","network.0.1.local_perception.network.1"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.0.1.attn.oper_q.0.block.0","network.0.1.attn.oper_q.0.block.1","network.0.1.attn.oper_q.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.0.1.attn.oper_k.0.block.0","network.0.1.attn.oper_k.0.block.1","network.0.1.attn.oper_k.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.2.0.local_perception.network.0","network.2.0.local_perception.network.1"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.2.0.attn.oper_q.0.block.0","network.2.0.attn.oper_q.0.block.1","network.2.0.attn.oper_q.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.2.0.attn.oper_k.0.block.0","network.2.0.attn.oper_k.0.block.1","network.2.0.attn.oper_k.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.2.1.local_perception.network.0","network.2.1.local_perception.network.1"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.2.1.attn.oper_q.0.block.0","network.2.1.attn.oper_q.0.block.1","network.2.1.attn.oper_q.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.2.1.attn.oper_k.0.block.0","network.2.1.attn.oper_k.0.block.1","network.2.1.attn.oper_k.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.4.0.local_perception.network.0","network.4.0.local_perception.network.1"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.4.0.attn.oper_q.0.block.0","network.4.0.attn.oper_q.0.block.1","network.4.0.attn.oper_q.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.4.0.attn.oper_k.0.block.0","network.4.0.attn.oper_k.0.block.1","network.4.0.attn.oper_k.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.4.1.local_perception.network.0","network.4.1.local_perception.network.1"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.4.1.attn.oper_q.0.block.0","network.4.1.attn.oper_q.0.block.1","network.4.1.attn.oper_q.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.4.1.attn.oper_k.0.block.0","network.4.1.attn.oper_k.0.block.1","network.4.1.attn.oper_k.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.4.2.local_perception.network.0","network.4.2.local_perception.network.1"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.4.2.attn.oper_q.0.block.0","network.4.2.attn.oper_q.0.block.1","network.4.2.attn.oper_q.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.4.2.attn.oper_k.0.block.0","network.4.2.attn.oper_k.0.block.1","network.4.2.attn.oper_k.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.4.3.local_perception.network.0","network.4.3.local_perception.network.1"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.4.3.attn.oper_q.0.block.0","network.4.3.attn.oper_q.0.block.1","network.4.3.attn.oper_q.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.4.3.attn.oper_k.0.block.0","network.4.3.attn.oper_k.0.block.1","network.4.3.attn.oper_k.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.6.0.local_perception.network.0","network.6.0.local_perception.network.1"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.6.0.attn.oper_q.0.block.0","network.6.0.attn.oper_q.0.block.1","network.6.0.attn.oper_q.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.6.0.attn.oper_k.0.block.0","network.6.0.attn.oper_k.0.block.1","network.6.0.attn.oper_k.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.6.1.local_perception.network.0","network.6.1.local_perception.network.1"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.6.1.attn.oper_q.0.block.0","network.6.1.attn.oper_q.0.block.1","network.6.1.attn.oper_q.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)
    model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32_fused, [["network.6.1.attn.oper_k.0.block.0","network.6.1.attn.oper_k.0.block.1","network.6.1.attn.oper_k.0.block.2"]],inplace=True,fuse_custom_config_dict=fuse_custom_config_dict)

    
    model_fp32_fused.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    # model.qconfig=torch.ao.quantization.default_qconfig
    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')
    print(model_fp32_fused.qconfig)
    # print('modulesssssssssssssssssss')
    # for name, mod in model.named_modules():
    #     print("name is")
    #     print(name)
    #     print('module is')
    #     print(mod)
    
    
    
    model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused, inplace=True)
    # # model_fp32_prepared = torch.ao.quantization.prepare(model, inplace=True)
    

    evaluate(data_loader_val, model_fp32_prepared, device='cpu', use_amp=args.use_amp)
    # model_fp32_prepared.eval()
    quantized_model = torch.ao.quantization.convert(model_fp32_fused)
    n_parameters_quant = sum(p.numel() for p in quantized_model.parameters() if p.requires_grad)
    print('number of params after quant:', n_parameters_quant)

    print('modulesssssssssssssssssss')
    for name, mod in quantized_model.named_modules():
        print("name is")
        print(name)
        print('module is')
        print(mod)
    # quantized_model.to(device)
    # torch.jit.save(torch.jit.script(quantized_model),"/w/331/stutiwadhwa/Visual_mobile_project/classification/ptsq/scripted_quantized.pth")
    # model_without_ddp = quantized_model

    
    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, quantized_model, device='cpu', use_amp=args.use_amp) #quantized_model
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        torch.save(quantized_model,"/w/331/stutiwadhwa/Visual_mobile_project/classification/ptsq/scripted_quantized_t.pth")
        return
    
    


  
    # torch.jit.save(torch.jit.script(quantized_model),"/w/331/stutiwadhwa/Visual_mobile_project/classification/ptsq/scripted_quantized.pth")
# # Save the quantized model
#     quantized_checkpoint_path = "/w/331/stutiwadhwa/Visual_mobile_project/classification/post_training_static_quant"
#     torch.save(quantized_model.state_dict(), quantized_checkpoint_path)
#     print("Quantization complete. Model saved at:", quantized_checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CAS-ViT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)



