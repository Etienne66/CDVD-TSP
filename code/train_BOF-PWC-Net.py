from __future__ import division
#import signal
import platform
import os
import traceback
import winsound
import argparse
from pathlib import Path
import time
import datetime
import math
import matplotlib.pyplot as plt
from statistics import mean 
#from distutils.version import LooseVersion
from packaging import version

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
#import future
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter
from tqdm import tqdm
import flow_vis

# to output the visualization, please download flow_io.py and viz_flow.py from https://github.com/jswulff/pcaflow/tree/master/pcaflow/utils
#from utils.viz_flow import viz_flow

import data_flow
import data_flow.flow_transforms as flow_transforms
import model_flows as models
import utils.utils as utils
from utils.util import flow2rgb, AverageMeter, save_checkpoint
import loss
import data_flow.datasets as datasets


model_names = sorted(name for name in models.__all__)
dataset_names = sorted(name for name in data_flow.__all__)

parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR', type=Path, default=Path('../datasets'),
                    help='path to datasets')
parser.add_argument('--pretrained', type=Path, default=None,
                    help='path to pre-trained model')
parser.add_argument('--resume', type=Path, default=None,
                    help='path to pre-trained model')
parser.add_argument('--save_path', type=Path, default='../runs',
                    help='path to save models/tensorboard metrics')
parser.add_argument('--loader_device', type=str, default='cpu', #'cuda' if torch.cuda.is_available() else 'cpu',
                    choices=['cpu','cuda'],
                    help='Device to initially load images/flows into to perform augmentation.')
parser.add_argument('--dataset', metavar='DATASET', default='autoflow', #flying_things_both flying_chairs2 flying_things_final flying_things_clean
                    choices=dataset_names,# nargs='*',
                    help='dataset type: ' + ' | '.join(dataset_names))
parser.add_argument('--datasets', metavar='DATASET', default=['things3_clean.1','things3_clean','things3_clean.up'], #flying_things_both flying_chairs2 flying_things_final flying_things_clean
                    choices=['things3','things3.1','things3.up1','things3.up',
                             'things3_clean','things3_clean.1','things3_clean.up',
                             'things3_final','things3_final.1','things3_final.up',
                             'viper1','viper3','viper3.1','viper3.2'], nargs='+',
                    help='dataset type: ' + ' | '.join(['autoflow','chairs2','things','things_clean','viper']))
parser.add_argument('--testdataset', metavar='DATASET', default='mpi_sintel3_final', #mpi_sintel_final mpi_sintel_clean KITTI_2015_occ
                    choices=dataset_names,
                    help='dataset type: ' + ' | '.join(dataset_names))
parser.add_argument('--arch', '-a', metavar='ARCH', default='boflow_pwc',
                    choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' + ' | '.join(model_names))
parser.add_argument('--n_GPUs', type=int, default=1, metavar='N',
                    help='number of GPUs')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers. Set to number of cpus')
parser.add_argument('-b', '--batch_size', dest='batch_size', default=None, type=int, #22
                    choices=[1,2,4,8,16,32,64,128],
                    metavar='N', help='mini-batch size')
parser.add_argument('--test_batch_size', default=4, type=int, #6
                    metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num_steps', default=6_200_000, type=int, metavar='N',
                    help='number of total steps to run')
parser.add_argument('--max_num_steps', default=0, type=int, metavar='N',
                    help='step number to stop run')
parser.add_argument('--accumulation_steps', type=int, default=1,
                    help='True_Batch_Size = accumalation_steps * batch_size')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch_size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
                                             # autoflow,chairs2,things,viper
#parser.add_argument('--epoch_sizes', default=[1250,1388,9820,3328], type=int, metavar='N', # 40000,44464,39280,13356  78564->39280
#                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('--test_epoch_size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('--print_freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--val_freq', type=int, default=5000,
                    help='validation frequency')
parser.add_argument('--max_dataset_len', type=int, default=1_000_000,
                    help='validation frequency')
parser.add_argument('--image_freq', default=60, type=int,
                    metavar='N', help='test image frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--no_date', action='store_true',
                    help='don\'t append date timestamp to folder' )
parser.add_argument('--div_flow', type=int, default=1, metavar='N',
                    choices=[1,2,4,8,16,20,32],
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
parser.add_argument('--batchnorm', action='store_true',
                    help='use torch.nn.BatchNorm2d after torch.nn.Conv2d')
parser.add_argument('--batchnorm2', action='store_true',
                    help='use torch.nn.BatchNorm2d after Activation layer')
parser.add_argument('--normalize', action='store_true',
                    help='use Normalize images')
parser.add_argument('--large_motion', action='store_true',
                    help='Change to increase channels for flow 4,5,6 to better detect large motions')
parser.add_argument('--full_motion', action='store_true',
                    help='Change to increase channels for flow 3,4,5,6 to better detect large motions')
parser.add_argument('--useOrginalModel', action='store_true', help='use Original Model with LeakyReLu for optical flow estimator')
parser.add_argument('--useFlow1', action='store_true', help='use flow 1 as the final stage instead of flow 2')
parser.add_argument('--FullScale', action='store_true', help='Scale Flow2 to full scale with ConvTranspose2d')
parser.add_argument('--UpScale', action='store_true', help='Scale small input images by 2')
parser.add_argument('--useResidual', action='store_true', help='use residual features in optical flow estimator')
parser.add_argument('--useDenseNet', action='store_true', help='use DenseNet in optical flow estimator')
parser.add_argument('--usePReLu', action='store_true', help='use PReLu instead of LeakyReLU in optical flow estimator Extractor')
parser.add_argument('--usePReLuD', action='store_true', help='use PReLu instead of LeakyReLU in optical flow estimator Decoder')
parser.add_argument('--Reset_Layer6', action='store_true', help='use PReLu instead of LeakyReLU in optical flow estimator')
parser.add_argument('--DenseNetSize', type=int, default=0, choices=[0,1], help='Size of DenseNet. 0 = Original; 1 = FullDenseNet')
parser.add_argument('--useContext', action='store_true', help='use context network to refine flow 2')
parser.add_argument('--scale_supervision', action='store_true', help='scale supervision signal at each level instead of using div_flow')
parser.add_argument('--scale_output', action='store_true', help='scale outputs for evaluation during training.')
parser.add_argument('--EC', type=int, default=[16,32,64,128,512,2048], metavar='N', nargs=6, #16,32,64,96,128,192  16,32,64,128,256,512   16,32,64,128,512,2048
                    help='Extract Channels list')
parser.add_argument('--max_displacement', type=int, default=[4,4,4,4,4,4], metavar='N', nargs=6,
                    help='search range for correlation function')
parser.add_argument('--Crop_Size', type=int, default=None, metavar='N', nargs=2,
                    help='Size of random Crop. H W')
parser.add_argument('--loss', default='0*mWAUCl+1*mEPE', type=str,
                    choices=['0*mWAUCl+1*mEPE1','0*mWAUCl+1*mEPE',
                             '8.4*mWAUCl+1.6*mEPE1','1.6*mWAUCl+8.4*mEPE','1.6*mWAUCl+8.4*mEPE1',
                             '1*mWAUCl+2*mEPE1','1*mWAUCl+2*mEPE',
                             '1*mWAUCl+4*mEPE1','1*mWAUCl+4*mEPE',
                             '1*mWAUCl+1*mEPE1','1*mWAUCl+1*mEPE'],
                    help='Loss function. EPE = Average endpoint error; Fl = Percentage of optical flow outliers from 0 to 100 percent')
parser.add_argument('--weights', type=float, nargs='+', default=None) #[1, 0.5, 0.25, 0.25, 0.25, 0.25] [0, 1, 0.75, 0.5, 0.375, 0.25]

parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--seed', type=int, default=1966,
                    help='random seed')
parser.add_argument('--split_losses', action='store_true',
                    help='split flow losses and graph them')
parser.add_argument('--clip', type=float, default=None, help='gradient clipping')
parser.add_argument('--clamp', type=float, default=None, help='clamp vectors to a maximum')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint')

AugmenterGroup = parser.add_argument_group('Augmenter', description='Augmenter options')
#AugmenterGroup.add_argument('--stage', type=str, default='autoflow',
#                    help="determines which dataset to use for training")
AugmenterGroup.add_argument('--image_size', type=int, nargs='+'#, default=[488, 576]
                            )
AugmenterGroup.add_argument('--shiftprob', dest='shift_aug_prob', type=float, default=0.0,
                    help='Probability of shifting augmentation')
AugmenterGroup.add_argument('--spatial_aug_prob', type=float, default=0.80,
                    help='Probability of shifting augmentation')
AugmenterGroup.add_argument('--shiftsigmas', dest='shift_sigmas', default="16,10", type=str,
                    help='Stds of shifts for shifting consistency loss')

OptimizerGroup = parser.add_argument_group('Optimizer', description='Optimizer options')
OptimizerGroup.add_argument('--solver', default='adamw', type=str,
                    choices=['adam','adamw','sgd'],
                    help='solver algorithms')
OptimizerGroup.add_argument('--lr', '--learning-rate', default=None, type=float, metavar='LR', #1.0e-7
                    help='initial learning rate')
OptimizerGroup.add_argument('--max_lr', default=2.5e-4, type=float, metavar='M',
                    help='Sets maximum Learning Rate for LRFinder, OneCycleLR, and CyclicLr')
OptimizerGroup.add_argument('--lr_decay', type=int, default=200, metavar='N',
                    help='learning rate decay per N epochs')
OptimizerGroup.add_argument('--weight_decay', type=float, default=0.,
                    help='weight decay')
OptimizerGroup.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd')
OptimizerGroup.add_argument('--betas', type=float, default=[0.9,0.999], metavar='M', nargs=2,
                    help='ADAM betas')


SchedulerGroup = parser.add_argument_group('Scheduler', description='Scheduler options')
SchedulerGroup.add_argument('--scheduler', default='OneCycle',
                    choices=['multisteplr','steplr','onecyclelr','OneCycle','cycliclr','lr_finder'],
                    help='scheduler algorithms')
SchedulerGroup.add_argument('--step_size_up', type=int, default=5_000, metavar='N',  #337
                    help='Number of training iterations in the increasing half of a cycle.')
SchedulerGroup.add_argument('--gamma', type=float, default=0.5, metavar='M',
                    help='learning rate decay factor for step decay')
SchedulerGroup.add_argument('--CyclicLR_gamma', type=float, default=0.9999930685522169957388034152483, metavar='M',
                    help='learning rate decay factor for step decay')
SchedulerGroup.add_argument('--CyclicLR_mode', type=str, default='exp_range',
                    choices=['triangular','triangular2','exp_range'],
                    help='learning rate policy')
SchedulerGroup.add_argument('--milestones', default=[72,108,144,180], metavar='N', nargs='*',
                    help='epochs at which learning rate is divided by 2')
SchedulerGroup.add_argument('--lr_finder_Leslie_Smith', action='store_true',
                    help='Run the LRFinder using Leslie Smith''s approach')
OptimizerGroup.add_argument('--pct_start', default=0.10, type=float, metavar='M',
                    help='pct_start for OneCycle')
OptimizerGroup.add_argument('--anneal_strategy', default='linear', type=str,
                    choices=['cos','linear'],
                    help='anneal_strategy for OneCycle')


batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()
flow_Fls = AverageMeter()
flow_EPEs = AverageMeter()
flow_WAUCs = AverageMeter()
flow1_WAUC = AverageMeter()
flow2_WAUC = AverageMeter()
flow3_WAUC = AverageMeter()
flow4_WAUC = AverageMeter()
flow5_WAUC = AverageMeter()
flow6_WAUC = AverageMeter()
flow1_EPE = AverageMeter()
flow2_EPE = AverageMeter()
flow3_EPE = AverageMeter()
flow4_EPE = AverageMeter()
flow5_EPE = AverageMeter()
flow6_EPE = AverageMeter()
learningRates = AverageMeter()

terminate = False

#def signal_handling(signum,frame):           
#    global terminate                         
#    terminate = True

def reset_all_weights(self, model: torch.nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: torch.nn.Module):
        # - check if the current module has reset_parameters & if it's callable, call it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def min_max_parameters(model):
    model_min = 0
    model_max = 0
    model_mean = AverageMeter()
    for p in model.parameters():
        if p.requires_grad:
            model_min = min(model_min, p.min().item())
            model_max = max(model_max, p.max().item())
            model_mean.update = (p.norm().item())
    return model_min, model_max, model_mean.avg

def min_max_gradients(model):
    model_max = 0.0
    model_mean = AverageMeter()
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            model_max = max(model_max, p.grad.abs().max().item())
            model_mean.update = (p.grad.abs().mean().item())
    return model_max, model_mean.avg

def norm_gradients(model):
    model_mean = AverageMeter()
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            model_mean.update = (p.grad.norm())
    return model_mean.avg, mdoel_mean.count

def main():
    global args, terminate, tqdm_main, tqdm_train0, tqdm_train1, tqdm_train2, tqdm_test0, tqdm_test1, tqdm_test2
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cpu = False
    args.ddp = False
    args.lr_finder = True if args.scheduler == 'lr_finder' else False
    args.n_sequence = 2
    args.total_steps = 0
    args.train_loader_start = 0
    args.best_EPE = 100
    args.best_Fl = 100
    args.best_WAUC = -1
    args.load = None #Used by loss module
    args.n_channel=2
    #args.normalize = True
    #args.mini_batch_size = int(args.batch_size / args.accumulation_steps)

    if args.max_num_steps == 0:
        args.max_num_steps = args.num_steps

    if args.pretrained:
        network_data = torch.load('../pretrain_models' / args.pretrained, map_location=torch.device(args.device))
        network_data.pop('optimizer_state_dict', None)
        network_data.pop('scheduler_state_dict', None)
        save_path = args.save_path
        if 'state_dict' not in network_data.keys():
            network_data = {'state_dict': network_data}
            args.arch = 'flow_pwc'
            args.div_flow = 20
            args.batchnorm = False
            args.large_motion = False
            args.useFlow1 = False
            args.normalize = False
        elif 'args' in network_data.keys():
            old_args = network_data['args']
            args.arch = old_args.arch
            args.large_motion = old_args.large_motion
            args.FullScale = old_args.FullScale
            args.clip = old_args.clip if args.clip is None else args.clip
            args.clamp = old_args.clamp
            args.div_flow = old_args.div_flow
            args.useDenseNet = old_args.useDenseNet
            args.useContext = old_args.useContext
            args.max_displacement = old_args.max_displacement
            args.normalize = old_args.normalize
            args.batchnorm = old_args.batchnorm
            args.batchnorm2 = old_args.batchnorm2
            args.useFlow1 = old_args.useFlow1
            args.usePReLu = old_args.usePReLu
            args.usePReLuD = old_args.usePReLuD
            args.max_displacement = old_args.max_displacement
            args.EC = old_args.EC
            ##Don't use state_dict for scheduler and optimizer because this is not a resume
            #if 'optimizer_state_dict' in network_data.keys(): network_data.pop('optimizer_state_dict')
            #if 'scheduler_state_dict' in network_data.keys(): network_data.pop('scheduler_state_dict')
            #
            ## Reset device in case the model was saved with a different device
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #args.save_path = save_path
            #args.start_epoch = 0
            #args.n_iter = 0
        if args.batch_size is None:
            args.batch_size = 8
        if args.evaluate:
            args.run_path = Path(args.arch + '-' + datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S") + '-PreTrained-Eval')
        else:
            args.run_path = Path(args.arch + '-' + datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S") + '-PreTrained')
        print("=> using pre-trained model '{}\\{}'".format(args.arch,args.pretrained))
    elif args.resume is not None:
        # Save resume and epochs to restore after loading
        resume = args.resume
        clip = args.clip
        clamp = args.clamp
        epochs = args.epochs
        max_num_steps = args.max_num_steps
        batch_size = args.batch_size
        pct_start = args.pct_start
        num_workers = args.num_workers
        network_data = torch.load(args.save_path / args.resume,
                                  map_location=torch.device(args.device),
                                  weights_only=True)
        args = network_data['args']
        #Restore resume and epochs.
        args.resume = resume
        args.epochs = epochs
        args.num_workers = num_workers
        args.max_num_steps = max_num_steps
        args.clip = args.clip if clip is None else clip
        args.clamp = args.clamp if clamp is None else clamp
        if batch_size is not None:
            args.batch_size = batch_size
        if args.pct_start is None:
            args.pct_start = pct_start
        if not hasattr(args, 'EC'):
            args.EC = [16,32,64,128,512,2048]
            args.batchnorm = True
        

        # Reset device in case the model was saved with a different device
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reset flow counts to continues after a resume
        if args.flow_EPE_Count > 0: flow_EPEs.update(args.flow_EPE_Avg, args.flow_EPE_Count)
        if args.flow_F1_Count > 0: flow_Fls.update(args.flow_F1_Avg, args.flow_F1_Count)
        if args.flow_WAUC_Count > 0: flow_WAUCs.update(args.flow_WAUC_Avg, args.flow_WAUC_Count)
        if args.flow1_WAUC_Count > 0: flow1_WAUC.update(args.flow1_WAUC_Avg, args.flow1_WAUC_Count)
        if args.flow2_WAUC_Count > 0: flow2_WAUC.update(args.flow2_WAUC_Avg, args.flow2_WAUC_Count)
        if args.flow3_WAUC_Count > 0: flow3_WAUC.update(args.flow3_WAUC_Avg, args.flow3_WAUC_Count)
        if args.flow4_WAUC_Count > 0: flow4_WAUC.update(args.flow4_WAUC_Avg, args.flow4_WAUC_Count)
        if args.flow5_WAUC_Count > 0: flow5_WAUC.update(args.flow5_WAUC_Avg, args.flow5_WAUC_Count)
        if args.flow6_WAUC_Count > 0: flow6_WAUC.update(args.flow6_WAUC_Avg, args.flow6_WAUC_Count)
        if args.flow1_EPE_Count > 0: flow1_EPE.update(args.flow1_EPE_Avg, args.flow1_EPE_Count)
        if args.flow2_EPE_Count > 0: flow2_EPE.update(args.flow2_EPE_Avg, args.flow2_EPE_Count)
        if args.flow3_EPE_Count > 0: flow3_EPE.update(args.flow3_EPE_Avg, args.flow3_EPE_Count)
        if args.flow4_EPE_Count > 0: flow4_EPE.update(args.flow4_EPE_Avg, args.flow4_EPE_Count)
        if args.flow5_EPE_Count > 0: flow5_EPE.update(args.flow5_EPE_Avg, args.flow5_EPE_Count)
        if args.flow6_EPE_Count > 0: flow6_EPE.update(args.flow6_EPE_Avg, args.flow6_EPE_Count)

        # Reset random number generators
        torch.set_rng_state(network_data['rng_state'].type(torch.ByteTensor))
        random.setstate(network_data['random_state'])
        np.random.set_state(network_data['numpy_random_state'])
        print("=> using trained model '{}\\{}' with divisions '{}'".format(args.arch,args.resume,args.div_flow))
        print("Clip at ", args.clip)
    else:
        network_data = {}
        print("=> creating model '{}' with divisions '{}'".format(args.arch,args.div_flow))
        print("=> spatial_aug_prob '{}'".format(args.spatial_aug_prob))
        args.run_path = Path(args.arch + '-' + datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S"))
        args.clip = 1.0 if args.clip is None else args.clip

    args.step_size = int(args.batch_size / 8) # Size of the original author's batch size
    args.loss_Flow_net = loss.Loss(args)

    print('=> will save everything to {}'.format(args.save_path / args.run_path))
    if not (args.save_path / args.run_path).exists():
        (args.save_path / args.run_path).mkdir(parents=True)

    # Set default seed for repeatability...
    if args.seed is not None and args.resume is None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Data loading code
    # transforms.Normalize output[channel] = (input[channel] - mean[channel]) / std[channel]
    # All channels are RGB. Loads from OpenCV were corrected
    if args.normalize:
        input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]), # (0,255) -> (0,1)
            #transforms.Normalize(mean=[0.45,0.432,0.411], std=[1,1,1]) # (0,1) -> (-0.5,0.5)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # from ImageNet dataset
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Standard normalization
        ])

        inverse_transform = transforms.Compose([
            transforms.Normalize(mean= [0., 0., 0.], std = [1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean= [-0.485, -0.456, -0.406], std = [1., 1., 1.])
        ])
        png_normalize = transforms.Compose([
            transforms.Normalize(mean= [0., 0., 0.], std = [1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean= [-0.485, -0.456, -0.406], std = [1., 1., 1.]),
            transforms.Normalize(mean=[0, 0, 0], std=[1/255, 1/255, 1/255]) # (0,1) -> (0,255)
        ])
    else:
        input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]), # (0,255) -> (0,1)
        ])
        inverse_transform = None
        png_normalize = transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0], std=[1/255, 1/255, 1/255]) # (0,1) -> (0,255)
        ])

    target_transform = transforms.Compose([
        flow_transforms.ArrayToTensor()#,
        #transforms.Normalize(mean=[0,0],std=[args.div_flow,args.div_flow])
    ])

    co_validate_transform = flow_transforms.Compose([
        flow_transforms.CenterCrop((704,1280)) if args.testdataset in ('viper') else None, #1920x1080
        flow_transforms.CenterCrop((320,768)) if args.testdataset in ('mpi_sintel_final','mpi_sintel_clean') else None
    ])

    co_transform = flow_transforms.Compose([
        #flow_transforms.RandomTranslate(10),
        flow_transforms.RandomCrop((512,960)) if args.dataset in ('flying_things_clean','flying_things_final','flying_things_both') else None,
        flow_transforms.RandomCrop((384,1024)) if args.dataset in ('mpi_sintel_final','mpi_sintel_clean') else None,
        flow_transforms.RandomCrop((320,1216)) if args.dataset in ('KITTI_2015_occ','KITTI_2015_noc','KITTI_2012_occ','KITTI_2012_noc') else None,
        #flow_transforms.RandomCrop((704,1280)) if args.dataset in ('viper') else None,#1920x1080
        flow_transforms.RandomCrop((1024,1920)) if args.dataset in ('viper') else None,#1920x1080
        flow_transforms.RandomCrop(Crop_Size) if args.Crop_Size is not None else None,
        #flow_transforms.RandomRot90() if args.RandomRot90 else None, #flow_transforms.RandomRotate(10,5),
        flow_transforms.RandomVerticalFlip(), #transforms.RandomVerticalFlip,
        flow_transforms.RandomHorizontalFlip() #transforms.RandomHorizontalFlip()
    ])

    #print("=> fetching train image pairs in '{}\\{}'".format(args.data,args.dataset))
    #train_set = data_flow.__dict__[args.dataset](
    #    args.data,
    #    transform=input_transform,
    #    target_transform=target_transform,
    #    co_transform=co_transform
    #)
    #train_loader = torch.utils.data.DataLoader(
    #    train_set,
    #    batch_size=args.batch_size,
    #    num_workers=args.num_workers,
    #    pin_memory=True,
    #    shuffle=True)
    #print("=> fetching train image pairs in '{}\\{}'".format(args.data,args.dataset))
    
    #train_loader = datasets.fetch_dataloader(args,
    #                                         image_size      = args.image_size,
    #                                         mini_batch_size = args.mini_batch_size,
    #                                         stage           = args.dataset)

    if args.image_size is not None:
        image_sizes = {
            "autoflow"      : [min(448, args.image_size[0]), min(576, args.image_size[1])],
            "chairs2"       : [min(384, args.image_size[0]), min(512, args.image_size[1])],
            "things"        : [min(512, args.image_size[0]), min(960, args.image_size[1])],
            "things_clean"  : [min(512, args.image_size[0]), min(960, args.image_size[1])],
            "things_final"  : [min(512, args.image_size[0]), min(960, args.image_size[1])],
            "things3"       : [min(512, args.image_size[0]), min(960, args.image_size[1])],
            "things3.up"    : [min(512, args.image_size[0]), min(960, args.image_size[1])],
            "things3_clean" : [min(512, args.image_size[0]), min(960, args.image_size[1])],
            "things3_final" : [min(512, args.image_size[0]), min(960, args.image_size[1])],
            "viper"         : [min(1024, args.image_size[0]), min(1920, args.image_size[1])],
            "viper1"        : [min(1024, args.image_size[0]), min(1920, args.image_size[1])],
            "viper3"        : [min(1024, args.image_size[0]), min(1920, args.image_size[1])]
        }
    else:
        image_sizes = {
            "autoflow"          : [ 448, 576], #258k x2 896x1152 1032k
            "autoflow.up"       : [ 448, 576], #258k x2 896x1152 1032k
            "chairs2"           : [ 384, 512], #197k x2 768x1024 786k
            "chairs2.up"        : [ 384, 512], #197k x2 768x1024 786k
            "things"            : [ 512, 960], #492k
            "things.1"          : [ 512, 512], #229k
            "things_clean"      : [ 540, 960], #506k
            "things_clean.up"   : [ 540, 960], #506k
            "things_clean.1"    : [ 512, 960], #294k
            "things_clean.2"    : [ 512, 512], #229k
            "things_final"      : [ 540, 960], #506k
            "things_final.up"   : [ 540, 960], #506k
            "things_final.1"    : [ 384, 768], #294k
            "things_final.2"    : [ 512, 512], #229k
            #"things3"           : [ 512 if args.spatial_aug_prob == 0.0 else 384,
            #                        960 if args.spatial_aug_prob == 0.0 else 768], #506k
            #"things3"           : [ 512, 960], #200k
            "things3"           : [ 448, 896], #200k
            "things3.1"         : [ 448, 896], #200k
            "things3.up1"       : [ 512, 960], #200k
            #"things3.up1"       : [ 384, 768], #200k
            "things3.up"        : [ 384, 768], #506k
            "things3_clean"     : [ 540, 960], #506k
            "things3_clean.1"   : [ 540, 960], #200k
            "things3_clean.up"  : [ 540, 960], #506k
            "things3_final"     : [ 540, 960], #506k
            "things3_final.1"   : [ 540, 960], #200k
            "things3_final.up"  : [ 540, 960], #506k
            "viper"             : [1024,1920], #1966k
            #"viper1"            : [1024,1920], #1966k
            "viper1"            : [1080,1920], #1966k
            "viper1.1"          : [ 704,1280], #901k
            "viper1.2"          : [ 576, 576], #332k
            "viper1.3"          : [ 384, 768], #294k
            "viper1.4"          : [ 512, 512], #262k
            "viper3"            : [1080,1920], #1966k
            "viper3.1"          : [ 768,1408], #1966k
            "viper3.2"          : [ 512, 960]  #1966k 368 768
        }

    if args.useFlow1 and args.useDenseNet and (args.useResidual or args.useContext):
        mini_batch_sizes = {
            "autoflow"      : min(4,args.batch_size),
            "chairs2"       : min(8,args.batch_size),
            "things"        : min(4,args.batch_size),
            "things_clean"  : min(4,args.batch_size),
            "things_final"  : min(4,args.batch_size),
            "viper"         : min(2,args.batch_size),
            "viper1"        : min(2,args.batch_size),
            "viper3"        : min(2,args.batch_size)
        }
    elif args.useFlow1 and (args.useResidual or args.useContext):
        mini_batch_sizes = {
            "autoflow"      : min(16,args.batch_size),
            "chairs2"       : min(16,args.batch_size),
            "things"        : min(8,args.batch_size),
            "things_clean"  : min(8,args.batch_size),
            "things_final"  : min(8,args.batch_size),
            "viper"         : min(4,args.batch_size),
            "viper1"        : min(4,args.batch_size),
            "viper3"        : min(4,args.batch_size)
        }
    elif args.useFlow1 and args.batchnorm:
        mini_batch_sizes = {
            "autoflow"      : min(16,args.batch_size),
            "chairs2"       : min(32,args.batch_size),
            "things"        : min(8,args.batch_size),
            "things_clean"  : min(8,args.batch_size),
            "things_final"  : min(8,args.batch_size),
            "viper"         : min(2,args.batch_size),
            "viper1"        : min(2,args.batch_size),
            "viper3"        : min(2,args.batch_size)
        }
    elif args.max_displacement[0] == 8 \
     and args.DenseNetSize == 1 \
     and args.image_size is not None \
     and args.image_size[0] <= 704 \
     and args.image_size[1] <= 1280:
        mini_batch_sizes = {
            "autoflow"      : min(8,args.batch_size),
            "chairs2"       : min(8,args.batch_size),
            "things"        : min(4,args.batch_size),
            "things_clean"  : min(4,args.batch_size),
            "things_final"  : min(4,args.batch_size),
            "viper"         : min(4,args.batch_size),
            "viper1"        : min(4,args.batch_size),
            "viper3"        : min(4,args.batch_size)
        }
    elif args.max_displacement[0] == 8 and args.DenseNetSize == 1:
        mini_batch_sizes = {
            "autoflow"      : min(8,args.batch_size),
            "chairs2"       : min(16,args.batch_size),
            "things"        : min(4,args.batch_size),
            "things_clean"  : min(4,args.batch_size),
            "things_final"  : min(4,args.batch_size),
            "viper"         : min(2,args.batch_size),
            "viper1"        : min(2,args.batch_size),
            "viper3"        : min(2,args.batch_size)
        }
    elif args.max_displacement[0] == 8 \
     and args.DenseNetSize == 0 \
     and args.image_size is not None \
     and args.image_size[0] <= 704 \
     and args.image_size[1] <= 1280:
        mini_batch_sizes = {
            "autoflow"          : min(16,args.batch_size),
            "chairs2"           : min(32,args.batch_size),
            "things"            : min(8,args.batch_size),
            "things_clean"      : min(8,args.batch_size),
            "things_final"      : min(8,args.batch_size),
            "things3"           : min(8,args.batch_size),
            "things3.1"         : min(16,args.batch_size),
            "things3.up1"       : min(8,args.batch_size),
            "things3.up"        : min(4,args.batch_size),
            "things3_clean"     : min(8,args.batch_size),
            "things3_clean.1"   : min(16,args.batch_size),
            "things3_clean.up"  : min(2,args.batch_size),
            "things3_final"     : min(8,args.batch_size),
            "things3_final.1"   : min(16,args.batch_size),
            "things3_final.up"  : min(2,args.batch_size),
            "viper"             : min(2,args.batch_size),
            "viper1"            : min(2,args.batch_size),
            "viper1.1"          : min(2,args.batch_size),
            "viper1.2"          : min(16,args.batch_size),
            "viper3"            : min(8,args.batch_size)
        }
    elif args.max_displacement[0] == 8 and args.UpScale and args.useContext:
        mini_batch_sizes = {
            "autoflow"      : min(4,args.batch_size),
            "chairs2"       : min(8,args.batch_size),
            "things"        : min(2,args.batch_size),
            "things.1"      : min(16,args.batch_size),
            "things_clean"  : min(2,args.batch_size),
            "things_clean.1": min(8,args.batch_size),
            "things_clean.2": min(16,args.batch_size),
            "things_final"  : min(8,args.batch_size),
            "things_final.1": min(8,args.batch_size),
            "things_final.2": min(16,args.batch_size),
            "viper"         : min(4,args.batch_size),
            "viper1"        : min(2,args.batch_size),
            "viper1.1"      : min(8,args.batch_size),
            "viper1.2"      : min(8,args.batch_size),
            "viper1.3"      : min(16,args.batch_size),
            "viper1.4"      : min(16,args.batch_size),
            "viper3"        : min(2,args.batch_size)
        }
    elif args.max_displacement[0] == 8 and args.UpScale:
        mini_batch_sizes = {
            "autoflow"          : min(4,args.batch_size),
            "autoflow.up"       : min(4,args.batch_size),
            "chairs2"           : min(4 if args.EC[5] >= 256 else 8,args.batch_size),
            "chairs2.up"        : min(4 if args.EC[5] >= 256 else 8,args.batch_size),
            "things"            : min(4,args.batch_size),
            "things.1"          : min(16,args.batch_size),
            "things_clean"      : min(4,args.batch_size),
            "things_clean.up"   : min(4,args.batch_size),
            "things_clean.1"    : min(8,args.batch_size),
            "things_clean.2"    : min(16,args.batch_size),
            "things_final"      : min(8,args.batch_size),
            "things_final.up"   : min(4,args.batch_size),
            "things_final.1"    : min(8,args.batch_size),
            "things_final.2"    : min(16,args.batch_size),
            "viper"             : min(4,args.batch_size),
            "viper1"            : min(2,args.batch_size),
            "viper1.1"          : min(8,args.batch_size),
            "viper1.2"          : min(8,args.batch_size),
            "viper1.3"          : min(16,args.batch_size),
            "viper1.4"          : min(16,args.batch_size),
            "viper3"            : min(2,args.batch_size)
        }
    elif args.max_displacement[0] == 8:
        mini_batch_sizes = {
            "autoflow"          : min(16,args.batch_size),
            "autoflow.up"       : min(4 if args.EC[5] > 512 else 8,args.batch_size),
            "chairs2"           : min(16,args.batch_size),
            "chairs2.up"        : min(4 if args.EC[5] > 512 else 8,args.batch_size),
            "things"            : min(8,args.batch_size),
            "things.1"          : min(16,args.batch_size),
            "things_clean"      : min(8,args.batch_size),
            "things_clean.up"   : min(4,args.batch_size),
            "things_clean.1"    : min(8,args.batch_size),
            "things_clean.2"    : min(16,args.batch_size),
            "things_final"      : min(8,args.batch_size),
            "things_final.up"   : min(4,args.batch_size),
            "things_final.1"    : min(8,args.batch_size),
            "things_final.2"    : min(16,args.batch_size),
            "things3"           : min(8,args.batch_size),
            "things3.1"         : min(16,args.batch_size),
            "things3.up1"       : min(4,args.batch_size),
            "things3.up"        : min(4,args.batch_size),
            "things3_clean"     : min(8,args.batch_size),
            "things3_clean.1"   : min(16,args.batch_size),
            "things3_clean.up"  : min(2,args.batch_size),
            "things3_final"     : min(8,args.batch_size),
            "things3_final.1"   : min(16,args.batch_size),
            "things3_final.up"  : min(2,args.batch_size),
            "viper"             : min(4,args.batch_size),
            "viper1"            : min(4,args.batch_size),
            "viper1.1"          : min(8,args.batch_size),
            "viper1.2"          : min(8,args.batch_size),
            "viper1.3"          : min(16,args.batch_size),
            "viper1.4"          : min(16,args.batch_size),
            "viper3"            : min(4,args.batch_size),
            "viper3.1"          : min(4,args.batch_size),
            "viper3.2"          : min(8,args.batch_size)
        }
    elif 4 < args.max_displacement[0] <= 8:
        mini_batch_sizes = {
            "autoflow"          : min(16,args.batch_size),
            "autoflow.up"       : min(4,args.batch_size),
            "chairs2"           : min(32,args.batch_size),
            "chairs2.up"        : min(4 if args.EC[5] >= 256 else 8,args.batch_size),
            "things"            : min(8,args.batch_size),
            "things.1"          : min(16,args.batch_size),
            "things.1"          : min(16,args.batch_size),
            "things_clean"      : min(8,args.batch_size),
            "things_clean.up"   : min(4,args.batch_size),
            "things_clean.1"    : min(8,args.batch_size),
            "things_clean.2"    : min(16,args.batch_size),
            "things_final"      : min(8,args.batch_size),
            "things_final.up"   : min(4,args.batch_size),
            "things_final.1"    : min(8,args.batch_size),
            "things_final.2"    : min(16,args.batch_size),
            "things3"           : min(8,args.batch_size),
            "things3.1"         : min(16,args.batch_size),
            "things3.up1"       : min(8,args.batch_size),
            "things3.up"        : min(4,args.batch_size),
            "things3_clean"     : min(8,args.batch_size),
            "things3_clean.1"   : min(16,args.batch_size),
            "things3_clean.up"  : min(4,args.batch_size),
            "things3_final"     : min(8,args.batch_size),
            "things3_final.1"   : min(16,args.batch_size),
            "things3_final.up"  : min(4,args.batch_size),
            "viper"             : min(2,args.batch_size),
            "viper1"            : min(2,args.batch_size),
            "viper1.1"          : min(4,args.batch_size),
            "viper1.2"          : min(8,args.batch_size),
            "viper1.3"          : min(16,args.batch_size),
            "viper1.4"          : min(16,args.batch_size),
            "viper3"            : min(4,args.batch_size),
            "viper3.1"          : min(8,args.batch_size)
        }
    elif args.image_size is not None and args.image_size[0] <= 704 and args.image_size[1] <= 1280 :
        mini_batch_sizes = {
            "autoflow"      : min(32,args.batch_size),
            "chairs2"       : min(32,args.batch_size),
            "things"        : min(16,args.batch_size),
            "things_clean"  : min(16,args.batch_size),
            "things_final"  : min(16,args.batch_size),
            "viper"         : min(8,args.batch_size),
            "viper1"        : min(8,args.batch_size),
            "viper1.1"      : min(8,args.batch_size),
            "viper1.2"      : min(32,args.batch_size),
            "viper3"        : min(8,args.batch_size)
        }
    else:
        mini_batch_sizes = {
            "autoflow"          : min(32 if args.EC[5] <= 256 else 16,args.batch_size),
            "autoflow.up"       : min(4 if args.EC[5] >= 512 else 8,args.batch_size),
            "chairs2"           : min(32,args.batch_size),
            "chairs2.up"        : min(8 if args.EC[5] >= 512 else 16,args.batch_size),
            "things"            : min(16,args.batch_size),
            "things_clean"      : min(16,args.batch_size),
            "things_clean.up"   : min(4,args.batch_size),
            "things_final"      : min(16,args.batch_size),
            "things3"           : min(8,args.batch_size),
            "things3.1"         : min(16,args.batch_size),
            "things3.up1"       : min(8,args.batch_size),
            "things3.up"        : min(4,args.batch_size),
            "things3_clean"     : min(8,args.batch_size),
            "things3_clean.1"   : min(16,args.batch_size),
            "things3_clean.up"  : min(4,args.batch_size),
            "things3_final"     : min(8,args.batch_size),
            "things3_final.1"   : min(16,args.batch_size),
            "things3_final.up"  : min(4,args.batch_size),
            "viper"             : min(4,args.batch_size),
            "viper1"            : min(4,args.batch_size),
            "viper1.1"          : min(8,args.batch_size),
            "viper1.2"          : min(32,args.batch_size),
            "viper3"            : min(4,args.batch_size),
            "viper3.1"          : min(8,args.batch_size),
            "viper3.2"          : min(8,args.batch_size)
        }
    scales = {
        "autoflow"          : 1,
        "autoflow.up"       : 2,
        "chairs2"           : 1,
        "chairs2.up"        : 2,
        "things"            : 1,
        "things_clean"      : 1,
        "things_clean.up"   : 2,
        "things_clean.1"    : 1,
        "things_final"      : 1,
        "things_final.up"   : 1,
        "things3"           : 1,
        "things3.1"         : 1,
        "things3.up1"       : 1.5,
        #"things3.up1"       : 2,
        "things3.up"        : 2,
        "things3_clean"     : 1,
        "things3_clean.1"   : 0.5,
        "things3_clean.up"  : 2,
        "things3_final"     : 1,
        "things3_final.1"   : 0.5,
        "things3_final.up"  : 2,
        "viper"             : 2,
        "viper1"            : 1,
        "viper1.1"          : 1,
        "viper1.2"          : 1,
        "viper3"            : 1,
        "viper3.1"          : 1,
        "viper3.2"          : 1
    }
    #print('mini_batch_sizes["autoflow"]', mini_batch_sizes["autoflow"])
    #print('mini_batch_sizes["chairs2"]', mini_batch_sizes["chairs2"])
    #print('mini_batch_sizes["things_clean"]', mini_batch_sizes["things_clean"])
    #print('mini_batch_sizes["viper1"]', mini_batch_sizes["viper1"])
    train_loaders = {}
    epoch_sizes = {}
    train_loader_len = 0
    datasetlist = ["things3","things3.1","things3.up1","things3.up",
                   "things3_clean","things3_clean.1","things3_clean.up",
                   "things3_final","things3_final.1","things3_final.up",
                   "viper","viper1","viper3","viper3.1","viper3.2"]
    for i in range(len(datasetlist)):
        if datasetlist[i] in args.datasets:
            train_loaders[datasetlist[i]] = datasets.fetch_dataloader(args,
                                                               image_size      = image_sizes[datasetlist[i]],
                                                               mini_batch_size = mini_batch_sizes[datasetlist[i]],
                                                               stage           = datasetlist[i])
            dataset_len = len(train_loaders[datasetlist[i]].dataset)
            if dataset_len > args.max_dataset_len:
                dataset_len = args.max_dataset_len
            train_loader_len += dataset_len
            epoch_sizes[datasetlist[i]] = math.floor(math.floor(dataset_len/args.batch_size) \
                                   * args.batch_size/mini_batch_sizes[datasetlist[i]])
            print('=> mini_batch_sizes[',datasetlist[i],']:', mini_batch_sizes[datasetlist[i]])
            print('=> image_sizes[',datasetlist[i],']:', image_sizes[datasetlist[i]])

    #for i in range(len(args.datasets)):
    #    train_loaders[args.datasets[i]] = datasets.fetch_dataloader(args,
    #                                                                image_size      = image_sizes[args.datasets[i]],
    #                                                                mini_batch_size = mini_batch_sizes[args.datasets[i]],
    #                                                                stage           = args.datasets[i])
    #    dataset_len = len(train_loaders[args.datasets[i]].dataset)
    #    if dataset_len > args.max_dataset_len:
    #        dataset_len = args.max_dataset_len
    #    train_loader_len += dataset_len
    #    epoch_sizes[args.datasets[i]] = math.floor(math.floor(dataset_len/args.batch_size) \
    #                                    * args.batch_size/mini_batch_sizes[args.datasets[i]])

    print("=> fetching test image pairs in '{}\\{}'".format(args.data,args.testdataset))
    test_set = data_flow.__dict__[args.testdataset](
        args.data,
        transform=input_transform,
        target_transform=target_transform,
        co_transform=None #co_validate_transform
    )
    val_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False)

    #print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
    #                                                                       len(train_set),
    #                                                                       len(test_set)))
    #print('{} samples found, {} train samples and {} test samples '.format(len(val_loader.dataset)+len(train_loader.dataset),
    #                                                                       len(train_loader.dataset),
    #                                                                       len(val_loader.dataset)))
    print('{} samples found, {} train samples and {} test samples '.format(len(val_loader.dataset)+train_loader_len,
                                                                           train_loader_len,
                                                                           len(val_loader.dataset)))
    # create model
    #model = models.__dict__[args.arch](network_data,
    #                                   device           = args.device,
    #                                   use_checkpoint   = args.use_checkpoint,
    #                                   useFlow1         = args.useFlow1,
    #                                   useContext       = args.useContext,
    #                                   useResidual      = args.useResidual,
    #                                   useDenseNet      = args.useDenseNet,
    #                                   max_displacement = args.max_displacement,
    #                                   lr_finder        = args.lr_finder,
    #                                   div_flow         = args.div_flow,
    #                                   batchnorm        = args.batchnorm,
    #                                   large_motion     = args.large_motion,
    #                                   mixed_precision  = args.mixed_precision).to(args.device)
    model = models.__dict__[args.arch](network_data,
                                       args).to(args.device)

    if args.device.type == "cuda" and args.n_GPUs > 1:
        model = torch.nn.DataParallel(model).cuda()
    cudnn.enabled = True
    cudnn.benchmark = True
    
    print('Model parameters: {:,}'.format(count_parameters(model)))
    #for name, param in model.named_parameters():
    #    print('name:',name)
        #print('name:',name,' data:', param.data)
        #if param.requires_grad and param.grad is not None:
        #    print('name:',name,' data:', param.data.norm().detach().cpu().numpy())
            #print('name:',name,' data:', param.data)
            #print('name:',name)
    #for name, module in model.named_modules():
    #    #print(idx, '->', m)
    #    if isinstance(module, torch.nn.PReLU):
    #        print('name:',name,' module:',module)
    #        for pname, param in module.named_parameters():
    #            if param.requires_grad:
    #                print('pname:',pname,' norm:',param.data.norm().detach().cpu().numpy())
    #exit()
    #if args.usePReLu and args.weight_decay > 0:
    #    for name, module in model.named_modules():
    #        if isinstance(module, torch.nn.PReLU):
    #            for pname, param in module.named_parameters():
    #                if param.requires_grad:
    #                    param.requires_grad = False
                    #print('name:',name)
                    #print('pname:',pname)
                    #print('module:',module)
                    #print('param:',param)
    #for param in model.moduleExtractor.moduleOne.parameters():
    #    param.requires_grad = True    
    #for param in model.moduleExtractor.moduleTwo.parameters():
    #    param.requires_grad = True    
    #for param in model.moduleExtractor.moduleThr.parameters():
    #    param.requires_grad = False    
    #for param in model.moduleExtractor.moduleFou.parameters():
    #    param.requires_grad = False    
    #for param in model.moduleExtractor.moduleFiv.parameters():
    #    param.requires_grad = False    
    #for param in model.moduleExtractor.moduleSix.parameters():
    #    param.requires_grad = False
    #
    #for param in model.moduleOne.parameters():
    #    param.requires_grad = True    
    #for param in model.moduleTwo.parameters():
    #    param.requires_grad = True    
    #for param in model.moduleThr.parameters():
    #    param.requires_grad = False    
    #for param in model.moduleFou.parameters():
    #    param.requires_grad = False    
    #for param in model.moduleFiv.parameters():
    #    param.requires_grad = False    
    #for param in model.moduleSix.parameters():
    #    param.requires_grad = False    
    #
    #for param in model.moduleRefiner.parameters():
    #    param.requires_grad = True

    if args.Reset_Layer6:
        for layer in model.moduleExtractor.moduleSix.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            for name, module in layer.named_modules():
                if isinstance(module, torch.nn.PReLU):
                    module.init = 5
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
                    torch.nn.init.kaiming_normal_(module.weight, 0.1)
                    if module.bias is not None:
                        torch.nn.init.constant_(module.bias, 0)
        for layer in model.moduleSix.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            for name, module in layer.named_modules():
                if isinstance(module, torch.nn.PReLU):
                    module.init = 1
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
                    torch.nn.init.kaiming_normal_(module.weight, 0.1)
                    if module.bias is not None:
                        torch.nn.init.constant_(module.bias, 0)

    print('Active Model parameters: {:,}'.format(count_parameters(model)))

    assert(args.solver in ['adam','adamw','sgd'])
    #if LooseVersion(torch.__version__) < LooseVersion('1.8.0') and args.solver == 'adamw':
    if version.parse(torch.__version__) < version.Version("1.8.0") and args.solver == 'adamw':
        args.solver = 'adam'
    print('=> setting {} solver'.format(args.solver))

    if args.solver == 'adam':
        kwargs = {'lr':             args.lr if args.lr is not None else 25,
                  'weight_decay':   args.weight_decay,
                  'betas':          (args.betas[0],args.betas[1])}
        optimizer = torch.optim.Adam(model.parameters(), **kwargs)
    elif args.solver == 'adamw':
        kwargs = {'lr':             args.lr if args.lr is not None else 25,
                  'weight_decay':   args.weight_decay,
                  'betas':          (args.betas[0],args.betas[1])}
        optimizer = torch.optim.AdamW(model.parameters(), **kwargs)
    elif args.solver == 'sgd':
        kwargs = {'lr':           args.lr if args.lr is not None else 0.001,
                  'weight_decay': args.weight_decay,
                  'momentum':     args.momentum}
        optimizer = torch.optim.SGD(model.parameters(), **kwargs)

    # If this is a resume then this will be set.
    if 'optimizer_state_dict' in network_data.keys():
        optimizer.load_state_dict(network_data['optimizer_state_dict'])
        # Resolve https://github.com/pytorch/pytorch/issues/80809
        # Fixed in pytorch 1.12.1
        #optimizer.param_groups[0]['capturable'] = True

    if args.evaluate:
        eval_writer = SummaryWriter(log_dir=args.save_path / args.run_path)
        args.best_EPE, args.best_Fl, args.best_WAUC = validate(val_loader, model, 0, eval_writer, inverse_transform)
        #if LooseVersion(torch.__version__) >= LooseVersion('1.3'):
        if version.parse(torch.__version__) >= version.Version("1.3"):
            eval_writer.add_hparams({'Batch Size':      '{:1d}'.format(args.batch_size),
                                     'Test Batch Size': '{:1d}'.format(args.test_batch_size),
                                     'Optimizer':       args.solver,
                                     'Scheduler':       args.scheduler,
                                     'Loss Function':   'N/A',
                                     'Dataset':         'N/A',
                                     'Test Dataset':    args.testdataset,
                                     'div flow':        '{:1d}'.format(args.div_flow),
                                     'Epoch':           0,
                                     'Iterations':      0},
                                    metric_dict = {'Test/EPE':    args.best_EPE,
                                                   'Test/Fl':     args.best_Fl,
                                                   'Test/WAUC':   args.best_WAUC,
                                                   'Train/EPE':   0,
                                                   'Train/Fl':    0,
                                                   'Train/WAUC':  0,
                                                   'Train/Loss':  0},
                                    hparam_domain_discrete={'Batch Size':       ['1','22'],
                                                            'Test Batch Size':  ['1','6'],
                                                            'div flow':     ['1','20','256','512','1024'],
                                                            'Dataset':      ['N/A',
                                                                             'flying_chairs2',
                                                                             'flying_things_final',
                                                                             'flying_things_clean',
                                                                             'flying_things_both',
                                                                             'viper'],
                                                            'Test Dataset': ['KITTI_2012_noc',
                                                                             'KITTI_2012_occ',
                                                                             'KITTI_2015_noc',
                                                                             'KITTI_2015_occ',
                                                                             'mpi_sintel_final',
                                                                             'mpi_sintel_clean']},
                                    run_name='hparams')
        model.eval()
        for i, (inputs, target) in enumerate(val_loader):
            inputs.to(args.device)
            b, N, c, intHeight, intWidth = inputs.size()

            # Need input image resolution to always be a multiple of 64
            intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
            intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

            if intPreprocessedWidth == intWidth and intPreprocessedHeight == intHeight:
                # Faster but same memory utilization. Without detach it is slower but takes less memory.
                tensorPreprocessedFirst = inputs[:, 0, :, :, :].detach()
                tensorPreprocessedSecond = inputs[:, 1, :, :, :].detach()
            else:
                tensorPreprocessedFirst = F.interpolate(
                                            input         = inputs[:, 0, :, :, :],
                                            size          = (intPreprocessedHeight, intPreprocessedWidth),
                                            mode          = 'bilinear',
                                            align_corners = False)
                tensorPreprocessedSecond = F.interpolate(
                                            input         = inputs[:, 1, :, :, :],
                                            size          = (intPreprocessedHeight, intPreprocessedWidth),
                                            mode          = 'bilinear',
                                            align_corners = False)
            eval_writer.add_graph(model, torch.stack((tensorPreprocessedFirst, tensorPreprocessedSecond), dim=1).to(args.device))
            break
        if args.lr_finder:
            #Need to crop video so that it doesn't need to be resized
            custom_train_iter = CustomTrainIter(train_loader)
            num_iter = train_loader_len if args.epoch_size == 0 else min(train_loader_len, args.epoch_size)
            if args.lr_finder_Leslie_Smith:
                custom_val_iter = CustomValIter(loader_test)
                lr_finder = LRFinder(model, optimizer, args.loss_Flow_net, device="cuda")
                lr_finder.range_test(custom_train_iter,
                                     val_loader = custom_val_iter,
                                     end_lr     = args.max_lr,
                                     num_iter   = num_iter,
                                     step_mode  = "linear")
            else:
                lr_finder = LRFinder(model, optimizer, args.loss_Flow_net, device="cuda")
                lr_finder.range_test(custom_train_iter,
                                     end_lr   = args.max_lr,
                                     num_iter = num_iter)
        return

    assert(args.scheduler in ['steplr','OneCycle','onecyclelr','cycliclr','multisteplr','lr_finder'])
    if args.scheduler == 'multisteplr':
        kwargs = {'milestones': args.milestones,
                  'gamma':      args.gamma}
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif args.scheduler == 'steplr':
        kwargs = {'step_size': args.lr_decay,
                  'gamma':     args.gamma}
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif args.scheduler == 'onecyclelr':
        div_factor = args.max_lr / args.lr
        kwargs = {'max_lr':          args.max_lr,
                  'epochs':          args.epochs,
                  'div_factor':      div_factor,
                  'steps_per_epoch': train_loader_len}
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
    elif args.scheduler == 'OneCycle':
        div_factor = args.max_lr / args.lr if args.lr is not None else 25
        #div_factor = 25
        OC_total_steps = int(args.num_steps / args.step_size) + 100
        kwargs = {'max_lr':          args.max_lr,
                  'total_steps':     OC_total_steps,
                  'pct_start':       args.pct_start,
                  'cycle_momentum':  False,
                  'anneal_strategy': args.anneal_strategy,
                  'div_factor':      div_factor,
                  'steps_per_epoch': train_loader_len}
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
    elif args.scheduler == 'cycliclr':
        args.epoch_size = int(train_loader_len/2.) * 2 if args.epoch_size == 0 else min(train_loader_len, args.epoch_size)
        args.step_size_up = int(args.epoch_size/2) if args.step_size_up == 0 else args.step_size_up
        kwargs = {'base_lr':        args.lr,
                  'max_lr':         args.max_lr,
                  'cycle_momentum': False if args.solver != 'sgd' else True, #needs to be False for Adam/AdamW
                  'step_size_up':   int(args.step_size_up/args.step_size),
                  'mode':           args.CyclicLR_mode,
                  'gamma':          args.CyclicLR_gamma}
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **kwargs)
    elif args.scheduler == 'lr_finder':
        # This is just for testing. Disable when not needed
        #torch.autograd.set_detect_anomaly(True)

        #Need to crop video so that it doesn't need to be resized
        #custom_train_iter = CustomTrainIter(train_loader)
        num_iter = train_loader_len if args.epoch_size == 0 else min(train_loader_len, args.epoch_size)
        fig_kw = {'figsize': [19.2,10.8]}
        #plt.rcParams.update({'font.size': 20})
        plt.ticklabel_format(axis="x",style="sci",scilimits=(0,0))
        fig, ax = plt.subplots(**fig_kw)
        if args.lr_finder_Leslie_Smith:
            custom_val_iter = CustomValIter(loader_test)
            lr_finder = LRFinder(model, optimizer, args.loss_Flow_net, device="cuda")
            lr_finder.range_test(train_loaders[args.datasets[0]], #train_loader, #custom_train_iter,
                                 #val_loader = custom_val_iter,
                                 end_lr     = args.max_lr,
                                 num_iter   = num_iter,
                                 step_mode  = "linear")
        else:
            lr_finder = LRFinder(model, optimizer, args.loss_Flow_net, device="cuda")
            lr_finder.range_test(train_loaders[args.datasets[0]], #train_loader, #custom_train_iter,
                                 end_lr   = args.max_lr,
                                 num_iter = num_iter)
        lr_finder.plot(ax=ax)
        plt.grid(True)
        fig.tight_layout()
        eval_writer = SummaryWriter(log_dir=args.save_path / args.run_path / 'eval')
        eval_writer.add_figure('lr_finder Length: {}/Loss: {}/div_flow: {}'.format(train_loader_len,args.loss, args.div_flow),
                               fig,
                               global_step=0)
        plt.close()
        #print('Type lr_finder.history', type(lr_finder.history))
        #print(lr_finder.history)
        for i in range(len(lr_finder.history['lr'])):
            eval_writer.add_scalars('lr_finder',
                                    {'lr': lr_finder.history['lr'][i],
                                     'loss': lr_finder.history['loss'][i]},
                                    i)

        #eval_writer.add_scalars('lr_finder', lr_finder.history, 0)
        #eval_writer.add_text('lr_finder/Loss: {}/div_flow: {}'.format(args.loss, args.div_flow), 'This is the best loss {:.6f}'.format(lr_finder.best_loss), 0)
        return

    # If this is a resume then this will be set.
    if 'scheduler_state_dict' in network_data.keys():
        scheduler.load_state_dict(network_data['scheduler_state_dict'])

    #train_writer_iterations = SummaryWriter(log_dir=args.save_path / args.run_path / 'iterations', purge_step=args.n_iter)
    train_writer = SummaryWriter(log_dir=args.save_path / args.run_path, filename_suffix='train', purge_step=int(args.total_steps+1))

    if args.start_epoch > args.epochs:
        return

    args.train_loss = 0
    args.train_EPE = 0
    args.train_Fl = 0
    args.train_WAUC = 0
    args.test_EPE = 0
    args.test_Fl = 0
    args.test_WAUC = 0
    args.test_epoch_size = len(val_loader) if args.test_epoch_size == 0 else args.test_epoch_size

    try:
        tqdm_main = tqdm(total=args.num_steps, position=0, initial=int(args.total_steps))
        tqdm_train0 = tqdm(total=args.epoch_size, position=1, desc='Train')
        tqdm_train1 = tqdm(total=args.epoch_size, position=2, bar_format='{desc}', desc='Waiting for first train batch')
        tqdm_train2 = tqdm(total=args.epoch_size, position=3, bar_format='{desc}', desc='Waiting for first train batch')
        tqdm_test0 = tqdm(total=args.test_epoch_size, position=4, desc='Test')
        tqdm_test1 = tqdm(total=args.test_epoch_size, position=5, bar_format='{desc}', desc='Waiting for first test batch')
        tqdm_test2 = tqdm(total=args.test_epoch_size, position=6, bar_format='{desc}', desc='Waiting for first test batch')

        # This is just for testing. Disable when not needed
        #torch.autograd.set_detect_anomaly(True)
        
        #model = torch.compile(model,
        #                      dynamic=True
        #                      #,mode='max-autotune'
        #                      )
        
        while args.total_steps < args.max_num_steps:
            #if args.total_steps < 120_000
            #for name, param in model.named_parameters():
            #    if param.requires_grad:
            #        print name, param.data
            #        exit()

            args.mini_batch_size = mini_batch_sizes[args.datasets[0]]
            args.accumulation_steps = int(args.batch_size / args.mini_batch_size)
            args.epoch_size = len(train_loaders[args.datasets[0]]) if epoch_sizes[args.datasets[0]] == 0 else epoch_sizes[args.datasets[0]]
            args.scale = scales[args.datasets[0]]
            tqdm_train0.reset(total=args.epoch_size)
            tqdm_train1.reset(total=args.epoch_size)
            tqdm_train2.reset(total=args.epoch_size)
            # train for one epoch
            #train(train_loader, val_loader, model, optimizer, scheduler, train_writer, inverse_transform)
            train(train_loaders[args.datasets[0]], val_loader, model, optimizer, scheduler, train_writer, inverse_transform, args.datasets[0])
            args.train_loader_start = 0

            if args.scheduler in ['steplr','multisteplr']:
                scheduler.step()

            #Rotate list left
            args.datasets.append(args.datasets.pop(0))
            
            args.flow_EPE_Avg = flow_EPEs.avg
            args.flow_EPE_Count = flow_EPEs.count
            args.flow_F1_Avg = flow_Fls.avg
            args.flow_F1_Count = flow_Fls.count
            args.flow_WAUC_Avg = flow_WAUCs.avg
            args.flow_WAUC_Count = flow_WAUCs.count
            args.flow1_WAUC_Avg = flow1_WAUC.avg
            args.flow2_WAUC_Avg = flow2_WAUC.avg
            args.flow3_WAUC_Avg = flow3_WAUC.avg
            args.flow4_WAUC_Avg = flow4_WAUC.avg
            args.flow5_WAUC_Avg = flow5_WAUC.avg
            args.flow6_WAUC_Avg = flow6_WAUC.avg
            args.flow1_WAUC_Count = flow1_WAUC.count
            args.flow2_WAUC_Count = flow2_WAUC.count
            args.flow3_WAUC_Count = flow3_WAUC.count
            args.flow4_WAUC_Count = flow4_WAUC.count
            args.flow5_WAUC_Count = flow5_WAUC.count
            args.flow6_WAUC_Count = flow6_WAUC.count
            args.flow1_EPE_Avg = flow1_EPE.avg
            args.flow2_EPE_Avg = flow2_EPE.avg
            args.flow3_EPE_Avg = flow3_EPE.avg
            args.flow4_EPE_Avg = flow4_EPE.avg
            args.flow5_EPE_Avg = flow5_EPE.avg
            args.flow6_EPE_Avg = flow6_EPE.avg
            args.flow1_EPE_Count = flow1_EPE.count
            args.flow2_EPE_Count = flow2_EPE.count
            args.flow3_EPE_Count = flow3_EPE.count
            args.flow4_EPE_Count = flow4_EPE.count
            args.flow5_EPE_Count = flow5_EPE.count
            args.flow6_EPE_Count = flow6_EPE.count
            
            
            

            model_state_dict = {}
            model_state_dict['args'] = args
            model_state_dict['state_dict'] = model.state_dict()
            model_state_dict['optimizer_state_dict'] = optimizer.state_dict()
            scheduler_dict = scheduler.state_dict()
            if '_scale_fn_ref' in scheduler_dict.keys():
                scheduler_dict.pop('_scale_fn_ref')
            model_state_dict['scheduler_state_dict'] = scheduler_dict
            model_state_dict['rng_state'] = torch.get_rng_state()
            model_state_dict['random_state'] = random.getstate()
            model_state_dict['numpy_random_state'] = np.random.get_state()
            #model_state_dict = {'state_dict': model.module.state_dict(),
            #                    'optimizer_state_dict': optimizer.state_dict(),
            #                    'scheduler_state_dict': scheduler.state_dict(),
            #                    'args': args,
            #                    'rng_state': rng_state,
            #                    'random_state': random_state,
            #                    'numpy_random_state': numpy_random_state}
            torch.save(model_state_dict, args.save_path / args.run_path / 'checkpoint.pt')
            if args.total_steps >= args.max_num_steps:
                break
            if terminate:
                break

        tqdm_main.close()
        tqdm_train0.close()
        tqdm_train1.close()
        tqdm_train2.close()
        tqdm_test0.close()
        tqdm_test1.close()
        tqdm_test2.close()

        # Save the final stats as a hparams even if there is a crash
        if args.total_steps >= args.num_steps:
            train_writer.add_hparams({'Batch Size':      '{:1d}'.format(args.batch_size),
                                      'Test Batch Size': '{:1d}'.format(args.test_batch_size),
                                      'Parameters':      count_parameters(model),
                                      'Optimizer':       args.solver,
                                      'Scheduler':       args.scheduler,
                                      'Loss Function':   args.loss,
                                      'Dataset':         ','.join(args.datasets),
                                      'Test Dataset':    args.testdataset,
                                      'div flow':        '{:1d}'.format(args.div_flow),
                                      'Epoch':           args.start_epoch,
                                      'Iterations':      args.num_steps},
                                     metric_dict = {'Test/EPE':    args.best_EPE,
                                                    'Test/Fl':     args.best_Fl,
                                                    'Test/WAUC':   args.best_WAUC},
                                     #hparam_domain_discrete={'Batch Size':       ['1','16','22'],
                                     #                        'Test Batch Size':  ['1','6'],
                                     #                        'div flow':     ['1','20','256','512','1024'],
                                     #                        'Dataset':      ['N/A',
                                     #                                         'flying_chairs2',
                                     #                                         'flying_things_final',
                                     #                                         'flying_things_clean',
                                     #                                         'flying_things_both',
                                     #                                         'AutoFlow'],
                                     #                        'Test Dataset': ['KITTI_2012_noc',
                                     #                                         'KITTI_2012_occ',
                                     #                                         'KITTI_2015_noc',
                                     #                                         'KITTI_2015_occ',
                                     #                                         'mpi_sintel_final',
                                     #                                         'mpi_sintel_clean']},
                                     run_name='hparams')

    except:
        tqdm_main.close()
        tqdm_train0.close()
        tqdm_train1.close()
        tqdm_train2.close()
        tqdm_test0.close()
        tqdm_test1.close()
        tqdm_test2.close()
        print()
        raise



def train(train_loader, val_loader, model, optimizer, scheduler, train_writer, inverse_transform, dataset_name):
    global args, tqdm_main, tqdm_train0, tqdm_train1, tqdm_train2
    realEPE = utils.EPE()
    realWAUC = utils.WAUC()
    realFl = utils.Fl_KITTI_2015(use_mask=True)

    # Recreate scaler every epoch.
    scaler = GradScaler(enabled=args.mixed_precision)
    lr = scheduler.get_last_lr()[0]

    # switch to train mode
    model.train()
    torch.cuda.empty_cache()
    model.zero_grad(set_to_none=True)

    frames_total = len(train_loader.dataset) * 2
    frames_completed = 0
    end = time.time()
    flowGT_mean = AverageMeter()
    flow1_mean = AverageMeter()
    flow2_mean = AverageMeter()
    flow3_mean = AverageMeter()
    flow4_mean = AverageMeter()
    flow5_mean = AverageMeter()
    flow6_mean = AverageMeter()
    flowGT_max = 0
    flow1_max = 0
    flow2_max = 0
    flow3_max = 0
    flow4_max = 0
    flow5_max = 0
    flow6_max = 0
    grad_max_mean = ''
    lastWAUC = None
    currentWAUC = 0
    batch_norm = []
    batch_norm_clipped = []

    for i, (inputs, target) in enumerate(train_loader):
        if i >= args.epoch_size:
            tqdm_train1.set_description('Dataset {0}  Train: EPE {1}  Fl {2}  WAUC {3}  LR {4:.3e} - exceeded {5}'.format(
                                        dataset_name,
                                        flow_EPEs,
                                        flow_Fls,
                                        flow_WAUCs,
                                        lr,
                                        args.epoch_size))
            break
        args.total_steps += args.step_size * 2/ args.accumulation_steps
        # measure data loading time
        data_time.update(time.time() - end)
        B2, N2, C2, intHeight2, intWidth2 = target[:,:,:2,:,:].size()
        frames_completed += B2 * 2
        infinity = max(intHeight2,intWidth2)
        maskB = target[:,0,2,:,:].to(args.device)
        maskF = target[:,1,2,:,:].to(args.device)
        #print('maskB',maskB)
        if not args.scale_output:
            target = target.to(args.device)
            target[:,:,0,:,:] = target[:,:,0,:,:].nan_to_num(0,intWidth2,-intWidth2)
            target[:,:,1,:,:] = target[:,:,1,:,:].nan_to_num(0,intHeight2,-intHeight2)
            #for i in range(B2):
            #    maskB[i,:,:] = data_flow.listdataset.get_gt_correspondence_mask(target[i,0,:2,:,:])
            #    maskF[i,:,:] = data_flow.listdataset.get_gt_correspondence_mask(target[i,1,:2,:,:])
            maskB = data_flow.listdataset.get_gt_correspondence_mask(target[:,0,:2,:,:])
            maskF = data_flow.listdataset.get_gt_correspondence_mask(target[:,1,:2,:,:])
            target[:,0,:2,:,:] *= maskB[:,None,:,:]
            target[:,1,:2,:,:] *= maskF[:,None,:,:]
            target[:,0,2,:,:] = maskB
            target[:,1,2,:,:] = maskF
        if args.clamp is not None:
            target = torch.clamp(target, min=-1 * args.clamp, max=args.clamp)
        inputs = inputs.to(args.device)
        #print()
        #print()
        #print()
        #print()
        #print()
        #print()
        #print()
        #print()
        #print('target[:,:,:2,:,:].size()',target[:,:,:2,:,:].size())
        #print('target[:,0,2,:,:].size()',target[:,0,2,:,:].size())
        #print('target[:,0,:2,:,:].max()',target[:,0,:2,:,:].max())
        #print('target[:,1,:2,:,:].max()',target[:,1,:2,:,:].max())
        #target[:,0,:2,:,:] *= maskB[:,None,:,:]
        #target[:,1,:2,:,:] *= maskF[:,None,:,:]
        #print('target[:,0,:2,:,:].max()',target[:,0,:2,:,:].max())
        #print('target[:,1,:2,:,:].max()',target[:,1,:2,:,:].max())
        #print('maskB.sum()',maskB.sum())
        #print('maskF.sum()',maskF.sum())
        #print()
        #print()
        #print()
        #print()
        #print()
        #print()
        #print()
        #print()
        #exit()

        #print('inputs[:,0,:,:,:]', inputs[:,0,:,:,:].size())
        #print('inputs[:,1,:,:,:]', inputs[:,1,:,:,:].size())
        #print('inputs[:,2,:,:,:]', inputs[:,2,:,:,:].size())
        B, N, C, intHeight, intWidth = inputs.size()
        
        
        #print('Input height: ',intHeight,', width: ',intWidth)
        intPreprocessedWidth = int(math.floor(math.ceil(intWidth * args.scale / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight * args.scale / 64.0) * 64.0))
        if intWidth > intPreprocessedWidth or intHeight > intPreprocessedHeight:
            tensorFirst = F.interpolate(
                            input         = inputs[:,0,:,:,:].float(),
                            size          = (intPreprocessedHeight, intPreprocessedWidth),
                            mode          = 'bilinear',
                            align_corners = False,
                            antialias     = True)
            tensorSecond = F.interpolate(
                            input         = inputs[:,1,:,:,:].float(),
                            size          = (intPreprocessedHeight, intPreprocessedWidth),
                            mode          = 'bilinear',
                            align_corners = False,
                            antialias     = True)
            tensorThird = F.interpolate(
                            input         = inputs[:,2,:,:,:].float(),
                            size          = (intPreprocessedHeight, intPreprocessedWidth),
                            mode          = 'bilinear',
                            align_corners = False,
                            antialias     = True)

            inputs = torch.stack((tensorFirst, tensorSecond, tensorThird), dim=1)
        
        elif intWidth < intPreprocessedWidth or intHeight < intPreprocessedHeight:
            tensorFirst = F.interpolate(
                            input         = inputs[:,0,:,:,:].float(),
                            size          = (intPreprocessedHeight, intPreprocessedWidth),
                            mode          = 'bilinear',
                            align_corners = False,
                            antialias     = True)
            tensorSecond = F.interpolate(
                            input         = inputs[:,1,:,:,:].float(),
                            size          = (intPreprocessedHeight, intPreprocessedWidth),
                            mode          = 'bilinear',
                            align_corners = False,
                            antialias     = True)
            tensorThird = F.interpolate(
                            input         = inputs[:,2,:,:,:].float(),
                            size          = (intPreprocessedHeight, intPreprocessedWidth),
                            mode          = 'bilinear',
                            align_corners = False,
                            antialias     = True)

            inputs = torch.stack((tensorFirst, tensorSecond, tensorThird), dim=1)
        
        with torch.cuda.amp.autocast():
            if intWidth > intPreprocessedWidth or intHeight > intPreprocessedHeight:
                OutputsUpB, OutputsUpF = model(inputs)
                outputsB = [None,None,None,None,None,None]
                outputsF = [None,None,None,None,None,None]
                scale_factor = [1,1,4,8,16,32,64]
                for i in range(len(OutputsUpF)):
                    if OutputsUpB[i] is not None:
                        Bf, Cf, intHeightf, intWidthf = OutputsUpB[i].size()
                        intHeightf = int(intHeight/scale_factor[i])
                        intWidthf = int(intWidth/scale_factor[i])
                        if True:
                        #if intHeightf > intHeight or intWidthf > intWidth:
                            outputsB[i] = F.interpolate(
                                            input         = OutputsUpB[i].float(),
                                            #size          = (intHeight, intWidth),
                                            size          = (intHeightf, intWidthf),
                                            mode          = 'bilinear',
                                            align_corners = False,
                                            antialias     = True)
                        else:
                            outputsB[i] = OutputsUpB[i]
                        outputsB[i][:, 0, :, :] *= intWidth / intPreprocessedWidth
                        outputsB[i][:, 1, :, :] *= intHeight / intPreprocessedHeight
                    if OutputsUpF[i] is not None:
                        Bf, Cf, intHeightf, intWidthf = OutputsUpF[i].size()
                        intHeightf = int(intHeight/scale_factor[i])
                        intWidthf = int(intWidth/scale_factor[i])
                        if True:
                        #if intHeightf > intHeight or intWidthf > intWidth:
                            outputsF[i] = F.interpolate(
                                            input         = OutputsUpF[i].float(),
                                            #size          = (intHeight, intWidth),
                                            size          = (intHeightf, intWidthf),
                                            mode          = 'bilinear',
                                            align_corners = False,
                                            antialias     = True)
                        else:
                            outputsF[i] = OutputsUpB[i]
                        outputsF[i][:, 0, :, :] *= intWidth / intPreprocessedWidth
                        outputsF[i][:, 1, :, :] *= intHeight / intPreprocessedHeight
                    if args.clamp is not None and outputsF[i] is not None:
                        outputsB[i] = torch.clamp(outputsB[i], min=-1 * args.clamp, max=args.clamp)
                        outputsF[i] = torch.clamp(outputsF[i], min=-1 * args.clamp, max=args.clamp)
            elif intWidth < intPreprocessedWidth or intHeight < intPreprocessedHeight:
                OutputsUpB, OutputsUpF = model(inputs)
                outputsB = [None,None,None,None,None,None]
                outputsF = [None,None,None,None,None,None]
                scale_factor = [1,1,4,8,16,32,64]
                for i in range(len(OutputsUpF)):
                    if OutputsUpB[i] is not None:
                        Bf, Cf, intHeightf, intWidthf = OutputsUpB[i].size()
                        intHeightf = int(intHeight/scale_factor[i])
                        intWidthf = int(intWidth/scale_factor[i])
                        if True:
                        #if intHeightf > intHeight or intWidthf > intWidth:
                            outputsB[i] = F.interpolate(
                                            input         = OutputsUpB[i].float(),
                                            #size          = (intHeight, intWidth),
                                            size          = (intHeightf, intWidthf),
                                            mode          = 'bilinear',
                                            align_corners = False,
                                            antialias     = True)
                        else:
                            outputsB[i] = OutputsUpB[i].float()
                        outputsB[i][:, 0, :, :] *= intWidth / intPreprocessedWidth
                        outputsB[i][:, 1, :, :] *= intHeight / intPreprocessedHeight
                    if OutputsUpF[i] is not None:
                        Bf, Cf, intHeightf, intWidthf = OutputsUpF[i].size()
                        intHeightf = int(intHeight/scale_factor[i])
                        intWidthf = int(intWidth/scale_factor[i])
                        if True:
                        #if intHeightf > intHeight or intWidthf > intWidth:
                            outputsF[i] = F.interpolate(
                                            input         = OutputsUpF[i].float(),
                                            #size          = (intHeight, intWidth),
                                            size          = (intHeightf, intWidthf),
                                            mode          = 'bilinear',
                                            align_corners = False,
                                            antialias     = True)
                        else:
                            outputsF[i] = OutputsUpB[i].float()
                        outputsF[i][:, 0, :, :] *= intWidth / intPreprocessedWidth
                        outputsF[i][:, 1, :, :] *= intHeight / intPreprocessedHeight
                    if args.clamp is not None and outputsF[i] is not None:
                        outputsB[i] = torch.clamp(outputsB[i], min=-1 * args.clamp, max=args.clamp)
                        outputsF[i] = torch.clamp(outputsF[i], min=-1 * args.clamp, max=args.clamp)
            else:
                outputsB, outputsF = model(inputs)
                if args.clamp is not None:
                    for i in range(len(outputsF)):
                        if outputsF[i] is not None:
                            outputsB[i] = torch.clamp(outputsB[i], min=-1 * args.clamp, max=args.clamp)
                            outputsF[i] = torch.clamp(outputsF[i], min=-1 * args.clamp, max=args.clamp)


            if args.split_losses:
                loss_totalB, loss_EPEs, loss_WAUCl = args.loss_Flow_net(outputsB, target[:,0,:,:,:])
              # Without .detach() there is a memory leak. Don't need grad on these
                if loss_WAUCl[0] is not None:
                    flow1_WAUC.update(100 - loss_WAUCl[0].detach(), B)
                    flow1_EPE.update(loss_EPEs[0].detach(), B)
                    loss_EPE = loss_EPEs[0].detach().cpu().numpy()
                    accuracy_WAUC = 100 - loss_WAUCl[0].detach()
                else:
                    loss_EPE = loss_EPEs[1].detach().cpu().numpy()
                    accuracy_WAUC = 100 - loss_WAUCl[1].detach()
                flow2_WAUC.update(100 - loss_WAUCl[1].detach(), B)
                flow3_WAUC.update(100 - loss_WAUCl[2].detach(), B)
                flow4_WAUC.update(100 - loss_WAUCl[3].detach(), B)
                flow5_WAUC.update(100 - loss_WAUCl[4].detach(), B)
                flow6_WAUC.update(100 - loss_WAUCl[5].detach(), B)
                flow2_EPE.update(loss_EPEs[1].detach(), B)
                flow3_EPE.update(loss_EPEs[2].detach(), B)
                flow4_EPE.update(loss_EPEs[3].detach(), B)
                flow5_EPE.update(loss_EPEs[4].detach(), B)
                flow6_EPE.update(loss_EPEs[5].detach(), B)

                loss_totalF, loss_EPEs, loss_WAUCl = args.loss_Flow_net(outputsF, target[:,1,:,:,:])
              # Without .detach() there is a memory leak. Don't need grad on these
                if loss_WAUCl[0] is not None:
                    flow1_WAUC.update(100 - loss_WAUCl[0].detach(), B)
                    flow1_EPE.update(loss_EPEs[0].detach(), B)
                    loss_EPE += loss_EPEs[0].detach().cpu().numpy()
                    accuracy_WAUC += 100 - loss_WAUCl[0].detach()
                else:
                    loss_EPE += loss_EPEs[1].detach().cpu().numpy()
                    accuracy_WAUC += 100 - loss_WAUCl[1].detach()
                loss_EPE /= 2
                accuracy_WAUC /= 2
                flow2_WAUC.update(100 - loss_WAUCl[1].detach(), B)
                flow3_WAUC.update(100 - loss_WAUCl[2].detach(), B)
                flow4_WAUC.update(100 - loss_WAUCl[3].detach(), B)
                flow5_WAUC.update(100 - loss_WAUCl[4].detach(), B)
                flow6_WAUC.update(100 - loss_WAUCl[5].detach(), B)
                flow2_EPE.update(loss_EPEs[1].detach(), B)
                flow3_EPE.update(loss_EPEs[2].detach(), B)
                flow4_EPE.update(loss_EPEs[3].detach(), B)
                flow5_EPE.update(loss_EPEs[4].detach(), B)
                flow6_EPE.update(loss_EPEs[5].detach(), B)
            else:
                loss_totalB = args.loss_Flow_net(outputsB, target[:,0,:,:,:])
                loss_totalF = args.loss_Flow_net(outputsF, target[:,1,:,:,:])
        
        loss_total = loss_totalF + loss_totalB
        loss_total /= args.accumulation_steps
        scaler.scale(loss_total).backward()
        # record loss and EPE
        flowGT_max = max(flowGT_max, target[:,:,:2,:,:].detach().abs().max())
        
        if loss_WAUCl[0] is not None:
            flow1_max = max(flow1_max, outputsF[0].detach().abs().max(), outputsB[0].detach().abs().max())
            flow1_test = (((outputsF[0].detach() * maskF[:,None,:,:]).abs().sum() + (outputsB[0].detach() * maskB[:,None,:,:]).abs().sum())) / (C * (maskF.sum() + maskB.sum()))
            flow1_mean.update(flow1_test, B2)
        flow2_max = max(flow2_max, outputsF[1].detach().abs().max(), outputsB[1].detach().abs().max())
        flow3_max = max(flow3_max, outputsF[2].detach().abs().max(), outputsB[2].detach().abs().max())
        flow4_max = max(flow4_max, outputsF[3].detach().abs().max(), outputsB[3].detach().abs().max())
        flow5_max = max(flow5_max, outputsF[4].detach().abs().max(), outputsB[4].detach().abs().max())
        flow6_max = max(flow6_max, outputsF[5].detach().abs().max(), outputsB[5].detach().abs().max())
        flowGT_mean.update(target[:,:,:2,:,:].detach().abs().sum() / (B2 * N2 * C2 * intHeight2 * intWidth2),B2)
        flow2_test = (((outputsF[1].detach().float().nan_to_num(0,infinity,-infinity) * maskF[:,None,:,:]).abs().sum().float().nan_to_num(0,infinity,-infinity) + (outputsB[1].detach() * maskB[:,None,:,:]).abs().sum()))
        flow3_test = (outputsF[2].detach().float().nan_to_num(0,infinity,-infinity).abs().sum() + outputsB[2].detach().float().nan_to_num(0,infinity,-infinity).abs().sum())
        flow4_test = (outputsF[3].detach().float().nan_to_num(0,infinity,-infinity).abs().sum() + outputsB[3].detach().float().nan_to_num(0,infinity,-infinity).abs().sum())
        flow5_test = (outputsF[4].detach().float().nan_to_num(0,infinity,-infinity).abs().sum() + outputsB[4].detach().float().nan_to_num(0,infinity,-infinity).abs().sum())
        flow6_test = (outputsF[5].detach().float().nan_to_num(0,infinity,-infinity).abs().sum() + outputsB[5].detach().float().nan_to_num(0,infinity,-infinity).abs().sum())
        flow2_mean.update(flow2_test/ (B2 * N2 * C2 * outputsF[1].size(2) * outputsF[1].size(3)), B2)
        flow3_mean.update(flow3_test/ (B2 * N2 * C2 * outputsF[2].size(2) * outputsF[2].size(3)), B2)
        flow4_mean.update(flow4_test/ (B2 * N2 * C2 * outputsF[3].size(2) * outputsF[3].size(3)), B2)
        flow5_mean.update(flow5_test/ (B2 * N2 * C2 * outputsF[4].size(2) * outputsF[4].size(3)), B2)
        flow6_mean.update(flow6_test/ (B2 * N2 * C2 * outputsF[5].size(2) * outputsF[5].size(3)), B2)

        lr = scheduler.get_last_lr()[0]

        if not args.split_losses:
            if loss_WAUCl[0] is not None:
                loss_EPE = (realEPE(outputsF[0].detach(), target[:,1,:,:,:]) + realEPE(outputsB[0].detach(), target[:,0,:,:,:])).cpu().numpy() / 2
                accuracy_WAUC = (realWAUC(outputsF[0].detach(), target[:,1,:,:,:]) + realWAUC(outputsB[0].detach(), target[:,0,:,:,:])) / 2
            else:
                loss_EPE = (realEPE(outputsF[1].detach(), target[:,1,:,:,:]) + realEPE(outputsB[1].detach(), target[:,0,:,:,:])).cpu().numpy() / 2
                accuracy_WAUC = (realWAUC(outputsF[1].detach(), target[:,1,:,:,:]) + realWAUC(outputsB[1].detach(), target[:,0,:,:,:])) / 2
        elif args.loss.find('mEPE1') > -1:
            if loss_WAUCl[0] is not None:
                loss_EPE = (realEPE(outputsF[0].detach(), target[:,1,:,:,:]) + realEPE(outputsB[0].detach(), target[:,0,:,:,:])).cpu().numpy() / 2
            else:
                loss_EPE = (realEPE(outputsF[1].detach(), target[:,1,:,:,:]) + realEPE(outputsB[1].detach(), target[:,0,:,:,:])).cpu().numpy() / 2
        if loss_WAUCl[0] is not None:
            loss_Fl = (realFl(outputsF[0].detach(), target[:,1,:,:,:]) + realFl(outputsB[0].detach(), target[:,0,:,:,:])) / 2
        else:
            loss_Fl = (realFl(outputsF[1].detach(), target[:,1,:,:,:]) + realFl(outputsB[1].detach(), target[:,0,:,:,:])) / 2
        flow_EPEs.update(loss_EPE, B)
        flow_Fls.update(loss_Fl, (maskF.sum() + maskB.sum()).item())
        flow_WAUCs.update(accuracy_WAUC, (maskF.sum() + maskB.sum()).item())
        #flow_WAUCs.update(accuracy_WAUC, target.size(0))
        learningRates.update(lr, B)
        tqdm_train1.set_description('Dataset {0}  Train: EPE {1}  Fl {2}  WAUC {3}  LR {4:.3e}'.format(
                                    dataset_name,
                                    flow_EPEs,
                                    flow_Fls,
                                    flow_WAUCs,
                                    lr) + grad_max_mean)
        if args.total_steps % args.step_size == 0:
            scaler.unscale_(optimizer)
           # calculate gradient norms
            for name, p in model.named_parameters():
                #if p.requires_grad and (".weight" in name or ".bias" in name) and p.grad is not None and not p.grad.norm().isfinite():
                #if p.requires_grad and p.grad is not None and not p.grad.norm().isfinite():
                ## Because of error_if_nonfinite for clip_grad_norm_ it will exit when this isn't finite
                ## So we will print out which modules aren't finite so we know where the problem exists.
                #    print('Grad Not Finite: ',name)
                if p.requires_grad and p.grad is not None and "weight" in name :
                    norm_grad = p.grad.norm().detach().cpu().numpy()
                    batch_norm.append(norm_grad)
            if args.clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip, error_if_nonfinite=True)
               # calculate gradient norms
                for name, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None and ".weight" in name :
                        norm_grad = p.grad.norm().detach().cpu().numpy()
                        batch_norm_clipped.append(norm_grad)
            grad_max_mean = '  [grad:{:.2f}/clipped:{:.2e}]'.format(np.mean(batch_norm),
                                                                    np.mean(batch_norm_clipped))
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            skip_lr_sched = (scale > scaler.get_scale())
            
            if not skip_lr_sched:
                scheduler.step()
                #if args.step_size / args.accumulation_steps == 1:
                if args.batch_size == args.mini_batch_size:
                    scheduler.step()

            model.zero_grad(set_to_none=True)                # Reset gradients tensors
            
            if args.total_steps % args.print_freq  == 0:
                train_writer.add_scalar('Train/EPE',              flow_EPEs.avg,               global_step = args.total_steps, new_style = True)
                train_writer.add_scalar('Train/Fl',               flow_Fls.avg,                global_step = args.total_steps, new_style = True)
                train_writer.add_scalar('Train/WAUC',             flow_WAUCs.avg,              global_step = args.total_steps, new_style = True)
                train_writer.add_scalar('Train/lr',               learningRates.avg,           global_step = args.total_steps, new_style = True)
                train_writer.add_scalar('Train/Gradient_Norm',    np.mean(batch_norm),         global_step = args.total_steps, new_style = True)
                train_writer.add_scalar('Train/Gradient_Clipped', np.mean(batch_norm_clipped), global_step = args.total_steps, new_style = True)
                batch_norm = []
                batch_norm_clipped = []
                flow_EPEs.reset()
                flow_Fls.reset()
                flow_WAUCs.reset()
                learningRates.reset()
            if (args.total_steps) % args.val_freq  == 0:
                if args.split_losses:
                    if loss_WAUCl[0] is not None:
                        currentWAUC = flow1_WAUC.avg
                        train_writer.add_scalars('split_WAUC',
                                                 {'flow1': flow1_WAUC.avg,
                                                  'flow2': flow2_WAUC.avg,
                                                  'flow3': flow3_WAUC.avg,
                                                  'flow4': flow4_WAUC.avg,
                                                  'flow5': flow5_WAUC.avg,
                                                  'flow6': flow6_WAUC.avg},
                                                 args.total_steps)
                        train_writer.add_scalars('split_EPE',
                                                 {'flow1': flow1_EPE.avg,
                                                  'flow2': flow2_EPE.avg,
                                                  'flow3': flow3_EPE.avg,
                                                  'flow4': flow4_EPE.avg,
                                                  'flow5': flow5_EPE.avg,
                                                  'flow6': flow6_EPE.avg},
                                                 args.total_steps)
                        flow1_WAUC.reset()
                        flow1_EPE.reset()
                    else:
                        currentWAUC = flow2_WAUC.avg
                        train_writer.add_scalars('split_WAUC',
                                                 {'flow2': flow2_WAUC.avg,
                                                  'flow3': flow3_WAUC.avg,
                                                  'flow4': flow4_WAUC.avg,
                                                  'flow5': flow5_WAUC.avg,
                                                  'flow6': flow6_WAUC.avg},
                                                 args.total_steps)
                        train_writer.add_scalars('split_EPE',
                                                 {'flow2': flow2_EPE.avg,
                                                  'flow3': flow3_EPE.avg,
                                                  'flow4': flow4_EPE.avg,
                                                  'flow5': flow5_EPE.avg,
                                                  'flow6': flow6_EPE.avg},
                                                 args.total_steps)
                    flow2_WAUC.reset()
                    flow3_WAUC.reset()
                    flow4_WAUC.reset()
                    flow5_WAUC.reset()
                    flow6_WAUC.reset()
                    flow2_EPE.reset()
                    flow3_EPE.reset()
                    flow4_EPE.reset()
                    flow5_EPE.reset()
                    flow6_EPE.reset()
                    
                    if lastWAUC is None or (lastWAUC - .1 * lastWAUC) < currentWAUC < (lastWAUC + .1 * lastWAUC):
                        lastWAUC = currentWAUC
                    else:
                        raise Exception("WAUC deviation is greater than 10%")
                #args.train_loader_start = i
                batch_time.update(time.time() - end)
                end = time.time()
                with torch.no_grad():
                    validate(val_loader, model, train_writer, inverse_transform)
                model.train()
            #if args.step_size / args.accumulation_steps == 1:
            if args.batch_size == args.mini_batch_size:
                tqdm_main.update(args.step_size * 2)
            else:
                tqdm_main.update(args.step_size)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        tqdm_train0.set_postfix_str(s='[{}/{}]'.format(frames_completed,
                                                       frames_total))
        tqdm_train0.update(1)
        tqdm_train2.set_description('Mean/Max '
            '[GT: {:.1f}/{:.1f}][Flow1: {:.1f}/{:.1f}][Flow2: {:.1f}/{:.1f}][Flow3: {:.1f}/{:.1f}][Flow4: {:.1f}/{:.1f}][Flow5: {:.1f}/{:.1f}][Flow6: {:.1f}/{:.1f}]'.format(
                flowGT_mean.avg, flowGT_max,
                flow1_mean.avg, flow1_max,
                flow2_mean.avg, flow2_max,
                flow3_mean.avg, flow3_max,
                flow4_mean.avg, flow4_max,
                flow5_mean.avg, flow5_max,
                flow6_mean.avg, flow6_max))
        
        
        if args.total_steps >= args.max_num_steps:
            tqdm_train1.set_description('Dataset {0}  Train: EPE {1}  Fl {2}  WAUC {3}  LR {4:.3e} - Exceeded total {5}'.format(
                                        dataset_name,
                                        flow_EPEs,
                                        flow_Fls,
                                        flow_WAUCs,
                                        lr,
                                        args.max_num_steps))
            break


def validate(val_loader, model, run_writer, inverse_transform):
    global args, tqdm_test0, tqdm_test1, tqdm_test2

    batch_time = AverageMeter()
    test_flow2_EPEs = AverageMeter()
    test_flow2_Fls = AverageMeter()
    test_flow2_WAUCs = AverageMeter()
    input_mean = AverageMeter() #Average of means
    input_std = AverageMeter() #Take square root of average of squares
    Fl = 0.0
    flowGT_mean = AverageMeter()
    flow2_mean = AverageMeter()
    flowGT_max = 0
    flow2_max = 0

    n_registered_pxs = 0.1

    # switch to evaluate mode
    model.eval()
    torch.cuda.empty_cache()

    end = time.time()
    #for i, (mini_batch) in enumerate(val_loader):
    realEPE = utils.EPE()
    realWAUC = utils.WAUC()
    realFl = utils.Fl_KITTI_2015(use_mask=True)
    test_epoch_size = len(val_loader) if args.test_epoch_size == 0 else min(len(val_loader), args.test_epoch_size)
    j = 0
    frames_total = len(val_loader.dataset)
    frames_completed = 0
    tqdm_test0.reset()
    tqdm_test1.reset()
    tqdm_test2.reset()

    for i, (inputs, target) in enumerate(val_loader):
        if i >= args.test_epoch_size:
            tqdm_test1.set_description('Train: EPE {0}   Fl {1}   WAUC {2}   - Test iterations exceeded {3}'.format(
                                       test_flow2_EPEs,
                                       test_flow2_Fls,
                                       test_flow2_WAUCs,
                                       args.test_epoch_size))
            break
        frames_completed += target.size(0)
        mask = target[:,2,:,:].to(args.device)
        target = target.to(args.device)
        B2, C2, intHeight2, intWidth2 = target[:,:2,:,:].size()
        target[:,0,:,:] = target[:,0,:,:].nan_to_num(0,intWidth2,-intWidth2)
        target[:,1,:,:] = target[:,1,:,:].nan_to_num(0,intHeight2,-intHeight2)
        target[:,:2,:,:] *= mask[:,None,:,:]
        inputs = inputs.to(args.device)

        #input_mean.update(torch.mean(inputs).item(), target.size(0))
        #input_std.update(np.square(torch.mean(inputs).item()), target.size(0))


        
        # Need input image resolution to always be a multiple of 64
        B, N, c, intHeight, intWidth = inputs.size()
        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))
        #args.div_flow = max(intPreprocessedWidth, intPreprocessedHeight)
        
        if intPreprocessedWidth == intWidth and intPreprocessedHeight == intHeight:
            # Faster but same memory utilization. Without detach it is slower but takes less memory.
            #pass
            tensorFirst = inputs[:,0,:,:,:]
            tensorSecond = inputs[:,1,:,:,:]
            tensorThird = inputs[:,2,:,:,:]
        else:
            tensorFirst = F.interpolate(
                                input         = inputs[:,0,:,:,:],
                                size          = (intPreprocessedHeight, intPreprocessedWidth),
                                mode          = 'bilinear',
                                align_corners = False,
                                antialias     = True)
            tensorSecond = F.interpolate(
                                input         = inputs[:,1,:,:,:],
                                size          = (intPreprocessedHeight, intPreprocessedWidth),
                                mode          = 'bilinear',
                                align_corners = False,
                                antialias     = True)
            tensorThird = F.interpolate(
                                input         = inputs[:,2,:,:,:],
                                size          = (intPreprocessedHeight, intPreprocessedWidth),
                                mode          = 'bilinear',
                                align_corners = False,
                                antialias     = True)
            inputs = torch.stack((tensorFirst, tensorSecond, tensorThird), dim=1)

        with torch.cuda.amp.autocast():
            outputB, outputF = model(inputs)
        
        infinity = max(intHeight,intWidth)
        outputF = F.interpolate(
                            input         = outputF,
                            size          = (intHeight, intWidth),
                            mode          = 'bilinear',
                            align_corners = False,
                            antialias     = True)
        #outputF *= mask[:,None,:,:]
        #outputF[:, 0, :, :] *= intWidth / intPreprocessedWidth
        #outputF[:, 1, :, :] *= intHeight / intPreprocessedHeight
        outputF[:, 0, :, :] = outputF[:, 0, :, :].nan_to_num(0,intWidth,-intWidth).float() * intWidth / intPreprocessedWidth
        outputF[:, 1, :, :] = outputF[:, 1, :, :].nan_to_num(0,intHeight,-intHeight).float() * intHeight / intPreprocessedHeight

        flowGT_max = max(flowGT_max, target[:,:2,:,:].float().abs().max())
        flow2_max = max(flow2_max, outputF.float().abs().max())
        flowGT_mean.update(target[:,:2,:,:].float().abs().sum() / (B2 * C2 * intHeight2 * intWidth2), B)
        flow2_test = outputF[:,:2,:,:].float().abs().sum()
        flow2_mean.update(flow2_test / (B2 * C2 * intHeight2 * intWidth2), B)


        # compute output
        flow2_EPE = realEPE(outputF, target)
        flow2_Fl = realFl(outputF, target, mask)
        flow2_WAUC = realWAUC(outputF, target, mask)

        # record EPE
        test_flow2_EPEs.update(flow2_EPE.item(), B)
        test_flow2_Fls.update(flow2_Fl.item(), torch.sum(mask).item())
        test_flow2_WAUCs.update(flow2_WAUC.item(), torch.sum(mask).item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # log first output of first batches
        if (i * args.test_batch_size) % args.image_freq == 0 and run_writer is not None and j < 3:
            frame = args.test_batch_size * i
            # sometimes output flows can have data that is too big to convert to an image. Clipping it to stop errors
            # target has already been masked
            c, h, w = outputF[0,:,:,:].shape
            invalid = torch.logical_or(torch.logical_or(torch.isnan(outputF[0,0,:,:]), torch.isnan(outputF[0,1,:,:])),
                                       torch.logical_or(torch.isinf(outputF[0,0,:,:]), torch.isinf(outputF[0,1,:,:])))
            outputF[0,0,:,:][invalid] = 0.0
            outputF[0,1,:,:][invalid] = 0.0
            invalid = torch.logical_or(invalid, torch.logical_or(torch.abs(outputF[0,0,:,:]) >= w,
                                                                 torch.abs(outputF[0,1,:,:]) >= h))
            outputF[0,0,:,:][invalid] = 0.0
            outputF[0,1,:,:][invalid] = 0.0


            if args.best_WAUC == -1:
                target_flow_image = flow_vis.flow_to_color(target[0,:2,:,:].permute((1,2,0)).detach().cpu().numpy(),
                                                           clip_flow=None).transpose((2, 0, 1))
                run_writer.add_image('GroundTruth{}'.format(frame), target_flow_image, 0)
                if inverse_transform is not None:
                    run_writer.add_image('Input{}'.format(frame),
                                         inverse_transform(inputs[:, 0, :, :, :][0,:3]).clamp(0,1).cpu(),
                                         0)
                    run_writer.add_image('Input{}'.format(frame),
                                         inverse_transform(inputs[:, 1, :, :, :][0,:3]).clamp(0,1).cpu(),
                                         1)
                    run_writer.add_image('Input{}'.format(frame),
                                         inverse_transform(inputs[:, 2, :, :, :][0,:3]).clamp(0,1).cpu(),
                                         2)
                else:
                    run_writer.add_image('Input{}'.format(frame),
                                         inputs[:, 0, :, :, :][0,:3].clamp(0,1).cpu(),
                                         0)
                    run_writer.add_image('Input{}'.format(frame),
                                         inputs[:, 1, :, :, :][0,:3].clamp(0,1).cpu(),
                                         1)
                    run_writer.add_image('Input{}'.format(frame),
                                         inputs[:, 2, :, :, :][0,:3].clamp(0,1).cpu(),
                                         2)
                if j == 0:
                    # The nondeterministic nature of ConvTranspose2d causes the graph to give a warning about deviations.
                    # Enabling deterministic resolves this issue but slows down performance so only do it for add_graph
                    cudnn.deterministic = True
                    run_writer.add_graph(model, inputs)
                    cudnn.deterministic = False

            output_flow_image = flow_vis.flow_to_color(outputF[0,:2,:,:].permute((1,2,0)).detach().cpu().numpy().astype(np.float32),
                                                       clip_flow=None).transpose((2, 0, 1))
            run_writer.add_image('Output{}'.format(frame),
                                 output_flow_image,
                                 args.total_steps)
            j += 1

        tqdm_test0.set_postfix_str(s='[{}/{}]'.format(frames_completed,
                                                      frames_total))
        tqdm_test0.update(1)
        tqdm_test1.set_description('Test: EPE {0}   Fl {1}   WAUC {2}'.format(
                                   test_flow2_EPEs,
                                   test_flow2_Fls,
                                   test_flow2_WAUCs))
        tqdm_test2.set_description('Mean/Max '
            '[GT: {:.1f}/{:.1f}][flow: {:.1f}/{:.1f}]'.format(
                flowGT_mean.avg, flowGT_max,
                flow2_mean.avg, flow2_max))

    run_writer.add_scalar('Test/EPE',  test_flow2_EPEs.avg,  global_step = args.total_steps, new_style = True)
    run_writer.add_scalar('Test/Fl',   test_flow2_Fls.avg,   global_step = args.total_steps, new_style = True)
    run_writer.add_scalar('Test/WAUC', test_flow2_WAUCs.avg, global_step = args.total_steps, new_style = True)
    is_best = test_flow2_EPEs.avg < args.best_EPE
    is_best_WAUC = test_flow2_WAUCs.avg > args.best_WAUC
    args.best_EPE  = min(test_flow2_EPEs.avg,  args.best_EPE)
    args.best_Fl   = min(test_flow2_Fls.avg,   args.best_Fl)
    args.best_WAUC = max(test_flow2_WAUCs.avg, args.best_WAUC)
    model_state_dict = {'state_dict': model.state_dict(),
                        'args':       args}
    torch.save(model_state_dict, args.save_path / args.run_path / (args.arch+'_model_last.pt'))
    if is_best:
        torch.save(model_state_dict, args.save_path / args.run_path / (args.arch+'_model_best.pt'))
    if is_best_WAUC:
        torch.save(model_state_dict, args.save_path / args.run_path / (args.arch+'_model_best_WAUC.pt'))

    #https://www.statology.org/averaging-standard-deviations/
    #Note that the standard deviation is the square root of the variance
    #print('input [mean: {:4f}][std: {:4f}][var: {:4f}]'.format(input_mean.avg, np.sqrt(input_std.avg), input_std.avg))

    #return flow2_EPEs.avg, flow2_Fls.avg, flow2_WAUCs.avg


if __name__ == '__main__':
    #signal.signal(signal.SIGINT,signal_handling)
    try:
        if platform.system() == 'Windows':
            # You can force synchronous computation by setting environment variable CUDA_LAUNCH_BLOCKING=1. This can be handy when an error occurs on the GPU. (With asynchronous execution, such an error isn't reported until after the operation is actually executed, so the stack trace does not show where it was requested.)
            # https://pytorch.org/docs/stable/notes/cuda.html
            # to debug cupy_backends.cuda.api.driver.CUDADriverError: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
            env = 'CUDA_LAUNCH_BLOCKING'
            #if env not in os.environ:
            #    os.environ[env] = '1'
            if env in os.environ:
                del os.environ[env]
        main()
        if platform.system() == 'Windows':
            winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
        elif platform.system() == 'Darwin':
            os.system('say "Your CDVD-TSP program has finished"')
        elif platform.system() == 'Linux':
            os.system('spd-say "Your CDVD-TSP program has finished"')
    except SystemExit:
        pass
    except:
        traceback.print_exc()
        if platform.system() == 'Windows':
            winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
        elif platform.system() == 'Darwin':
            os.system('say "Your CDVD-TSP program has crashed"')
        elif platform.system() == 'Linux':
            os.system('spd-say "Your CDVD-TSP program has crashed"')
    finally:
        #tqdm_main.close()
        #tqdm_train0.close()
        #tqdm_train1.close()
        #tqdm_train2.close()
        #tqdm_test0.close()
        #tqdm_test1.close()
        #tqdm_test2.close()
        if platform.system() == 'Windows':
            env = 'CUDA_LAUNCH_BLOCKING'
            if env in os.environ:
                del os.environ[env]
        print("Closed cleanly")
