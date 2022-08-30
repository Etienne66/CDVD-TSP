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
import future
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
parser.add_argument('--dataset', metavar='DATASET', default='autoflow', #flying_things_both flying_chairs2 flying_things_final flying_things_clean
                    choices=dataset_names,# nargs='*',
                    help='dataset type: ' + ' | '.join(dataset_names))
parser.add_argument('--datasets', metavar='DATASET', default=['autoflow','chairs2','things','viper'], #flying_things_both flying_chairs2 flying_things_final flying_things_clean
                    choices=dataset_names,# nargs='*',
                    help='dataset type: ' + ' | '.join(dataset_names))
parser.add_argument('--testdataset', metavar='DATASET', default='mpi_sintel_final', #mpi_sintel_final mpi_sintel_clean KITTI_2015_occ
                    choices=dataset_names,
                    help='dataset type: ' + ' | '.join(dataset_names))
parser.add_argument('--arch', '-a', metavar='ARCH', default='flow_pwc2',
                    choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' + ' | '.join(model_names))
parser.add_argument('--n_GPUs', type=int, default=1, metavar='N',
                    help='number of GPUs')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers. Set to number of cpus')
parser.add_argument('-b', '--batch_size', dest='batch_size', default=32, type=int, #22
                    metavar='N', help='mini-batch size')
parser.add_argument('--mini_batch_sizes', dest='batch_size', default=[32,32,32,16], type=int, #22
                    metavar='N', help='mini-batch size')

parser.add_argument('--test_batch_size', default=6, type=int, #6
                    metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num_steps', default=6_200_000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--accumulation_steps', type=int, default=2,
                    help='True_Batch_Size = accumalation_steps * batch_size')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch_size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('--epoch_sizes', default=[40000,43200,76800,12800], type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('--test_epoch_size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('--print_freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--val_freq', type=int, default=5000,
                    help='validation frequency')
parser.add_argument('--image_freq', default=60, type=int,
                    metavar='N', help='test image frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--no_date', action='store_true',
                    help='don\'t append date timestamp to folder' )
parser.add_argument('--div_flow', type=int, default=1, metavar='N', choices=[1,16,20,256,512,1024],
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
parser.add_argument('--batchnorm', action='store_true',
                    help='use torch.nn.BatchNorm2d after torch.nn.Conv2d')
parser.add_argument('--Crop_Size', type=int, default=None, metavar='N', nargs=2,
                    help='Size of random Crop. H W')
parser.add_argument('--loss', default='0*mWAUCl+1*mEPE', type=str, metavar='LOSS',
                    choices=['0*mWAUCl+0*mEPE+1*mL2','.1*mWAUCl+.7*mEPE',
                             '0*mWAUCl+1*mEPE','.1*mWAUCl+2*mEPE','1*mWAUCl+0*mEPE',
                             '1*mWAUCl+2*mEPE','1*mWAUCl+2*mEPE1'],
                    help='Loss function. EPE = Average endpoint error; Fl = Percentage of optical flow outliers from 0 to 100 percent')
parser.add_argument('--weights', type=float, nargs='+', default=[0, 1, 0.5, 0.25, 0.25, 0.25]) #[1, 0.5, 0.25, 0.25, 0.25, 0.25]

parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--split_losses', action='store_true',
                    help='split flow losses and graph them')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint')
parser.add_argument('--use_flow1', action='store_true', help='use flow 1 as the final stage instead of flow 2')

AugmenterGroup = parser.add_argument_group('Augmenter', description='Augmenter options')
#AugmenterGroup.add_argument('--stage', type=str, default='autoflow',
#                    help="determines which dataset to use for training")
AugmenterGroup.add_argument('--image_size', type=int, nargs='+', default=[488, 576])
AugmenterGroup.add_argument('--image_sizes', type=int, nargs='+', default=[[488, 576],[512,384],[960, 512],[1920, 1024]])
AugmenterGroup.add_argument('--shiftprob', dest='shift_aug_prob', type=float, default=0.0,
                    help='Probability of shifting augmentation')
AugmenterGroup.add_argument('--shiftsigmas', dest='shift_sigmas', default="16,10", type=str,
                    help='Stds of shifts for shifting consistency loss')

OptimizerGroup = parser.add_argument_group('Optimizer', description='Optimizer options')
OptimizerGroup.add_argument('--solver', default='adamw',choices=['adam','adamw','sgd'],
                    help='solver algorithms')
OptimizerGroup.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float, metavar='LR',
                    help='initial learning rate')
OptimizerGroup.add_argument('--max_lr', default=2.5e-4, type=float, metavar='M',
                    help='Sets maximum Learning Rate for LRFinder, OneCycleLR, and CyclicLr')
OptimizerGroup.add_argument('--lr_decay', type=int, default=200, metavar='N',
                    help='learning rate decay per N epochs')
OptimizerGroup.add_argument('--weight_decay', type=float, default=0.,
                    help='weight decay')
OptimizerGroup.add_argument('--AdamW_weight_decay', type=float, default=1e-8,
                    help='weight decay')
OptimizerGroup.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd')
OptimizerGroup.add_argument('--betas', type=float, default=[0.9,0.999], metavar='M', nargs=2,
                    help='ADAM betas')


SchedulerGroup = parser.add_argument_group('Scheduler', description='Scheduler options')
SchedulerGroup.add_argument('--scheduler', default='OneCycle',
                    choices=['multisteplr','steplr','onecyclelr','OneCycle','cycliclr','lr_finder'],
                    help='scheduler algorithms')
SchedulerGroup.add_argument('--step_size_up', type=int, default=0, metavar='N',  #337
                    help='Number of training iterations in the increasing half of a cycle.')
SchedulerGroup.add_argument('--gamma', type=float, default=0.5, metavar='M',
                    help='learning rate decay factor for step decay')
SchedulerGroup.add_argument('--CyclicLR_gamma', type=float, default=0.99999826713355001493796951419198, metavar='M',
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

batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()
flow1_Fls = AverageMeter()
flow2_Fls = AverageMeter()
flow1_EPEs = AverageMeter()
flow2_EPEs = AverageMeter()
flow1_WAUCs = AverageMeter()
flow2_WAUCs = AverageMeter()
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    args.normalize = True
    args.mini_batch_size = int(args.batch_size / args.accumulation_steps)
    args.step_size = int(args.batch_size / 8) # Size of the original author's batch size
    #args.batchnorm = True
    if args.pretrained:
        network_data = torch.load('../pretrain_models' / args.pretrained, map_location=torch.device(args.device))
        save_path = args.save_path
        if 'state_dict' not in network_data.keys():
            network_data = {'state_dict': network_data}
            args.arch = 'flow_pwc'
            args.div_flow = 20
            args.batchnorm = False
        elif 'args' in network_data.keys():
            args = network_data['args']
            #Don't use state_dict for scheduler and optimizer because this is not a resume
            network_data.pop('optimizer_state_dict')
            network_data.pop('scheduler_state_dict')

            # Reset device in case the model was saved with a different device
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            args.save_path = save_path
            args.start_epoch = 0
            args.n_iter = 0

            # Reset random number generators
            torch.set_rng_state(network_data['rng_state'].type(torch.ByteTensor))
            random.setstate(network_data['random_state'])
            np.random.set_state(network_data['numpy_random_state'])
        if args.evaluate:
            args.run_path = Path(args.arch + '-' + datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S") + '-PreTrained-Eval')
        else:
            args.run_path = Path(args.arch + '-' + datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S") + '-PreTrained')
        print("=> using pre-trained model '{}\{}'".format(args.arch,args.pretrained))
    elif args.resume is not None:
        # Save resume and epochs to restore after loading
        resume = args.resume
        epochs = args.epochs
        pct_start = args.pct_start
        num_workers = args.num_workers
        network_data = torch.load(args.save_path / args.resume, map_location=torch.device(args.device))
        args = network_data['args']
        #Restore resume and epochs.
        args.resume = resume
        args.epochs = epochs
        args.num_workers = num_workers
        if args.pct_start is None:
            args.pct_start = pct_start

        # Reset device in case the model was saved with a different device
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reset random number generators
        torch.set_rng_state(network_data['rng_state'].type(torch.ByteTensor))
        random.setstate(network_data['random_state'])
        np.random.set_state(network_data['numpy_random_state'])
        print("=> using trained model '{}\{}' with divisions '{}'".format(args.arch,args.resume,args.div_flow))
    else:
        network_data = {}
        print("=> creating model '{}' with divisions '{}'".format(args.arch,args.div_flow))
        args.run_path = Path(args.arch + '-' + datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S"))

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
    # tranforms.normalize output[channel] = (input[channel] - mean[channel]) / std[channel]
    # All channels are RGB. Loads from OpenCV were corrected
    if False: #args.batchnorm:
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

    #print("=> fetching train image pairs in '{}\{}'".format(args.data,args.dataset))
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
    print("=> fetching train image pairs in '{}\{}'".format(args.data,args.dataset))
    train_loader = datasets.fetch_dataloader(args,
                                             image_size      = args.image_size,
                                             mini_batch_size = args.mini_batch_size,
                                             stage           = args.dataset)
    
    #for i in range(len(args.datasets)):
    #    train_loaders.append = datasets.fetch_dataloader(args,
    #                                                     image_size      = args.image_sizes[i],
    #                                                     mini_batch_size = args.mini_batch_sizes[i],
    #                                                     stage           = args.datasets[i])
    
    print("=> fetching test image pairs in '{}\{}'".format(args.data,args.testdataset))
    test_set = data_flow.__dict__[args.testdataset](
        args.data,
        transform=input_transform,
        target_transform=target_transform,
        co_transform=None#co_validate_transform
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
    print('{} samples found, {} train samples and {} test samples '.format(len(val_loader.dataset)+len(train_loader.dataset),
                                                                           len(train_loader.dataset),
                                                                           len(val_loader.dataset)))
    # create model
    model = models.__dict__[args.arch](network_data,
                                       device         = args.device,
                                       use_checkpoint = args.use_checkpoint,
                                       use_flow1      = args.use_flow1,
                                       lr_finder      = args.lr_finder,
                                       div_flow       = args.div_flow,
                                       batchnorm      = args.batchnorm).to(args.device)

    if args.device.type == "cuda" and args.n_GPUs > 1:
        model = torch.nn.DataParallel(model).cuda()
    cudnn.enabled = True
    cudnn.benchmark = True


    assert(args.solver in ['adam','adamw','sgd'])
    #if LooseVersion(torch.__version__) < LooseVersion('1.8.0') and args.solver == 'adamw':
    if version.parse(torch.__version__) < version.Version("1.8.0") and args.solver == 'adamw':
        args.solver = 'adam'
    print('=> setting {} solver'.format(args.solver))

    if args.solver == 'adam':
        kwargs = {'lr':             args.lr,
                  'weight_decay':   args.weight_decay,
                  'betas':          (args.betas[0],args.betas[1])}
        optimizer = torch.optim.Adam(model.parameters(), **kwargs)
    elif args.solver == 'adamw':
        kwargs = {'lr':             args.lr,
                  'weight_decay':   args.AdamW_weight_decay,
                  'betas':          (args.betas[0],args.betas[1])}
        optimizer = torch.optim.AdamW(model.parameters(), **kwargs)
    elif args.solver == 'sgd':
        kwargs = {'lr':           args.lr,
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
            num_iter = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)
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
                  'steps_per_epoch': len(train_loader)}
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
    elif args.scheduler == 'OneCycle':
        OC_total_steps = int(args.num_steps / args.step_size) + 100
        kwargs = {'max_lr':          args.lr,
                  'total_steps':     OC_total_steps,
                  'pct_start':       args.pct_start,
                  'cycle_momentum':  False,
                  'anneal_strategy': 'linear',
                  'steps_per_epoch': len(train_loader)}
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
    elif args.scheduler == 'cycliclr':
        args.epoch_size = int(len(train_loader)/2.) * 2 if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)
        args.step_size_up = int(args.epoch_size/2) if args.step_size_up == 0 else args.step_size_up
        kwargs = {'base_lr':        args.lr,
                  'max_lr':         args.max_lr,
                  'cycle_momentum': False if args.solver != 'sgd' else True, #needs to be False for Adam/AdamW
                  'step_size_up':   args.step_size_up,
                  'mode':           args.CyclicLR_mode,
                  'gamma':          args.CyclicLR_gamma}
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **kwargs)
    elif args.scheduler == 'lr_finder':
        # This is just for testing. Disable when not needed
        #torch.autograd.set_detect_anomaly(True)

        #Need to crop video so that it doesn't need to be resized
        #custom_train_iter = CustomTrainIter(train_loader)
        num_iter = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)
        fig_kw = {'figsize': [19.2,10.8]}
        #plt.rcParams.update({'font.size': 20})
        plt.ticklabel_format(axis="x",style="sci",scilimits=(0,0))
        fig, ax = plt.subplots(**fig_kw)
        if args.lr_finder_Leslie_Smith:
            custom_val_iter = CustomValIter(loader_test)
            lr_finder = LRFinder(model, optimizer, args.loss_Flow_net, device="cuda")
            lr_finder.range_test(train_loader,#custom_train_iter,
                                 #val_loader = custom_val_iter,
                                 end_lr     = args.max_lr,
                                 num_iter   = num_iter,
                                 step_mode  = "linear")
        else:
            lr_finder = LRFinder(model, optimizer, args.loss_Flow_net, device="cuda")
            lr_finder.range_test(train_loader,#custom_train_iter,
                                 end_lr   = args.max_lr,
                                 num_iter = num_iter)
        lr_finder.plot(ax=ax)
        plt.grid(True)
        fig.tight_layout()
        eval_writer = SummaryWriter(log_dir=args.save_path / args.run_path / 'eval')
        eval_writer.add_figure('lr_finder Length: {}/Loss: {}/div_flow: {}'.format(len(train_loader),args.loss, args.div_flow),
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
    args.epoch_size = len(train_loader) if args.epoch_size == 0 else args.epoch_size
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
        while args.total_steps <= args.num_steps:
            # train for one epoch
            train(train_loader, val_loader, model, optimizer, scheduler, train_writer, inverse_transform)
            args.train_loader_start = 0

            if args.scheduler in ['steplr','multisteplr']:
                scheduler.step()

            model_state_dict = {}
            model_state_dict['args'] = args
            model_state_dict['state_dict'] = model.state_dict()
            model_state_dict['optimizer_state_dict'] = optimizer.state_dict()
            model_state_dict['scheduler_state_dict'] = scheduler.state_dict()
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
            if args.total_steps >= args.num_steps:
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
        train_writer.add_hparams({'Batch Size':      '{:1d}'.format(args.batch_size),
                                  'Test Batch Size': '{:1d}'.format(args.test_batch_size),
                                  'Parameters':      count_parameters(model),
                                  'Optimizer':       args.solver,
                                  'Scheduler':       args.scheduler,
                                  'Loss Function':   args.loss,
                                  'Dataset':         args.dataset,
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
        raise



def train(train_loader, val_loader, model, optimizer, scheduler, train_writer, inverse_transform):
    global args, tqdm_main, tqdm_train0, tqdm_train1, tqdm_train2
    realEPE = utils.EPE()
    realWAUC = utils.WAUC()
    realFl = utils.Fl_KITTI_2015(use_mask=True)

    # Recreate scaler every epoch.
    scaler = GradScaler(enabled=args.mixed_precision)

    # switch to train mode
    model.train()
    torch.cuda.empty_cache()
    model.zero_grad(set_to_none=True)

    frames_total = len(train_loader.dataset)
    frames_completed = 0
    end = time.time()
    tqdm_train0.reset()
    tqdm_train1.reset()
    tqdm_train2.reset()
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

    for i, (inputs, target) in enumerate(train_loader):
        if i >= args.epoch_size:
            tqdm_train1.set_description('Train: EPE {0}\t Fl {1}\t WAUC {2}\t - Train iterations exceeded {3}'.format(
                                        flow1_EPEs,
                                        flow1_Fls,
                                        flow1_WAUCs,
                                        args.epoch_size))
            break
        args.total_steps += args.step_size / args.accumulation_steps
        # measure data loading time
        data_time.update(time.time() - end)
        frames_completed += target.size(0)
        mask = target[:,2,:,:].to(args.device)
        target = target.to(args.device)
        target[:,:2,:,:] *= mask[:,None,:,:]
        inputs = inputs.to(args.device)
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
        
       
        if args.split_losses:
            loss_total, loss_EPE, loss_WAUCl = args.loss_Flow_net(outputs, target)
          # Without .detach() there is a memory leak. Don't need grad on these
            if args.use_flow1:
                flow1_WAUC.update(100 - loss_WAUCl[0].detach(), target.size(0))
            flow2_WAUC.update(100 - loss_WAUCl[1].detach(), target.size(0))
            flow3_WAUC.update(100 - loss_WAUCl[2].detach(), target.size(0))
            flow4_WAUC.update(100 - loss_WAUCl[3].detach(), target.size(0))
            flow5_WAUC.update(100 - loss_WAUCl[4].detach(), target.size(0))
            flow6_WAUC.update(100 - loss_WAUCl[5].detach(), target.size(0))
            if args.use_flow1:
                flow1_EPE.update(loss_EPE[0].detach(), target.size(0))
            flow2_EPE.update(loss_EPE[1].detach(), target.size(0))
            flow3_EPE.update(loss_EPE[2].detach(), target.size(0))
            flow4_EPE.update(loss_EPE[3].detach(), target.size(0))
            flow5_EPE.update(loss_EPE[4].detach(), target.size(0))
            flow6_EPE.update(loss_EPE[5].detach(), target.size(0))
            if args.use_flow1:
                loss_EPE = loss_EPE[0].detach().cpu().numpy()
                accuracy_WAUC = 100 - loss_WAUCl[0].detach()
            else:
                loss_EPE = loss_EPE[1].detach().cpu().numpy()
                accuracy_WAUC = 100 - loss_WAUCl[1].detach()
        else:
            loss_total = args.loss_Flow_net(outputs, target)
        loss_total /= args.accumulation_steps
        scaler.scale(loss_total).backward()
        # record loss and EPE
        flowGT_max = max(flowGT_max, target.detach().abs().max())
        
        if args.use_flow1:
            flow1_max = max(flow1_max, outputs[0].detach().abs().max())
        flow2_max = max(flow2_max, outputs[1].detach().abs().max())
        flow3_max = max(flow3_max, outputs[2].detach().abs().max())
        flow4_max = max(flow4_max, outputs[3].detach().abs().max())
        flow5_max = max(flow5_max, outputs[4].detach().abs().max())
        flow6_max = max(flow6_max, outputs[5].detach().abs().max())
        flowGT_mean.update(target.detach().nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).abs().mean(), target.size(0))
        if args.use_flow1:
            flow1_test = outputs[0].detach().nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).abs().mean().cpu().numpy()
        flow2_test = outputs[1].detach().nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).abs().mean().cpu().numpy()
        flow3_test = outputs[2].detach().nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).abs().mean().cpu().numpy()
        flow4_test = outputs[3].detach().nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).abs().mean().cpu().numpy()
        flow5_test = outputs[4].detach().nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).abs().mean().cpu().numpy()
        flow6_test = outputs[5].detach().nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).abs().mean().cpu().numpy()
        if args.use_flow1:
            if not np.isinf(flow1_test):
                flow1_mean.update(flow1_test, target.size(0))
        if not np.isinf(flow2_test):
            flow2_mean.update(flow2_test, target.size(0))
        if not np.isinf(flow3_test):
            flow3_mean.update(flow3_test, target.size(0))
        if not np.isinf(flow4_test):
            flow4_mean.update(flow4_test, target.size(0))
        if not np.isinf(flow5_test):
            flow5_mean.update(flow5_test, target.size(0))
        if not np.isinf(flow6_test):
            flow6_mean.update(flow6_test, target.size(0))

        lr = scheduler.get_last_lr()[0]
        #outputs[0] = F.interpolate(
        #                    input         = outputs[0],
        #                    size          = (intHeight, intWidth),
        #                    mode          = 'bicubic',
        #                    align_corners = False)
        #outputs[0][:, 0, :, :] *= intWidth / intPreprocessedWidth
        #outputs[0][:, 1, :, :] *= intHeight / intPreprocessedHeight
        #loss_EPE = realEPE(outputs[0], target)
        #accuracy_WAUC = realWAUC(outputs[0], target)

        if not args.split_losses:
            if args.use_flow1:
                loss_EPE = realEPE(outputs[0].detach(), target).cpu().numpy()
                accuracy_WAUC = realWAUC(outputs[0].detach(), target)
            else:
                loss_EPE = realEPE(outputs[1].detach(), target).cpu().numpy()
                accuracy_WAUC = realWAUC(outputs[1].detach(), target)
        elif args.loss ==  '1*mWAUCl+2*mEPE1':
            if args.use_flow1:
                loss_EPE = realEPE(outputs[0].detach(), target).cpu().numpy()
            else:
                loss_EPE = realEPE(outputs[1].detach(), target).cpu().numpy()
        if args.use_flow1:
            loss_Fl = realFl(outputs[0].detach(), target)
        else:
            loss_Fl = realFl(outputs[1].detach(), target)
        if not np.isinf(loss_EPE):
            flow1_EPEs.update(loss_EPE, target.size(0))
        flow1_Fls.update(loss_Fl, torch.sum(mask).item())
        flow1_WAUCs.update(accuracy_WAUC, torch.sum(mask).item())
        learningRates.update(lr, target.size(0))
        tqdm_train1.set_description('Train: EPE {0}\t Fl {1}\t WAUC {2}'.format(
                                    flow1_EPEs,
                                    flow1_Fls,
                                    flow1_WAUCs))
        if args.total_steps % args.step_size == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            model.zero_grad(set_to_none=True)                # Reset gradients tensors
            skip_lr_sched = (scale > scaler.get_scale())
            if not skip_lr_sched:
                scheduler.step()
            if args.total_steps % args.print_freq  == 0:
                train_writer.add_scalar('Train/EPE',  flow1_EPEs.avg,    args.total_steps)
                train_writer.add_scalar('Train/Fl',   flow1_Fls.avg,     args.total_steps)
                train_writer.add_scalar('Train/WAUC', flow1_WAUCs.avg,   args.total_steps)
                train_writer.add_scalar('Train/lr',   learningRates.avg, args.total_steps)
                flow1_EPEs.reset()
                flow1_Fls.reset()
                flow1_WAUCs.reset()
                learningRates.reset()
            if (args.total_steps) % args.val_freq  == 0:
                if args.split_losses:
                    if args.use_flow1:
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
                #args.train_loader_start = i
                batch_time.update(time.time() - end)
                end = time.time()
                with torch.no_grad():
                    validate(val_loader, model, train_writer, inverse_transform)
                model.train()
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
        
        
        if args.total_steps >= args.num_steps:
            tqdm_train1.set_description('Train: EPE {0}\t Fl {1}\t WAUC {2}\t - Train Exceeded total steps {3}'.format(
                                        flow1_EPEs,
                                        flow1_Fls,
                                        flow1_WAUCs,
                                        args.num_steps))
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
            tqdm_test1.set_description('Train: EPE {0}\t Fl {1}\t WAUC {2}\t - Test iterations exceeded {3}'.format(
                                       test_flow2_EPEs,
                                       test_flow2_Fls,
                                       test_flow2_WAUCs,
                                       args.test_epoch_size))
            break
        frames_completed += target.size(0)
        mask = target[:,2,:,:].to(args.device)
        #mask = mask.to(args.device)
        target = target[:,:2,:,:].to(args.device)
        target *= mask[:,None,:,:]
        #target = target[:,:2,:,:].to(args.device)
        inputs = inputs.to(args.device)
        input_mean.update(torch.mean(inputs).item(), target.size(0))
        input_std.update(np.square(torch.mean(inputs).item()), target.size(0))


        
        # Need input image resolution to always be a multiple of 64
        #b, N, c, intHeight, intWidth = inputs.size()
        #intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        #intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))
        #if intPreprocessedWidth == intWidth and intPreprocessedHeight == intHeight:
        #    # Faster but same memory utilization. Without detach it is slower but takes less memory.
        #    pass
        #    #tensorFirst = inputs[:, 0, :, :, :]
        #    #tensorSecond = inputs[:, 1, :, :, :]
        #else:
        #    tensorFirst = F.interpolate(
        #                        input         = inputs[:,0,:,:,:],
        #                        size          = (intPreprocessedHeight, intPreprocessedWidth),
        #                        mode          = 'bicubic',
        #                        align_corners = False,
        #                        antialias     = True)
        #    tensorSecond = F.interpolate(
        #                        input         = inputs[:,1,:,:,:],
        #                        size          = (intPreprocessedHeight, intPreprocessedWidth),
        #                        mode          = 'bicubic',
        #                        align_corners = False,
        #                        antialias     = True)
        #    inputs = torch.stack((tensorFirst, tensorSecond), dim=1)

        with torch.cuda.amp.autocast():
            output = model(inputs)
        
        #output = F.interpolate(
        #                    input         = output,
        #                    size          = (intHeight, intWidth),
        #                    mode          = 'bicubic',
        #                    align_corners = False)
        #output[:, 0, :, :] *= intWidth / intPreprocessedWidth
        #output[:, 1, :, :] *= intHeight / intPreprocessedHeight


        flowGT_max = max(flowGT_max, target.detach().abs().max())
        flow2_max = max(flow2_max, output.detach().abs().max())
        flowGT_mean.update(target.detach().nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).abs().mean(), target.size(0))
        flow2_test = output.detach().nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).abs().mean().cpu().numpy()
        if not np.isinf(flow2_test):
            flow2_mean.update(flow2_test, target.size(0))


        # compute output
        flow2_EPE = realEPE(output*mask[:,None,:,:], target)
        flow2_Fl = realFl(output, target, mask)
        flow2_WAUC = realWAUC(output, target, mask)

        # record EPE
        test_flow2_EPEs.update(flow2_EPE.item(), target.size(0))
        test_flow2_Fls.update(flow2_Fl.item(), torch.sum(mask).item())
        test_flow2_WAUCs.update(flow2_WAUC.item(), torch.sum(mask).item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # log first output of first batches
        if (i * args.test_batch_size) % args.image_freq == 0 and run_writer is not None and j < 3:
            frame = args.test_batch_size * i

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
                else:
                    run_writer.add_image('Input{}'.format(frame),
                                         inputs[:, 0, :, :, :][0,:3].clamp(0,1).cpu(),
                                         0)
                    run_writer.add_image('Input{}'.format(frame),
                                         inputs[:, 1, :, :, :][0,:3].clamp(0,1).cpu(),
                                         1)
                if j == 0:
                    print()
                    print()
                    print()
                    print()
                    print()
                    print()
                    print()
                    print()
                    run_writer.add_graph(model, inputs)
                    print()
                    print()

            output_flow_image = flow_vis.flow_to_color(output[0,:2,:,:].permute((1,2,0)).detach().cpu().numpy(),
                                                       clip_flow=None).transpose((2, 0, 1))
            run_writer.add_image('Output{}'.format(frame),
                                 output_flow_image,
                                 args.total_steps)
            j += 1

        tqdm_test0.set_postfix_str(s='[{}/{}]'.format(frames_completed,
                                                      frames_total))
        tqdm_test0.update(1)
        tqdm_test1.set_description('Test: EPE {0}\t Fl {1}\t WAUC {2}'.format(
                                   test_flow2_EPEs,
                                   test_flow2_Fls,
                                   test_flow2_WAUCs))
        tqdm_test2.set_description('Mean/Max '
            '[GT: {:.1f}/{:.1f}][flow2: {:.1f}/{:.1f}]'.format(
                flowGT_mean.avg, flowGT_max,
                flow2_mean.avg, flow2_max))

    run_writer.add_scalar('Test/EPE',  test_flow2_EPEs.avg,  args.total_steps)
    run_writer.add_scalar('Test/Fl',   test_flow2_Fls.avg,   args.total_steps)
    run_writer.add_scalar('Test/WAUC', test_flow2_WAUCs.avg, args.total_steps)
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
