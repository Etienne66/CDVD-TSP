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
from distutils.version import LooseVersion

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
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


model_names = sorted(name for name in models.__all__)
dataset_names = sorted(name for name in data_flow.__all__)

parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR', type=Path, default=Path('../dataset'),
                    help='path to datasets')
parser.add_argument('--pretrained', type=Path, default=None,
                    help='path to pre-trained model')
parser.add_argument('--resume', type=Path, default=None,
                    help='path to pre-trained model')
parser.add_argument('--save_path', type=Path, default='../runs',
                    help='path to save models/tensorboard metrics')
parser.add_argument('--dataset', metavar='DATASET', default='flying_chairs2', #flying_things_both flying_chairs2 flying_things_final flying_things_clean
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
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers. Set to number of cpus')
parser.add_argument('-b', '--batch_size', default=22, type=int, #22
                    metavar='N', help='mini-batch size')
parser.add_argument('--test_batch_size', default=6, type=int, #6
                    metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch_size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('--test_epoch_size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('--print_freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--image_freq', default=60, type=int,
                    metavar='N', help='test image frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--no_date', action='store_true',
                    help='don\'t append date timestamp to folder' )
parser.add_argument('--div_flow', type=int, default=1024, metavar='N', choices=[1,16,20,256,512,1024],
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
parser.add_argument('--batchnorm', action='store_true',
                    help='use torch.nn.BatchNorm2d after torch.nn.Conv2d')
parser.add_argument('--Crop_Size', type=int, default=None, metavar='N', nargs=2,
                    help='Size of random Crop. H W')
parser.add_argument('--loss', default='1*mWAUCl+2*mEPE', type=str, metavar='LOSS',
                    choices=['0*mWAUCl+0*mEPE+1*mL2','.1*mWAUCl+.7*mEPE','1*mWAUCl+2*mEPE','0*mWAUCl+1*mEPE','.1*mWAUCl+2*mEPE','1*mWAUCl+0*mEPE'],
                    help='Loss function. EPE = Average endpoint error; Fl = Percentage of optical flow outliers from 0 to 100 percent')
parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--split_losses', action='store_true',
                    help='split flow losses and graph them')


OptimizerGroup = parser.add_argument_group('Optimizer', description='Optimizer options')
OptimizerGroup.add_argument('--solver', default='adamw',choices=['adam','adamw','sgd'],
                    help='solver algorithms')
OptimizerGroup.add_argument('--lr', '--learning-rate', default=1e-7, type=float, metavar='LR',
                    help='initial learning rate')
OptimizerGroup.add_argument('--max_lr', default=2e-5, type=float, metavar='M',
                    help='Sets maximum Learning Rate for LRFinder, OneCycleLR, and CyclicLr')
OptimizerGroup.add_argument('--lr_decay', type=int, default=200, metavar='N',
                    help='learning rate decay per N epochs')
OptimizerGroup.add_argument('--weight_decay', type=float, default=0.,
                    help='weight decay')
OptimizerGroup.add_argument('--AdamW_weight_decay', type=float, default=1e-2,
                    help='weight decay')
OptimizerGroup.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd')
OptimizerGroup.add_argument('--betas', type=float, default=[0.9,0.999], metavar='M', nargs=2,
                    help='ADAM betas')


SchedulerGroup = parser.add_argument_group('Scheduler', description='Scheduler options')
SchedulerGroup.add_argument('--scheduler', default='cycliclr',
                    choices=['multisteplr','steplr','onecyclelr','cycliclr','lr_finder'],
                    help='scheduler algorithms')
SchedulerGroup.add_argument('--step_size_up', type=int, default=0, metavar='N',  #337
                    help='Number of training iterations in the increasing half of a cycle.')
SchedulerGroup.add_argument('--gamma', type=float, default=0.5, metavar='M',
                    help='learning rate decay factor for step decay')
SchedulerGroup.add_argument('--CyclicLR_gamma', type=float, default=0.99999542930698003792558023284368, metavar='M',
                    help='learning rate decay factor for step decay')
SchedulerGroup.add_argument('--CyclicLR_mode', type=str, default='triangular',
                    choices=['triangular','triangular2','exp_range'],
                    help='learning rate policy')
SchedulerGroup.add_argument('--milestones', default=[72,108,144,180], metavar='N', nargs='*',
                    help='epochs at which learning rate is divided by 2')
SchedulerGroup.add_argument('--lr_finder_Leslie_Smith', action='store_true',
                    help='Run the LRFinder using Leslie Smith''s approach')




def main():
    global args
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cpu = False
    args.lr_finder = True if args.scheduler == 'lr_finder' else False
    args.n_sequence = 2
    args.n_iter = int(args.start_epoch)
    args.best_EPE = -1
    args.best_Fl = -1
    args.best_WAUC = -1
    args.load = None #Used by loss module
    args.n_channel=2
    args.normalize = True
    #args.batchnorm = True
    if args.pretrained:
        network_data = torch.load('../pretrain_models' / args.pretrained, map_location=torch.device(args.device))
        save_path = args.save_path
        if 'state_dict' not in network_data.keys():
            network_data = {'state_dict': network_data}
            args.arch = 'flow_pwc'
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
        workers = args.workers
        network_data = torch.load(args.save_path / args.resume, map_location=torch.device(args.device))
        args = network_data['args']
        #Restore resume and epochs. 
        args.resume = resume
        args.epochs = epochs
        args.workers = workers
        
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
    if args.batchnorm:
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

    print("=> fetching train image pairs in '{}\{}'".format(args.data,args.dataset))
    train_set = data_flow.__dict__[args.dataset](
        args.data,
        transform=input_transform,
        target_transform=target_transform,
        co_transform=co_transform
    )
    print("=> fetching test image pairs in '{}\{}'".format(args.data,args.testdataset))
    test_set = data_flow.__dict__[args.testdataset](
        args.data,
        transform=input_transform,
        target_transform=target_transform,
        co_transform=co_validate_transform
    )
    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                           len(train_set),
                                                                           len(test_set)))
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False)

    # create model
    model = models.__dict__[args.arch](network_data,
                                       device         = args.device,
                                       use_checkpoint = True,
                                       lr_finder      = args.lr_finder,
                                       div_flow       = args.div_flow,
                                       batchnorm      = args.batchnorm).to(args.device)

    if args.device.type == "cuda" and args.n_GPUs > 1:
        model = torch.nn.DataParallel(model).cuda()
    cudnn.enabled = True
    cudnn.benchmark = True

    
    assert(args.solver in ['adam','adamw','sgd'])
    if LooseVersion(torch.__version__) < LooseVersion('1.8.0') and args.solver == 'adamw':
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
    
    if args.evaluate:
        eval_writer = SummaryWriter(log_dir=args.save_path / args.run_path)
        args.best_EPE, args.best_Fl, args.best_WAUC = validate(val_loader, model, 0, eval_writer, inverse_transform)
        if LooseVersion(torch.__version__) >= LooseVersion('1.3'):
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
                tensorPreprocessedFirst = torch.nn.functional.interpolate(
                                            input         = inputs[:, 0, :, :, :],
                                            size          = (intPreprocessedHeight, intPreprocessedWidth),
                                            mode          = 'bilinear',
                                            align_corners = False)
                tensorPreprocessedSecond = torch.nn.functional.interpolate(
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

    assert(args.scheduler in ['steplr','onecyclelr','cycliclr','multisteplr','lr_finder'])
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

    train_writer_iterations = SummaryWriter(log_dir=args.save_path / args.run_path / 'iterations', purge_step=args.n_iter)
    train_writer = SummaryWriter(log_dir=args.save_path / args.run_path, filename_suffix='train', purge_step=args.start_epoch)

    if args.start_epoch > args.epochs:
        return

    train_loss = 0
    train_EPE = 0
    train_Fl = 0
    test_EPE = 0
    test_Fl = 0
    test_WAUC = 0
    train_WAUC = 0
    
    # This is just for testing. Disable when not needed
    #torch.autograd.set_detect_anomaly(True)
    try:
        for epoch in range(args.start_epoch, args.epochs):
            print('Epoch: {:d}'.format(epoch))
            # train for one epoch
            train_loss, train_EPE, train_Fl, train_WAUC, lr = train(train_loader, model, optimizer, scheduler, epoch, train_writer, train_writer_iterations)
            train_writer.add_scalar('Train/Loss', train_loss, epoch)
            train_writer.add_scalar('Train/EPE', train_EPE, epoch)
            train_writer.add_scalar('Train/Fl', train_Fl, epoch)
            train_writer.add_scalar('Train/WAUC', train_WAUC, epoch)
            train_writer.add_scalar('Train/lr', lr, epoch)

            # evaluate on validation set
            with torch.no_grad():
                test_EPE, test_Fl, test_WAUC = validate(val_loader, model, epoch, train_writer, inverse_transform)
            train_writer.add_scalar('Test/EPE', test_EPE, epoch)
            train_writer.add_scalar('Test/Fl', test_Fl, epoch)
            train_writer.add_scalar('Test/WAUC', test_WAUC, epoch)

            if args.scheduler in ['steplr','multisteplr']:
                scheduler.step()

            if args.best_EPE < 0:
                args.best_EPE = test_EPE

            is_best = test_WAUC > args.best_WAUC
            args.best_EPE = min(test_EPE, args.best_EPE)
            args.best_Fl = min(test_Fl, args.best_Fl)
            args.best_WAUC = max(test_WAUC, args.best_WAUC)
            args.start_epoch = epoch + 1
            
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
            if is_best:
                model_state_dict = {'state_dict': model.state_dict(),
                                    'args': args}
                torch.save(model_state_dict, args.save_path / args.run_path / (args.arch+'_model_best.pt'))
    finally:
        # Save the final stats as a hparams even if there is a crash
        train_writer.add_hparams({'Batch Size':      '{:1d}'.format(args.batch_size),
                                  'Test Batch Size': '{:1d}'.format(args.test_batch_size),
                                  'Optimizer':       args.solver,
                                  'Scheduler':       args.scheduler,
                                  'Loss Function':   args.loss,
                                  'Dataset':         args.dataset,
                                  'Test Dataset':    args.testdataset,
                                  'div flow':        '{:1d}'.format(args.div_flow),
                                  'Epoch':           args.start_epoch,
                                  'Iterations':      args.n_iter},
                                 metric_dict = {'Test/EPE':    test_EPE,
                                                'Test/Fl':     test_Fl,
                                                'Test/WAUC':   test_WAUC,
                                                'Train/EPE':   train_EPE,
                                                'Train/Fl':    train_Fl,
                                                'Train/WAUC':  train_WAUC,
                                                'Train/Loss':  train_loss},
                                 hparam_domain_discrete={'Batch Size':       ['1','22'],
                                                         'Test Batch Size':  ['1','6'],
                                                         'div flow':     ['1','20','256','512','1024'],
                                                         'Dataset':      ['N/A',
                                                                          'flying_chairs2',
                                                                          'flying_things_final',
                                                                          'flying_things_clean',
                                                                          'flying_things_both',
                                                                          'Viper'],
                                                         'Test Dataset': ['KITTI_2012_noc',
                                                                          'KITTI_2012_occ',
                                                                          'KITTI_2015_noc',
                                                                          'KITTI_2015_occ',
                                                                          'mpi_sintel_final',
                                                                          'mpi_sintel_clean']},
                                 run_name='hparams')




def train(train_loader, model, optimizer, scheduler, epoch, train_writer, train_writer_iterations):
    global args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()
    flow2_Fls = AverageMeter()
    flow2_WAUCs = AverageMeter()
    split2_WAUC = AverageMeter()
    split3_WAUC = AverageMeter()
    split4_WAUC = AverageMeter()
    split5_WAUC = AverageMeter()
    split6_WAUC = AverageMeter()
    split2_EPE = AverageMeter()
    split3_EPE = AverageMeter()
    split4_EPE = AverageMeter()
    split5_EPE = AverageMeter()
    split6_EPE = AverageMeter()
    learningRates = AverageMeter()
    realEPE = utils.EPE()
    realWAUC = utils.WAUC()
    realFl = utils.Fl_KITTI_2015(use_mask=True)
    output0_min = 0
    output0_max = 0
    output1_min = 0
    output1_max = 0
    output2_min = 0
    output2_max = 0
    output3_min = 0
    output3_max = 0
    output4_min = 0
    output4_max = 0
    
    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    # switch to train mode
    model.train()

    frames_total = len(train_loader.dataset)
    frames_completed = 0
    end = time.time()
    with tqdm(total=epoch_size, position=1, bar_format='{desc}', desc='Waiting for first train batch') as tqdm_desc:
        tqdm_train = tqdm(train_loader, total=epoch_size, position=0)
        for i, (inputs, target) in enumerate(tqdm_train):
            if i >= epoch_size:
                tqdm_desc.set_description('Train: EPE {0}\t Fl {1}\t WAUC {2}\t - Train iterations exceeded {3}'.format(
                                          flow2_EPEs,
                                          flow2_Fls,
                                          flow2_WAUCs,
                                          epoch_size))
                break
            # measure data loading time
            data_time.update(time.time() - end)
            frames_completed += target.size(0)
            mask = target[:,2,:,:].to(args.device)
            target = target.to(args.device)
            target[:,:2,:,:] *= mask[:,None,:,:]
            inputs = inputs.to(args.device)

            # compute gradient and do optimization step
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                outputs[0] = outputs[0] * mask[:,None,:,:]
                output0_min = min(output0_min, torch.min(outputs[0]).item())
                output0_max = max(output0_max, torch.max(outputs[0]).item())
                output1_min = min(output1_min, torch.min(outputs[1]).item())
                output1_max = max(output1_max, torch.max(outputs[1]).item())
                output2_min = min(output2_min, torch.min(outputs[2]).item())
                output2_max = max(output2_max, torch.max(outputs[2]).item())
                output3_min = min(output3_min, torch.min(outputs[3]).item())
                output3_max = max(output3_max, torch.max(outputs[3]).item())
                output4_min = min(output4_min, torch.min(outputs[4]).item())
                output4_max = max(output4_max, torch.max(outputs[4]).item())

                if args.split_losses:
                    loss_total, loss_EPE, loss_WAUCl = args.loss_Flow_net(outputs, target)
                    split2_WAUC.update(100 - loss_WAUCl[0], target.size(0))
                    split3_WAUC.update(100 - loss_WAUCl[1], target.size(0))
                    split4_WAUC.update(100 - loss_WAUCl[2], target.size(0))
                    split5_WAUC.update(100 - loss_WAUCl[3], target.size(0))
                    split6_WAUC.update(100 - loss_WAUCl[4], target.size(0))
                    split2_EPE.update(loss_EPE[0], target.size(0))
                    split3_EPE.update(loss_EPE[1], target.size(0))
                    split4_EPE.update(loss_EPE[2], target.size(0))
                    split5_EPE.update(loss_EPE[3], target.size(0))
                    split6_EPE.update(loss_EPE[4], target.size(0))
                    loss_EPE = loss_EPE[0]
                    accuracy_WAUC = 100 - loss_WAUCl[0]
                else:
                    loss_total = args.loss_Flow_net(outputs, target)
                loss_total.backward()
                optimizer.step()
            # OneCycleLR & CyclicLR need the step right after each training batch
            if args.scheduler in ['onecyclelr','cycliclr']:
                scheduler.step()

            with torch.no_grad():
                if not args.split_losses:
                    loss_EPE = realEPE(outputs[0], target)
                    accuracy_WAUC = realWAUC(outputs[0], target)
                loss_Fl = realFl(outputs[0], target)
            
            lr = scheduler.get_last_lr()[0]
            train_writer_iterations.add_scalar('Epoch{}/Loss'.format(epoch), loss_total, args.n_iter)
            train_writer_iterations.add_scalar('Epoch{}/EPE'.format(epoch), loss_EPE, args.n_iter)
            train_writer_iterations.add_scalar('Epoch{}/Fl'.format(epoch), loss_Fl, args.n_iter)
            train_writer_iterations.add_scalar('Epoch{}/WAUC'.format(epoch), accuracy_WAUC, args.n_iter)
            train_writer_iterations.add_scalar('Epoch{}/lr'.format(epoch), lr, args.n_iter)
            
            # record loss and EPE
            losses.update(loss_total, target.size(0))
            flow2_EPEs.update(loss_EPE, target.size(0))
            flow2_Fls.update(loss_Fl, torch.sum(mask).item())
            flow2_WAUCs.update(accuracy_WAUC, torch.sum(mask).item())
            learningRates.update(lr, target.size(0))
            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #if i % args.print_freq == 0:
            #    print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}\t EPE {6}'
            #          .format(epoch, i, epoch_size, batch_time,
            #                  data_time, losses, flow2_EPEs))
            tqdm_desc.set_description('Train: EPE {0}\t Fl {1}\t WAUC {2}'.format(
                                       flow2_EPEs,
                                       flow2_Fls,
                                       flow2_WAUCs))
            tqdm_train.set_postfix_str(s='[{}/{}]'.format(frames_completed,
                                                          frames_total))
            args.n_iter += 1
            #if i+1 >= epoch_size:
            #    break
    #print('args.n_iter',args.n_iter)
    print('batch_time: {:.5f}'.format(batch_time.sum))
    print('outputs[0] [min: {:4f}][max: {:4f}]'.format(output0_min, output0_max))
    print('outputs[1] [min: {:4f}][max: {:4f}]'.format(output1_min, output1_max))
    print('outputs[2] [min: {:4f}][max: {:4f}]'.format(output2_min, output2_max))
    print('outputs[3] [min: {:4f}][max: {:4f}]'.format(output3_min, output3_max))
    print('outputs[4] [min: {:4f}][max: {:4f}]'.format(output4_min, output4_max))


    if args.split_losses:
        train_writer.add_scalars('split_WAUC',
                                 {'flow2': split2_WAUC.avg,
                                  'flow3': split3_WAUC.avg,
                                  'flow4': split4_WAUC.avg,
                                  'flow5': split5_WAUC.avg,
                                  'flow6': split6_WAUC.avg},
                                 epoch)
        train_writer.add_scalars('split_EPE',
                                 {'flow2': split2_EPE.avg,
                                  'flow3': split3_EPE.avg,
                                  'flow4': split4_EPE.avg,
                                  'flow5': split5_EPE.avg,
                                  'flow6': split6_EPE.avg},
                                 epoch)
    return losses.avg, flow2_EPEs.avg, flow2_Fls.avg, flow2_WAUCs.avg, learningRates.avg


def validate(val_loader, model, epoch, run_writer, inverse_transform):
    global args

    batch_time = AverageMeter()
    flow2_EPEs = AverageMeter()
    flow2_Fls = AverageMeter()
    flow2_WAUCs = AverageMeter()
    input_mean = AverageMeter() #Average of means
    input_std = AverageMeter() #Take square root of average of squares
    Fl = 0.0
    target_min = 0
    target_max = 0
    inputs_min = 0
    inputs_max = 0
    output_min = 0
    output_max = 0
    n_registered_pxs = 0.1

    # switch to evaluate mode
    model.eval()

    end = time.time()
    #for i, (mini_batch) in enumerate(val_loader):
    realEPE = utils.EPE()
    realWAUC = utils.WAUC()
    realFl = utils.Fl_KITTI_2015(use_mask=True)
    test_epoch_size = len(val_loader) if args.test_epoch_size == 0 else min(len(val_loader), args.test_epoch_size)
    j = 0
    frames_total = len(val_loader.dataset)
    frames_completed = 0
    with tqdm(total=test_epoch_size, position=1, bar_format='{desc}', desc='Waiting for first test batch') as tqdm_desc:
        tqdm_test = tqdm(val_loader, total=test_epoch_size, position=0)
        #for i, (inputs, target, mask) in enumerate(tqdm_test):
        for i, (inputs, target) in enumerate(tqdm_test):
            if i >= test_epoch_size:
                tqdm_desc.set_description('Train: EPE {0}\t Fl {1}\t WAUC {2}\t - Test iterations exceeded {3}'.format(
                                          flow2_EPEs,
                                          flow2_Fls,
                                          flow2_WAUCs,
                                          epoch_size))
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
            
            #b, N, c, intHeight, intWidth = inputs.size()
            #
            ## Need input image resolution to always be a multiple of 64
            #intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
            #intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))
            #
            #if intPreprocessedWidth == intWidth and intPreprocessedHeight == intHeight:
            #    # Faster but same memory utilization. Without detach it is slower but takes less memory.
            #    tensorPreprocessedFirst = inputs[:, 0, :, :, :].detach()
            #    tensorPreprocessedSecond = inputs[:, 1, :, :, :].detach()
            #else:
            #    tensorPreprocessedFirst = torch.nn.functional.interpolate(
            #                                input         = inputs[:, 0, :, :, :],
            #                                size          = (intPreprocessedHeight, intPreprocessedWidth),
            #                                mode          = 'bilinear',
            #                                align_corners = False)
            #    tensorPreprocessedSecond = torch.nn.functional.interpolate(
            #                                input         = inputs[:, 1, :, :, :],
            #                                size          = (intPreprocessedHeight, intPreprocessedWidth),
            #                                mode          = 'bilinear',
            #                                align_corners = False)

            output = model(inputs)
            #output = model(tensorPreprocessedFirst, tensorPreprocessedSecond)
            #output = torch.nn.functional.interpolate(
            #            input         = output,
            #            size          = (intHeight, intWidth),
            #            mode          = 'bilinear',
            #            align_corners = False)
            #if intPreprocessedWidth != intWidth or intPreprocessedHeight != intHeight:
            #    output[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
            #    output[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
            output_min = min(output_min, torch.min(output).item())
            output_max = max(output_max, torch.max(output).item())
            target_min = min(target_min, torch.min(target).item())
            target_max = max(target_max, torch.max(target).item())
            inputs_min = min(inputs_min, torch.min(inputs).item())
            inputs_max = max(inputs_max, torch.max(inputs).item())
            #print('target.size',target.size())
            #print('target [min: {:4f}][max: {:4f}]'.format(target_min, target_max))
            #print('inputs [min: {:4f}][max: {:4f}]'.format(inputs_min, inputs_max))
            #print('output [min: {:4f}][max: {:4f}]'.format(output_min, output_max))
            
            # compute output
            flow2_EPE = realEPE(output*mask[:,None,:,:], target)
            flow2_Fl = realFl(output, target, mask)
            flow2_WAUC = realWAUC(output, target, mask)
            
            # record EPE
            flow2_EPEs.update(flow2_EPE.item(), target.size(0))
            flow2_Fls.update(flow2_Fl.item(), torch.sum(mask).item())
            flow2_WAUCs.update(flow2_WAUC.item(), torch.sum(mask).item())
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # log first output of first batches
            if (i * args.test_batch_size) % args.image_freq == 0 and run_writer is not None and j < 3:
                frame = args.test_batch_size * i
                
                if epoch == 0:
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
                        run_writer.add_graph(model, inputs)

                output_flow_image = flow_vis.flow_to_color(output[0,:2,:,:].permute((1,2,0)).detach().cpu().numpy(),
                                                           clip_flow=None).transpose((2, 0, 1))
                run_writer.add_image('Output{}'.format(frame),
                                     output_flow_image,
                                     epoch)
                j += 1

            tqdm_desc.set_description('Test: EPE {0}\t Fl {1}\t WAUC {2}'.format(
                                       flow2_EPEs,
                                       flow2_Fls,
                                       flow2_WAUCs))
            tqdm_test.set_postfix_str(s='[{}/{}]'.format(frames_completed,
                                                         frames_total))
            #if i % args.print_freq == 0:
            #    print('Test: [{0}/{1}]\t Time {2}\t EPE {3}\t Fl {4}\t'
            #          .format(i, len(val_loader), batch_time, flow2_EPEs, flow2_Fls))

    print('batch_time: {:.5f}'.format(batch_time.sum))
    print('target [min: {:4f}][max: {:4f}]'.format(target_min, target_max))
    print('inputs [min: {:4f}][max: {:4f}]'.format(inputs_min, inputs_max))
    print('output [min: {:4f}][max: {:4f}]'.format(output_min, output_max))
    
    #https://www.statology.org/averaging-standard-deviations/
    #Note that the standard deviation is the square root of the variance
    #print('input [mean: {:4f}][std: {:4f}][var: {:4f}]'.format(input_mean.avg, np.sqrt(input_std.avg), input_std.avg))

    return flow2_EPEs.avg, flow2_Fls.avg, flow2_WAUCs.avg


if __name__ == '__main__':
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
        if platform.system() == 'Windows':
            env = 'CUDA_LAUNCH_BLOCKING'
            if env in os.environ:
                del os.environ[env]
        print("Closed cleanly")
