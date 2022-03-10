import argparse
import os
from pathlib import Path
import time
import datetime
import math

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
parser.add_argument('--dataset', metavar='DATASET', default='KITTI_2015_occ', #flying_things_both
                    choices=dataset_names,
                    help='dataset type : ' + '|'.join(dataset_names))
parser.add_argument('--testdataset', metavar='DATASET', default='mpi_sintel_clean', #mpi_sintel_final mpi_sintel_clean
                    choices=dataset_names,
                    help='dataset type : ' + '|'.join(dataset_names))
parser.add_argument('--arch', '-a', metavar='ARCH', default='flow_pwc',
                    choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' + '|'.join(model_names))
parser.add_argument('--n_GPUs', type=int, default=1, metavar='N',
                    help='number of GPUs')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch_size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch_size', default=2, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--print_freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--image_freq', default=60, type=int,
                    metavar='N', help='test image frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', type=Path, default=None,
                    help='path to pre-trained model')
parser.add_argument('--no_date', action='store_true',
                    help='don\'t append date timestamp to folder' )
parser.add_argument('--div_flow', default=20, metavar='N',
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')

OptimizerGroup = parser.add_argument_group('Optimizer', description='Optimizer options')
OptimizerGroup.add_argument('--solver', default='adam',choices=['adam','adamw','sgd'],
                    help='solver algorithms')
OptimizerGroup.add_argument('--lr', '--learning-rate', default=2e-7, type=float,
                    metavar='LR', help='initial learning rate')
OptimizerGroup.add_argument('--max_lr', default=7e-5, type=float, metavar='M',
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
SchedulerGroup.add_argument('--step_size_up', type=int, default=2000, metavar='N',
                    help='Number of training iterations in the increasing half of a cycle.')
SchedulerGroup.add_argument('--gamma', type=float, default=0.5, metavar='M',
                    help='learning rate decay factor for step decay')
SchedulerGroup.add_argument('--CyclicLR_gamma', type=float, default=0.994, metavar='M',
                    help='learning rate decay factor for step decay')
SchedulerGroup.add_argument('--CyclicLR_mode', type=str, default='exp_range',
                    choices=['triangular','triangular2','exp_range'],
                    help='learning rate policy')
SchedulerGroup.add_argument('--milestones', default=[100,150,200], metavar='N', nargs='*',
                    help='epochs at which learning rate is divided by 2')
SchedulerGroup.add_argument('--lr_finder_Leslie_Smith', action='store_true',
                    help='Run the LRFinder using Leslie Smith''s approach')







def main():
    global args
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cpu = False
    args.lr_finder = False
    args.n_sequence = 2
    args.LossL1HEM = True
    args.load = None
    args.loss = '1*EPE+1*F1'
    if args.pretrained:
        network_data = torch.load(args.pretrained, map_location=torch.device(args.device))
        if 'state_dict' not in network_data.keys():
            #for key in network_data.keys():
            #    print(key)
            network_data = {'state_dict': network_data}
            #print(network_data.keys())
            args.arch = 'flow_pwc'
            save_path = Path('../runs/' + datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S") + args.arch)
        elif 'args' in network_data.keys():
            args = network_data['args']
            # Reset device in case the model was saved with a different device
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Reset random number generators
            torch.set_rng_state(network_data['rng_state'])
            random.setstate(network_data['random_state'])
            np.random.set_state(network_data['numpy_random_state'])
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        exit()
        args.best_EPE = -1
        args.n_iter = int(args.start_epoch)
        
        network_data = None
        print("=> creating model '{}'".format(args.arch))
        save_path = '{},{},{}epochs{},b{},lr{}'.format(
            args.arch,
            args.solver,
            args.epochs,
            ',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
            args.batch_size,
            args.lr)
        if not args.no_date:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
            save_path = os.path.join(timestamp,save_path)
        save_path = os.path.join(args.dataset,save_path)

    print('=> will save everything to {}'.format(save_path))
    if not save_path.exists():
        save_path.mkdir(parents=True)

    train_writer = SummaryWriter(log_dir=save_path / 'train')
    train_writer_epoch = SummaryWriter(log_dir=save_path / 'train_epoch')
    test_writer_epoch = SummaryWriter(log_dir=save_path / 'test_epoch')
    #train_writer = None
    #train_writer_epoch = None
    #test_writer_epoch = None
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(log_dir=save_path / 'test_epoch' / str(i)))
        #output_writers.append(None)
    loss_fn = loss.Loss(args)

    # Data loading code
    validate_transform = flow_transforms.Compose([
        flow_transforms.CenterCrop((320,768))
    ])
    # tranforms.normalize output[channel] = (input[channel] - mean[channel]) / std[channel]
    # All channels are RGB. Loads from OpenCV were corrected
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255])#, # (0,255) -> (0,1) 
        #transforms.Normalize(mean=[0.45,0.432,0.411], std=[1,1,1]) # (0,1) -> (-0.5,0.5)
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # from ImageNet dataset
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Standard normalization
    ])
    target_transform = transforms.Compose([
        flow_transforms.ArrayToTensor()#,
        #transforms.Normalize(mean=[0,0],std=[args.div_flow,args.div_flow])
    ])

    co_transform = flow_transforms.Compose([
        #flow_transforms.RandomTranslate(10),
        flow_transforms.RandomCrop((512,512)),
        flow_transforms.RandomRot90(), #flow_transforms.RandomRotate(10,5),
        flow_transforms.RandomVerticalFlip(), #transforms.RandomVerticalFlip,
        flow_transforms.RandomHorizontalFlip() #transforms.RandomHorizontalFlip()
    ])

    print("=> fetching img pairs in '{}'".format(args.data))
    train_set = data_flow.__dict__[args.dataset](
        args.data,
        transform=input_transform,
        target_transform=target_transform,
        co_transform=co_transform
    )
    test_set = data_flow.__dict__[args.testdataset](
        args.data,
        transform=input_transform,
        target_transform=target_transform,
        co_transform=validate_transform
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
        batch_size=1,#args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False)

    # create model
    model = models.__dict__[args.arch](network_data).to(args.device)

    if args.device.type == "cuda" and args.n_GPUs > 1:
        model = torch.nn.DataParallel(model).cuda()
    cudnn.enabled = True
    cudnn.benchmark = True

    if args.evaluate:
        #print(type(args.testdataset))
        #return
        args.best_EPE, args.best_F1 = validate(val_loader, model, 0, output_writers)
        if float(torch.__version__[:3]) >= 1.3:
            test_writer_epoch.add_hparams({'lr': args.lr, 'bsize': args.batch_size, 'dataset': args.testdataset},
                                          {'hparam/EPE':     args.best_EPE,
                                           #'hparam/Valid_EPE':     Valid_EPE,
                                           'hparam/F1':      args.best_F1,
                                           'hparam/Epoch':   args.start_epoch})

        return

    assert(args.solver in ['adam','adamw','sgd'])
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


    assert(args.scheduler in ['steplr','onecyclelr','cycliclr','multisteplr'])
    if args.scheduler == 'multisteplr':
        kwargs = {'milestones': args.milestones,
                  'gamma':      args.gamma}
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif args.scheduler == 'steplr':
        kwargs = {'step_size': args.lr_decay,
                  'gamma':     args.gamma}
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif args.scheduler == 'onecyclelr':
        div_factor = self.args.max_lr / self.args.lr
        kwargs = {'max_lr':          args.max_lr,
                  'epochs':          args.epochs,
                  'div_factor':      div_factor,
                  'steps_per_epoch': len(train_loader)}
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
    elif args.scheduler == 'cycliclr':
        args.step_size_up = len(train_loader) if args.step_size_up == 0 else min(len(train_loader), args.step_size_up)
        kwargs = {'base_lr':        args.lr,
                  'max_lr':         args.max_lr,
                  'cycle_momentum': False, #needs to be False for Adam/AdamW
                  'step_size_up':   args.step_size_up,
                  'mode':           args.CyclicLR_mode,
                  'gamma':          args.CyclicLR_gamma}
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **kwargs)
    elif args.scheduler == 'lr_finder':
        #Need to crop video so that it doesn't need to be resized
        custom_train_iter = CustomTrainIter(train_loader)
        num_iter = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)
        if args.lr_finder_Leslie_Smith:
            custom_val_iter = CustomValIter(loader_test)
            lr_finder = LRFinder(model, optimizer, loss, device="cuda")
            lr_finder.range_test(custom_train_iter,
                                 val_loader = custom_val_iter,
                                 end_lr     = args.max_lr,
                                 num_iter   = num_iter,
                                 step_mode  = "linear")
        else:
            lr_finder = LRFinder(model, optimizer, loss, device="cuda")
            lr_finder.range_test(custom_train_iter,
                                 end_lr   = args.max_lr,
                                 num_iter = num_iter)

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()

        # train for one epoch
        train_loss, train_EPE = train(train_loader, model, optimizer, scheduler, epoch, train_writer)
        train_writer.add_scalar('mean EPE', train_EPE, epoch)

        # evaluate on validation set
        with torch.no_grad():
            EPE, F1 = validate(val_loader, model, epoch, output_writers)
        test_writer_epoch.add_scalar('mean EPE', EPE, epoch)

        if args.best_EPE < 0:
            args.best_EPE = EPE

        is_best = EPE < args.best_EPE
        args.best_EPE = min(EPE, args.best_EPE)
        args.start_epoch = epoch + 1
        rng_state = torch.get_rng_state()
        random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        state = {'state_dict': model.module.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict(),
                 'args': args,
                 'rng_state': rng_state,
                 'random_state': random_state,
                 'numpy_random_state': numpy_random_state}
        torch.save(state, save_path / 'checkpoint.pt')
        if is_best:
            state = {'state_dict': model.module.state_dict(),
                     'args': args}
            torch.save(state, save_path / (args.arch+'model_best.pt'))
    
    test_writer_epoch.add_hparams({'lr': args.lr, 'bsize': args.batch_size, 'dataset': args.testdataset},
                                  {'hparam/EPE':     EPE,
                                   'hparam/F1':      F1,
                                   'hparam/Epoch':   epoch})


def train(train_loader, model, optimizer, epoch, train_writer):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()

    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.to(args.device)
        input = torch.cat(input,1).to(args.device)

        # compute output
        output = model(input)

        loss = multiscaleEPE(output, target, weights=args.multiscale_weights, sparse=args.sparse)
        #loss = 
        flow2_EPE = args.div_flow * realEPE(output[0], target, sparse=args.sparse)
        # record loss and EPE
        losses.update(loss.item(), target.size(0))
        train_writer.add_scalar('train_loss', loss.item(), n_iter)
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}\t EPE {6}'
                  .format(epoch, i, epoch_size, batch_time,
                          data_time, losses, flow2_EPEs))
        n_iter += 1
        if i >= epoch_size:
            break

    return losses.avg, flow2_EPEs.avg


def validate(val_loader, model, epoch, output_writers):
    global args

    batch_time = AverageMeter()
    flow2_EPEs = AverageMeter()
    flow2_Alt_EPEs = AverageMeter()
    flow2_valid_EPEs = AverageMeter()
    flow2_F1s = AverageMeter()
    flow2_Alt_F1s = AverageMeter()
    input_mean = AverageMeter() #Average of means
    input_std = AverageMeter() #Take square root of average of squares
    F1 = 0.0
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
    realAltEPE = utils.EPE(mean=False)
    realEPE = utils.EPE()
    realF1  = utils.F1_KITTI_2015()
    j = 0
    for i, (inputs, target, mask) in enumerate(val_loader):
        mask = mask.to(args.device)
        target = target.to(args.device)
        inputs[0] = inputs[0].to(args.device)
        inputs[1] = inputs[1].to(args.device)
        input_mean.update(torch.mean(inputs[0]).item())
        input_mean.update(torch.mean(inputs[1]).item())
        input_std.update(np.square(torch.mean(inputs[0]).item()))
        input_std.update(np.square(torch.mean(inputs[1]).item()))
        
        b, c, intHeight, intWidth = inputs[0].size()
        
        # Need input image resolution to always be a multiple of 64
        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))
        
        if intPreprocessedWidth == intWidth and intPreprocessedHeight == intHeight:
            # Faster but same memory utilization. Without detach it is slower but takes less memory.
            tensorPreprocessedFirst = inputs[0].detach()
            tensorPreprocessedSecond = inputs[1].detach()
        else:
            tensorPreprocessedFirst = torch.nn.functional.interpolate(
                                        input         = inputs[0],
                                        size          = (intPreprocessedHeight, intPreprocessedWidth),
                                        mode          = 'bilinear',
                                        align_corners = False)
            tensorPreprocessedSecond = torch.nn.functional.interpolate(
                                        input         = inputs[1],
                                        size          = (intPreprocessedHeight, intPreprocessedWidth),
                                        mode          = 'bilinear',
                                        align_corners = False)

        output = model(tensorPreprocessedFirst, tensorPreprocessedSecond)
        output = torch.nn.functional.interpolate(
                    input         = output,
                    size          = (intHeight, intWidth),
                    mode          = 'bilinear',
                    align_corners = False)
        if intPreprocessedWidth == intWidth or intPreprocessedHeight == intHeight:
            output[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
            output[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
        output *= args.div_flow
        output_min = min(output_min, torch.min(output).item())
        output_max = max(output_max, torch.max(output).item())
        target_min = min(target_min, torch.min(target).item())
        target_max = max(target_max, torch.max(target).item())
        inputs_min = min(inputs_min, torch.min(inputs[0]).item(), torch.min(inputs[1]).item())
        inputs_max = max(inputs_max, torch.max(inputs[0]).item(), torch.max(inputs[1]).item())
        
        flow2_EPE = realEPE(output*mask[:,None,:,:], target*mask[:,None,:,:])
        flow2_Alt_EPE = realAltEPE(output*mask[:,None,:,:], target*mask[:,None,:,:]).sum()
        
        # record EPE
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))
        flow2_Alt_EPEs.update(flow2_Alt_EPE.item()/torch.sum(mask).item(), torch.sum(mask).item())
        
        flow2_F1 = realF1(output*mask[:,None,:,:], target*mask[:,None,:,:])
        flow2_F1s.update(flow2_F1.item()/torch.sum(mask).item(), torch.sum(mask).item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.image_freq == 0:
            if j  < len(output_writers):  # log first output of first batches
                if epoch == args.start_epoch:
                    #mean_values = torch.tensor([0.45,0.432,0.411], dtype=inputs[0].dtype).view(3,1,1)
                    mean_values = torch.tensor([0,0,0], dtype=inputs[0].dtype).view(3,1,1)
                    target_flow_image = flow_vis.flow_to_color(target[0].permute((1,2,0)).detach().cpu().numpy(), clip_flow=None).transpose((2, 0, 1))
                    output_writers[j].add_image('GroundTruth', target_flow_image, 0)
                    output_writers[j].add_image('input from',(inputs[0][0,:3].cpu() + mean_values).clamp(0,1), 0)
                    output_writers[j].add_image('input to', (inputs[1][0,:3].cpu() + mean_values).clamp(0,1), 0)
                    if i == 0:
                        output_writers[j].add_graph(model, (tensorPreprocessedFirst, tensorPreprocessedSecond))
                output_flow_image = flow_vis.flow_to_color(output[0].permute((1,2,0)).detach().cpu().numpy(), clip_flow=None).transpose((2, 0, 1))
                output_writers[j].add_image('FlowNet Outputs', output_flow_image, epoch)
            j += 1

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t Time {2}\t EPE {3}\t Valid EPE {4}\t Alt EPE {5}\t F1 {6}\t'
                  .format(i, len(val_loader), batch_time, flow2_EPEs, flow2_valid_EPEs, flow2_Alt_EPEs, flow2_F1s))

    print(' * EPE {:.5f}'.format(flow2_EPEs.avg))
    print(' * Alt EPE {:.5f}'.format(flow2_Alt_EPEs.avg))
    print(' * F1.avg {:.5%}'.format(flow2_F1s.avg))
    print('target_min',target_min)
    print('target_max',target_max)
    print('inputs_min',inputs_min)
    print('inputs_max',inputs_max)
    print('output_min',output_min)
    print('output_max',output_max)
    print('input_mean',input_mean.avg)
    print('input_std',np.sqrt(input_std.avg))#https://www.statology.org/averaging-standard-deviations/
    print('input_var',input_std.avg)#Note that the standard deviation is the square root of the variance

    return flow2_EPEs.avg, flow2_F1s.avg


class CustomTrainIter(TrainDataLoaderIter):
    def inputs_labels_from_batch(self, input):
        r""" class for LRFinder
            This is to adjust the data from the loader so that it can be used by LRFinder
        """
        input_gpu = input[0]
        gt = input[1]
        n_sequence = len(input_gpu[0])
        gt_list = [gt[:, i, :, :, :] for i in range(n_sequence)]
        if n_sequence == 3:
            gt_cat = torch.cat([gt_list[1]], dim=1)
        elif n_sequence == 5:
            gt_cat = torch.cat([gt_list[1], gt_list[2], gt_list[3],
                                gt_list[2]], dim=1)
        elif n_sequence == 7:
            gt_cat = torch.cat([gt_list[1], gt_list[2], gt_list[3], gt_list[4], gt_list[5],
                                gt_list[2], gt_list[3], gt_list[4],
                                gt_list[3]], dim=1)
        return input_gpu, gt_cat


class CustomValIter(ValDataLoaderIter):
    def inputs_labels_from_batch(self, input):
        r""" class for LRFinder
            This is to adjust the data from the loader so that it can be used by LRFinder
        """
        input_gpu = input[0]
        gt = input[1]
        n_sequence = len(input_gpu[0])
        gt_gpu = gt[:, n_sequence // 2, :, :, :]
        return input_gpu, gt_gpu


if __name__ == '__main__':
    main()
