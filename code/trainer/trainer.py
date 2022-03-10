import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from datetime import datetime
from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
from pathlib import Path

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


class Trainer:
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = self.make_optimizer()

        if self.args.lr_finder:
            custom_train_iter = CustomTrainIter(self.loader_train)
            fig_kw = {'figsize': [38.4,21.6]}
            plt.rcParams.update({'font.size': 20})
            plt.ticklabel_format(axis="x",style="sci",scilimits=(0,0))
            if self.args.lr_finder_Leslie_Smith:
                custom_val_iter = CustomValIter(self.loader_test)
                lr_finder = LRFinder(self.model, self.optimizer, self.loss, device="cuda")
                lr_finder.range_test(custom_train_iter,
                                     val_loader = custom_val_iter,
                                     end_lr     = self.args.max_lr,
                                     num_iter   = 100,
                                     step_mode  = "linear")
                fig, ax = plt.subplots(**fig_kw)
                lr_finder.plot(ax=ax)
            else:
                lr_finder = LRFinder(self.model, self.optimizer, self.loss, device="cuda")
                lr_finder.range_test(custom_train_iter,
                                     end_lr   = self.args.max_lr,
                                     num_iter = 1000)
                fig, ax = plt.subplots(**fig_kw)
                lr_finder.plot(ax=ax)
            plt.grid(True)
            fig.tight_layout()
            fig.savefig(args.experiment_dir / args.save / 'LRvsLoss.png', dpi=100)
            plt.close()
            #lr_finder.reset()
        else:
            self.scheduler = self.make_scheduler()
            self.ckp = ckp

            if args.load is not None:
                self.optimizer.load_state_dict(torch.load(ckp.dir / 'optimizer.pt'))
                self.scheduler.load_state_dict(torch.load(ckp.dir / 'scheduler.pt'))
                checkpoint = torch.load(ckp.dir / 'checkpoint.tar')
                self.args.total_train_time = checkpoint['total_train_time']
                self.args.total_test_time  = checkpoint['total_test_time']
                self.args.epochs_completed = checkpoint['epochs_completed']
                rng_state                  = checkpoint['rng_state']
                random_state               = checkpoint['random_state']
                numpy_random_state         = checkpoint['numpy_random_state']
                torch.set_rng_state(rng_state)
                random.setstate(random_state)
                np.random.set_state(numpy_random_state)
                self.args.start_time = datetime.now() - (self.args.total_test_time + self.args.total_train_time)
                print('total_train_time: {}'.format(self.args.total_train_time), file=ckp.config_file)
                print('total_test_time: {}'.format(self.args.total_test_time), file=ckp.config_file)
                print('epochs_completed: {}'.format(self.args.epochs_completed), file=ckp.config_file)
                print('start_time: {}'.format(self.args.start_time), file=ckp.config_file)
                print('patch_size: {}'.format(self.args.patch_size), file=ckp.config_file)
                ckp.config_file.close()


    def make_optimizer(self):
        if self.args.Adam:
            kwargs = {'lr':             self.args.lr,
                      'weight_decay':   self.args.weight_decay}
            return optim.Adam(self.model.parameters(), **kwargs)
        else:
        #elif self.args.OneCycleLR or self.args.StepLR:
            kwargs = {'lr':             self.args.lr,
                      'weight_decay':   self.args.AdamW_weight_decay}
            return optim.AdamW(self.model.parameters(), **kwargs)
        #else:
        #    kwargs = {'lr':           self.args.lr,
        #              'weight_decay': self.args.weight_decay,
        #              'momentum':     0.9}
        #    return optim.SGD(self.model.parameters(), **kwargs)

    def make_scheduler(self):
        if self.args.StepLR:
            kwargs = {'step_size': self.args.lr_decay,
                      'gamma':     self.args.gamma}
            return lr_scheduler.StepLR(self.optimizer, **kwargs)
        elif self.args.OneCycleLR:
            div_factor = self.args.max_lr / self.args.lr
            kwargs = {'max_lr':          self.args.max_lr,
                      'epochs':          self.args.epochs,
                      'div_factor':      div_factor,
                      'steps_per_epoch': len(self.loader_train)}
            return lr_scheduler.OneCycleLR(self.optimizer, **kwargs)
        else:
            kwargs = {'base_lr':        self.args.lr,
                      'max_lr':         self.args.max_lr,
                      'cycle_momentum': False, #needs to be False for Adam/AdamW
                      'step_size_up':   self.args.step_size_up}
            return lr_scheduler.CyclicLR(self.optimizer, **kwargs)

    def train(self):
        pass

    def test(self):
        pass

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.args.epochs_completed
            return epoch >= self.args.epochs
