import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from datetime import datetime
from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter
import matplotlib
import matplotlib.pyplot as plt


class CustomTrainIter(TrainDataLoaderIter):
    def inputs_labels_from_batch(self, input):
        r""" class for LRFinder
            This is to adjust the data from the loader so that it can be used by LRFinder
        """
        input_gpu = input[0]
        gt = input[1]
        n_sequence = len(input_gpu[0])
        gt_gpu = gt[:, n_sequence // 2, :, :, :]
        return input_gpu, gt_gpu

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
            if self.args.lr_finder_Leslie_Smith:
                custom_val_iter = CustomValIter(self.loader_test)
                lr_finder = LRFinder(self.model, self.optimizer, self.loss, device="cuda")
                lr_finder.range_test(custom_train_iter,
                                     val_loader = custom_val_iter,
                                     end_lr     = self.args.max_lr,
                                     num_iter   = 1000,
                                     step_mode  = "linear")
                lr_finder.plot()
            else:
                lr_finder = LRFinder(self.model, self.optimizer, self.loss, device="cuda")
                lr_finder.range_test(custom_train_iter,
                                     end_lr   = self.args.max_lr,
                                     num_iter = 1000)
                lr_finder.plot()
            plt.savefig('{}/LRvsLoss.png'.format(args.experiment_dir + args.save), dpi=300)
            plt.close()
            #lr_finder.reset()

        self.scheduler = self.make_scheduler()
        self.ckp = ckp

        if args.load != '.':
            self.optimizer.load_state_dict(torch.load(os.path.join(ckp.dir, 'optimizer.pt')))
            self.scheduler.load_state_dict(torch.load(os.path.join(ckp.dir, 'scheduler.pt')))
            checkpoint = torch.load(os.path.join(ckp.dir, 'checkpoint.tar'))
            self.args.total_train_time = checkpoint['total_train_time']
            self.args.total_test_time  = checkpoint['total_test_time']
            self.args.epochs_completed = checkpoint['epochs_completed']
            rng_state                  = checkpoint['rng_state']
            torch.set_rng_state(rng_state)
            self.args.start_time = datetime.now() - (self.args.total_test_time + self.args.total_train_time)
            print('total_train_time: {}'.format(self.args.total_train_time), file=ckp.config_file)
            print('total_test_time: {}'.format(self.args.total_test_time), file=ckp.config_file)
            print('epochs_completed: {}'.format(self.args.epochs_completed), file=ckp.config_file)
            print('start_time: {}'.format(self.args.start_time), file=ckp.config_file)

    def make_optimizer(self):
        if self.args.Adam:
            kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
            return optim.Adam(self.model.parameters(), **kwargs)
        else:
            kwargs = {'lr':             self.args.lr,
                      'weight_decay':   self.args.AdamW_weight_decay}
            return optim.AdamW(self.model.parameters(), **kwargs)

    def make_scheduler(self):
        if self.args.StepLR:
            kwargs = {'step_size': self.args.lr_decay, 'gamma': self.args.gamma}
            return lr_scheduler.StepLR(self.optimizer, **kwargs)
        else:
            div_factor = self.args.max_lr / self.args.lr
            kwargs = {'max_lr':          self.args.max_lr,
                      'epochs':          self.args.epochs,
                      'div_factor':      div_factor,
                      'steps_per_epoch': len(self.loader_train)}
            return lr_scheduler.OneCycleLR(self.optimizer, **kwargs)

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
