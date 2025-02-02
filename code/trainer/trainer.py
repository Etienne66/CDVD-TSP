import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from datetime import datetime

class Trainer:
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = self.make_optimizer()
        self.scheduler = self.make_scheduler()
        self.ckp = ckp

        if args.load != '.':
            self.optimizer.load_state_dict(torch.load(os.path.join(ckp.dir, 'optimizer.pt')))
            for _ in range(len(ckp.psnr_log)):
                # Starting with PyTorch 1.1.0 and later optimizer.step() must be called before scheduler.step()
                self.optimizer.step()
                self.scheduler.step()
            checkpoint = torch.load(os.path.join(ckp.dir, 'checkpoint.tar'))
            self.args.total_train_time = checkpoint['total_train_time']
            self.args.total_test_time  = checkpoint['total_test_time']
            self.args.epochs_completed = checkpoint['epochs_completed']
            self.args.start_time = datetime.now() - (self.args.total_test_time + self.args.total_train_time)

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        return optim.Adam(self.model.parameters(), **kwargs)

    def make_scheduler(self):
        kwargs = {'step_size': self.args.lr_decay, 'gamma': self.args.gamma}
        return lr_scheduler.StepLR(self.optimizer, **kwargs)

    def train(self):
        pass

    def test(self):
        pass

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.args.epochs
