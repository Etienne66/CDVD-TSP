import os
from importlib import import_module
import numpy as np
import torch
import torch.nn as nn
from loss.hard_example_mining import HEM
from datetime import datetime
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from loss.ssim import MS_SSIM_LOSS

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        device = torch.device('cpu' if args.cpu else 'cuda')

        self.n_GPUs = args.n_GPUs
        self.lr_finder = args.lr_finder
        self.loss = []
        self.loss_module = nn.ModuleList()

        if args.LossL1HEM:
            loss_string = args.loss
        else:
            loss_string = args.loss_MSL

        for loss in loss_string.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'HEM':
                loss_function = HEM(device=device)
            elif loss_type == 'MSL':
                loss_function = MS_SSIM_LOSS(data_range=args.rgb_range, channel=args.n_channel)
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')()
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(args, loss_type)
            else:
                raise NotImplementedError('Loss type [{:s}] is not found'.format(loss_type))

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        self.loss_module.to(device)

        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load != '.':
            self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr):
        losses = []
        start_time = datetime.now()
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                start_time_function = datetime.now()
                loss = l['function'](sr, hr)
                losses.append(l['weight'] * loss)
                if not self.lr_finder:
                    self.log[-1, i, 0] += loss.item()
                    self.log[-1, i, 1] = l['weight']
                    self.log[-1, i, 2] += (datetime.now() - start_time_function).total_seconds()
            elif l['type'] == 'DIS' and not self.lr_finder:
                start_time_function = datetime.now()
                self.log[-1, i, 0] += self.loss[i - 1]['function'].loss
                self.log[-1, i, 1] = l['weight']
                self.log[-1, i, 2] += (datetime.now() - start_time_function).total_seconds()

        loss_sum = sum(losses)
        if len(self.loss) > 1 and not self.lr_finder:
            self.log[-1, -1, 0] += loss_sum.item()
            self.log[-1, -1, 1] = 0
            self.log[-1, -1, 2] += (datetime.now() - start_time).total_seconds()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros([1, len(self.loss), 3])))

    def end_log(self, n_batches):
        self.log[-1,:,0].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            if l['type'] == 'MSL':
                log.append('[{}: {:.2f}(1 - {:.6f})]'.format('MS-SSIM', l['weight'], 1 - (c[0] / n_samples)))
            elif l['type'] == 'Total':
                log.append('[{}: {:.6f}]'.format(l['type'], c[0] / n_samples))
            else:
                log.append('[{}: {:.2f}({:.6f})]'.format(l['type'], l['weight'], c[0] / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        if epoch > 1:
            axis = np.linspace(1, epoch, epoch)
            #fig = plt.figure()
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            plt.title('Loss Functions')
            for i, l in enumerate(self.loss):
                loss_label = '{} Loss'.format(l['type'])
                weight_label = '{} Weight'.format(l['type'])
                ax1.plot(axis, self.log[:, i, 0].numpy(), label=loss_label)
                if l['type'] != 'Total':
                    ax2.plot(axis, self.log[:, i, 1].numpy(), ':',label=weight_label)
                if l['type'] == 'MSL':
                    #weight_label = 'MS-SSIM'
                    #ax2.plot(axis, 1 - self.log[:, i, 0].numpy(), '--',label=weight_label)
                    MSSSIM = 1 - self.log[:, i, 0].numpy()
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            plt.legend(lines + lines2, labels + labels2)
            #plt.legend(handles = (lines,lines2),
            #           labels = (labels,labels2),
            #           loc = 'best')
            plt.xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax2.set_ylabel('Weight') 
            ax1.grid(True)
            #plt.grid(True)
            fig.tight_layout()
            plt.savefig('{}/loss.png'.format(apath), dpi=300)
            plt.close(fig)
            
            if MSSSIM is not None:
                label = 'MS-SSIM'
                fig = plt.figure()
                plt.title(label)
                plt.plot(axis, MSSSIM, label=label)
                plt.xlabel('Epochs')
                plt.ylabel(label)
                plt.grid(True)
                plt.savefig('{}/{}.png'.format(apath, label), dpi=300)  
                plt.close(fig)

            #axis = np.linspace(1, epoch, epoch)
            #for i, l in enumerate(self.loss):
            #    label = '{} Loss'.format(l['type'])
            #    fig = plt.figure()
            #    plt.title(label)
            #    plt.plot(axis, self.log[:, i].numpy(), label=label)
            #    plt.legend()
            #    plt.xlabel('Epochs')
            #    plt.ylabel('Loss')
            #    plt.grid(True)
            #    plt.savefig('{}/loss_loss_{}.pdf'.format(apath, l['type']))            
            #    plt.close(fig)
            
            
            
            
    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss_state.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss_state.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

