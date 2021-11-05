import os
from importlib import import_module
import numpy as np
import torch
import torch.nn as nn
from loss.hard_example_mining import HEM, HEM_MSSIM_L1
from datetime import datetime
import time
from pathlib import Path

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
from loss.ssim import MS_SSIM
import traceback

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        device = torch.device('cpu' if args.cpu else 'cuda')

        self.n_GPUs = args.n_GPUs
        self.lr_finder = args.lr_finder
        self.loss = []
        self.loss_module = nn.ModuleList()
        frames_per_stage_list = [0,1,5,9]
        self.stages = (args.n_sequence // 2)
        self.frames_per_stage = frames_per_stage_list[self.stages]

        if self.stages == 1: 
            self.stage_average = [1]
            self.stage_list = [[0]]
        elif self.stages == 2: 
            self.stage_average = [3,1]
            self.stage_list = [[0,1,2],
                               [3]]
        elif self.stages == 3: 
            self.stage_average = [5,3,1]
            self.stage_list = [[0,1,2,3,4],
                               [5,6,7],
                               [8]]

        if args.LossL1HEM:
            loss_string = args.loss
        elif args.LossMslL1:
            loss_string = args.loss_MSL
        else:
            loss_string = args.loss_HEM_MSL

        for loss in loss_string.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'HEM':
                loss_function = HEM(device=device)
            elif loss_type == 'MSL':
                loss_function = MS_SSIM(data_range=args.rgb_range, channel=args.n_channel)
            elif loss_type == 'HML':
                loss_function = HEM_MSSIM_L1(device=device, rbg_range=args.rgb_range, channel=args.n_channel)
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

        self.loss_module.to(device=device, memory_format=torch.contiguous_format)

        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load is not None:
            self.load(ckp.dir, cpu=args.cpu)


    def forward(self, sr, hr):
        output_images = torch.chunk(sr, self.frames_per_stage, dim=1)
        gt_images = torch.chunk(hr, self.frames_per_stage, dim=1)
        loss_sum = []
        for stage in range(self.stages):
            losses = []
            for i, l in enumerate(self.loss):
                if l['function'] is not None:
                    loss = 0
                    for n in self.stage_list[stage]:
                        if l['type'] == 'MSL':
                            loss += 1 - l['function'](output_images[n], gt_images[n])
                        else:
                            loss += l['function'](output_images[n], gt_images[n])
                    loss /= self.stage_average[stage]
                    losses.append(l['weight'] * loss)
                    if not self.lr_finder:
                        self.log[-1, i, stage] += loss.detach().cpu().numpy()
                elif l['type'] == 'DIS' and not self.lr_finder:
                    self.log[-1, i, stage] += self.loss[i - 1]['function'].loss
            loss_sum.append(sum(losses))
            if len(self.loss) > 1 and not self.lr_finder:
                self.log[-1, -1, stage] += loss_sum[stage].detach().cpu().numpy()
        if self.lr_finder:
            loss_avg = sum(loss_sum)/len(loss_sum)
            return loss_avg
        else:
            return loss_sum


    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()


    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros([1, len(self.loss), self.stages])))


    def end_log(self, n_batches):
        self.log[-1,:,:].div_(n_batches)


    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            loss = 0
            if l['type'] == 'MSL':
                for stage in range(self.stages):
                    loss += c[stage]
                log.append('[{}: {:.2f}(1 - {:.6f})]'.format('MS-SSIM', l['weight'], 1 - (loss / (n_samples * self.stages))))
            elif l['type'] == 'Total':
                for stage in range(self.stages):
                    loss += c[stage]
                log.append('[{}: {:.6f}]'.format(l['type'], loss / (n_samples * self.stages)))
            else:
                for stage in range(self.stages):
                    loss += c[stage]
                log.append('[{}: {:.2f}({:.6f})]'.format(l['type'], l['weight'], loss / (n_samples * self.stages)))

        return ''.join(log)


    def display_loss_stage(self, batch, stage):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            loss = 0
            if l['type'] == 'MSL':
                loss += c[stage]
                log.append('[{}: {:.2f}(1 - {:.6f})]'.format('MS-SSIM', l['weight'], 1 - (loss / n_samples)))
            elif l['type'] == 'Total':
                loss += c[stage]
                log.append('[{}: {:.6f}]'.format(l['type'], loss / n_samples))
            else:
                loss += c[stage]
                log.append('[{}: {:.2f}({:.6f})]'.format(l['type'], l['weight'], loss / n_samples))

        return ''.join(log)


    def plot_loss(self, apath, epoch):
        if epoch > 1:
            axis = np.linspace(1, epoch, epoch)
            fig = plt.figure(figsize=(38.4,21.6))
            plt.rcParams.update({'font.size': 20})
            plt.title('Loss Functions (Training)')
            MSSSIM = []
            for i, l in enumerate(self.loss):
                for stage in range(self.stages):
                    if l['type'] == 'Total':
                        loss_label = '{} Loss ({})'.format(l['type'], stage+1)
                    else:
                        loss_label = '{}*{} Loss ({})'.format(l['weight'], l['type'], stage+1)
                    plt.plot(axis, self.log[:, i, stage].numpy(), label=loss_label)
                    if l['type'] == 'MSL':
                        MSSSIM.append(1 - self.log[:, i, stage])
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            fig.tight_layout()
            try:
                plt.savefig(apath / 'loss.png', dpi=100)
            except:
                try:
                    plt.savefig(apath / 'loss-{}.png'.format(time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())), dpi=100)
                except:
                    traceback.print_exc()
                    exit()
            plt.close(fig)
            plt.close()
            
            if MSSSIM is not None:
                fig = plt.figure(figsize=(38.4,21.6))
                plt.rcParams.update({'font.size': 20})
                plt.title('MS-SSIM (Training)')
                for stage in range(self.stages):
                    gain_label = 'Stage {}'.format(stage+1)
                    plt.plot(axis, MSSSIM[stage].numpy(), label=gain_label)
                plt.xlabel('Epochs')
                plt.ylabel('MS-SSIM')
                plt.legend()
                plt.grid(True)
                fig.tight_layout()
                try:
                    plt.savefig(apath / 'MS-SSIM.png', dpi=100)
                except:
                    try:
                        plt.savefig(apath / 'MS-SSIM-{}.png'.format(time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())), dpi=100)
                    except:
                        traceback.print_exc()
                        exit()
                plt.close(fig)
                plt.close()


    def plot_iteration_loss(self, loss_array, lr_array, apath, iterations, epoch, stages, running_average):
        round_length = round(running_average / 2)
        if len(lr_array) + round_length < iterations:
            fill_length = iterations - len(lr_array) - round_length
            reflect_length = -(round_length + 1)
        elif len(lr_array) < iterations:
            fill_length = 0
            reflect_length = len(lr_array)-iterations-1

        if len(lr_array) < iterations:
            loss_array      = np.concatenate((loss_array,
                                              loss_array[:reflect_length:-1,:],
                                              np.full((fill_length,
                                                       stages),
                                                      np.average(loss_array[:reflect_length:-1,:],
                                                                 axis=0))))
            lr_array        = np.concatenate((lr_array,
                                              lr_array[:reflect_length:-1],
                                              np.full((fill_length),
                                                      np.average(lr_array[:reflect_length:-1]))))
        axis = np.linspace(1, iterations, iterations)
        # Increase the size of the figure to give more detail to the graph
        # Default is [6.4, 4.8] inches. 38.4"x21.6" @ 100dpi = 3840x2160 = 4K resolution
        fig_kw = {'figsize': [38.4,21.6]}
        # Increase the font-size from 10 to 20
        plt.rcParams.update({'font.size': 20})
        fig, ax1 = plt.subplots(**fig_kw)
        ax2 = ax1.twinx()
        plt.title('Loss Running average per {} iterations'.format(running_average))
        plt.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
        for stage in range(stages):
            ax1.plot(axis,
                     uniform_filter1d(loss_array[:,stage],
                                      mode = 'mirror',
                                      size = running_average),
                     label='Loss Stage {}'.format(stage+1))
        ax2.plot(axis, lr_array, ':', label='Learning Rate')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2)
        plt.xlabel('Iterations')
        ax1.set_ylabel('Loss')
        ax2.set_ylabel('LR')
        ax1.grid(True)
        fig.tight_layout()
        try:
            plt.savefig(apath / 'average_loss_epoch_{:03d}.png'.format(epoch),
                        dpi=100)
        except:
            try:
                plt.savefig(apath / 'average_loss_epoch_{:03d}-{}.png'.format(epoch,
                                                                              time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())),
                            dpi=100)
            except:
                traceback.print_exc()
                exit()
        plt.close(fig)
        plt.close()


    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module


    def save(self, apath):
        torch.save(self.state_dict(), apath / 'loss_state.pt')
        torch.save(self.log, apath / 'loss_log.pt')


    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            apath / 'loss_state.pt',
            **kwargs
        ))
        self.log = torch.load(apath / 'loss_log.pt')
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

