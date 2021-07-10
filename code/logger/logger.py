import torch
import imageio
from torchvision.utils import save_image
import numpy as np
import os
import datetime
import skimage.color as sc

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


class Logger:
    def __init__(self, args):
        self.args = args
        self.psnr_log = torch.Tensor()
        self.ssim_log = torch.Tensor()
        self.lr_log = torch.Tensor()
        self.epoch = 0

        if args.load == '.':
            if args.save == '.':
                args.save = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S", now_time)
            self.dir = args.experiment_dir + args.save
        else:
            self.dir = args.experiment_dir + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.ssim_log = torch.load(self.dir + '/ssim_log.pt')
                self.psnr_log = torch.load(self.dir + '/psnr_log.pt')
                self.lr_log = torch.load(self.dir + '/lr_log.pt')
                self.epoch = len(self.psnr_log)
                print('Continue from epoch {}...'.format(self.epoch))

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            if not os.path.exists(self.dir + '/model'):
                os.makedirs(self.dir + '/model')
        if not os.path.exists(self.dir + '/result/' + self.args.data_test):
            print("Creating dir for saving images...", self.dir + '/result/' + self.args.data_test)
            os.makedirs(self.dir + '/result/' + self.args.data_test)

        print('Save Path : {}'.format(self.dir))

    # Do not log anything if we are just trying to find the learning rate to use
        if not args.lr_finder:
            now_time_file = args.start_time.strftime("%Y-%m-%dT%H%M%S")
            #open_type = 'a' if os.path.exists(self.dir + '/log_{}.txt'.format(now_time_file)) else 'w'
            if self.epoch == 0:
                open_type = 'w'
            else:
                open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
            self.log_file = open(self.dir + '/log.txt', open_type, buffering=1)
            self.config_file = open(self.dir + '/config.txt', open_type, buffering=1)
            #with open(self.dir + '/config_{}.txt'.format(now_time_file), open_type, buffering=1) as f:
            #    f.write('From epoch {}...'.format(len(self.psnr_log)) + '\n\n')
            #    for arg in sorted(vars(args)):
            #        f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            #    f.write('\n')
            self.config_file.write('From epoch {}...'.format(self.epoch) + '\n\n')
            if self.epoch == 0:
                for arg in sorted(vars(args)):
                    self.config_file.write('{}: {}\n'.format(arg, getattr(args, arg)))
            self.config_file.write('\n')

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')
        self.log_file.flush() # added to keep the log file up to date

    def save(self, trainer, epoch, is_best):
        trainer.model.save(self.dir, epoch, is_best)
        torch.save(self.ssim_log, os.path.join(self.dir, 'ssim_log.pt'))
        torch.save(self.psnr_log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(self.lr_log, os.path.join(self.dir, 'lr_log.pt'))
        torch.save(trainer.optimizer.state_dict(), os.path.join(self.dir, 'optimizer.pt'))
        torch.save(trainer.scheduler.state_dict(), os.path.join(self.dir, 'scheduler.pt'))
        rng_state = torch.get_rng_state()
        torch.save({'epoch': epoch,
                    'total_train_time': self.args.total_train_time,
                    'total_test_time': self.args.total_test_time,
                    'epochs_completed': self.args.epochs_completed,
                    'rng_state': rng_state
                   },
                   os.path.join(self.dir, 'checkpoint.tar'))
        trainer.loss.save(self.dir)
        if epoch > 1:
            trainer.loss.plot_loss(self.dir, epoch)
            self.plot_psnr_log(epoch)
            self.plot_ssim_log(epoch)
            self.plot_lr_log(epoch)

    def save_images(self, filename, save_list, epoch):
        if self.args.task == 'VideoDeblur':
            f = filename.split('.')
            dirname = '{}/result/{}/{}'.format(self.dir, self.args.data_test, f[0])
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            filename = '{}/{}'.format(dirname, f[1])
            #postfix = ['gt', 'blur', 'deblur', 'deblur1']
            postfix = ['gt', 'blur', 'deblur']
        else:
            raise NotImplementedError('Task [{:s}] is not found'.format(self.args.task))
        for img, post in zip(save_list, postfix):
            #img = img[0].data
            img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
            if img.shape[2] == 1:
                img = img.squeeze(axis=2)
            elif img.shape[2] == 3 and self.args.n_colors == 1:
                img = sc.ycbcr2rgb(img.astype('float')).clip(0, 1)
                img = (255 * img).round().astype('uint8')
            imageio.imwrite('{}_{}.png'.format(filename, post), img)

    def save_images_tensor(self, filename, save_list, epoch):
        if self.args.task == 'VideoDeblur':
            f = filename.split('.')
            dirname = '{}/result/{}/{}'.format(self.dir, self.args.data_test, f[0])
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            filename = '{}/{}'.format(dirname, f[1])
            #postfix = ['gt', 'blur', 'deblur', 'deblur1']
            postfix = ['gt', 'blur', 'deblur']
        else:
            raise NotImplementedError('Task [{:s}] is not found'.format(self.args.task))
        for img, post in zip(save_list, postfix):
            save_image('{}_{}.png'.format(filename, post), img)

    def start_log(self, type='PSNR'):
        if type == 'SSIM':
            self.ssim_log = torch.cat((self.ssim_log, torch.zeros(1)))
        elif type == 'PSNR':
            self.psnr_log = torch.cat((self.psnr_log, torch.zeros(1)))
        elif type == 'LR':
            self.lr_log = torch.cat((self.lr_log, torch.zeros(1)))

    def report_log(self, item, type='PSNR'):
        if type == 'SSIM':
            self.ssim_log[-1] += item
        elif type == 'PSNR':
            self.psnr_log[-1] += item
        elif type == 'LR':
            self.lr_log[-1] += item

    def end_log(self, n_div, type='PSNR'):
        if type == 'SSIM':
            self.ssim_log[-1].div_(n_div)
        elif type == 'PSNR':
            self.psnr_log[-1].div_(n_div)
        elif type == 'LR':
            self.lr_log[-1].div_(n_div)

    def plot_ssim_log(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title('SSIM Graph')
        plt.plot(axis, self.ssim_log.numpy())
        plt.xlabel('Epochs')
        plt.ylabel('SSIM')
        plt.grid(True)
        plt.savefig(os.path.join(self.dir, 'ssim.png'), dpi=300)
        plt.close(fig)

    def plot_psnr_log(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title('PSNR Graph')
        plt.plot(axis, self.psnr_log.numpy())
        #plt.legend() # Does not need a legend
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig(os.path.join(self.dir, 'psnr.png'), dpi=300)
        plt.close(fig)

    def plot_lr_log(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title('LR Graph')
        plt.plot(axis, self.lr_log.numpy())
        plt.xlabel('Epochs')
        plt.ylabel('LR')
        plt.grid(True)
        plt.savefig(os.path.join(self.dir, 'lr.png'), dpi=300)
        plt.close(fig)

    def done(self):
        self.log_file.close()
