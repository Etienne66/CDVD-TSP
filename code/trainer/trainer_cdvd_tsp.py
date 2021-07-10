"""This is used to train."""
import decimal
import torch
import torch.optim as optim
from tqdm import tqdm
from utils import utils
from loss import ssim
from trainer.trainer import Trainer
from datetime import datetime
from dateutil.relativedelta import relativedelta

def time_diff(t_a, t_b):
    t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
    return '{:02d}:{:02d}:{:02d}'.format(t_diff.hours, t_diff.minutes, t_diff.seconds)

class Trainer_CDVD_TSP(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_CDVD_TSP, self).__init__(
            args, loader, my_model, my_loss, ckp)
        print("Using Trainer-CDVD-TSP")
        assert args.n_sequence in [3,5,7], \
            "Only support args.n_sequence in [3,5,7]; but get args.n_sequence={}".format(
                args.n_sequence)

    #def make_optimizer(self):
    #    kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
    #    return optim.Adam([{"params": self.model.get_model().recons_net.parameters()},
    #                       {"params": self.model.get_model().flow_net.parameters(), "lr": 1e-6}],
    #                      **kwargs)

    def train(self):
        self.model.train()
        self.loss.start_log()
        self.ckp.start_log(type='LR')
        torch.cuda.empty_cache()
        self.ckp.write_log('\nNow training')
        self.loss.step()
        #print(self.scheduler.last_epoch)
        epoch = self.args.epochs_completed + 1
        job_epochs_completed = self.args.epochs_completed
        job_epochs_started = job_epochs_completed + 1
        job_total_epochs = self.args.epochs - (epoch - job_epochs_started)
        train_start_time = datetime.now()
        lr = self.scheduler.get_last_lr()[0]
        self.ckp.write_log('Epoch {:4d} with Lr {:.5e} start: {}]'.format(
            epoch,
            decimal.Decimal(lr),
            train_start_time.strftime("%Y-%m-%d %H:%M:%S")))

        if job_epochs_completed > 0:
            job_total_time = self.args.total_train_time + self.args.total_test_time
            epoch_end_time = self.args.start_time + (job_epochs_started * (job_total_time/job_epochs_completed))
            job_end_time = self.args.start_time + (job_total_epochs * (job_total_time/job_epochs_completed))
            self.ckp.write_log('\t[CE: {}][AE: {}]'.format(
                epoch_end_time.strftime("%Y-%m-%d %H:%M:%S"),
                job_end_time.strftime("%Y-%m-%d %H:%M:%S")))
        
        frames_total = len(self.loader_train.dataset)
        if self.args.StepLR:
            tqdm_train = tqdm(self.loader_train, ncols=148)
        else:
            tqdm_train = tqdm(self.loader_train, ncols=160)
        for batch, (input, gt, _) in enumerate(tqdm_train):
            self.ckp.report_log(float(lr), type='LR')
            input_gpu = input.to(self.device)
            gt = gt[:, self.args.n_sequence // 2, :, :, :].to(self.device)
            self.optimizer.zero_grad()
            #self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(True):
                output_image = self.model(input_gpu)
                loss = self.loss(output_image, gt)
                loss.backward()
                self.optimizer.step()
                if not self.args.StepLR:
                    self.scheduler.step()
            frames_completed = (batch + 1) * self.args.batch_size
            if frames_completed > frames_total:
                frames_completed = frames_total
            if self.args.StepLR:
                tqdm_train.set_postfix_str(s='[{}/{}]{}]'.format(
                    frames_completed,
                    frames_total,
                    self.loss.display_loss(batch)))
            else:
                lr = self.scheduler.get_last_lr()[0]
                tqdm_train.set_postfix_str(s='[{}/{}]{}[Lr: {:.5e}]'.format(
                    frames_completed,
                    frames_total,
                    self.loss.display_loss(batch),
                    decimal.Decimal(lr)))
        tqdm_train.close()
        
        self.ckp.write_log('\tLoss: {}'.format(self.loss.display_loss(batch)))
        self.loss.end_log(len(self.loader_train))
        self.ckp.end_log(len(self.loader_train), type='LR')

        current_time = datetime.now()
        elapsed_time = current_time - train_start_time
        train_end_time = train_start_time + elapsed_time
        epoch_end_time = train_end_time

        if job_epochs_completed > 0:
            job_total_time = self.args.total_train_time + self.args.total_test_time
            epoch_end_time += self.args.total_test_time/job_epochs_completed
            job_end_time = self.args.start_time + (job_total_epochs * (job_total_time/job_epochs_completed))
        else:
            job_end_time = self.args.start_time + (train_end_time - train_start_time)*job_total_epochs
        
        self.ckp.write_log('\t[Now: {}][CE: {}][AE: {}]'.format(
            current_time.strftime("%Y-%m-%d %H:%M:%S"),
            epoch_end_time.strftime("%Y-%m-%d %H:%M:%S"),
            job_end_time.strftime("%Y-%m-%d %H:%M:%S")
        ))
        self.args.total_train_time += (datetime.now() - train_start_time)

    def test(self):
        self.model.eval()
        epoch = self.args.epochs_completed + 1
        self.ckp.write_log('Now testing')
        self.ckp.write_log('\tTest Epoch {:3d}'.format(epoch))
        test_start_time = datetime.now()
        print('Evaluation:')
        self.ckp.start_log(type='PSNR')
        self.ckp.start_log(type='SSIM')
        frames_total = len(self.loader_test.dataset)
        with torch.no_grad():
            torch.cuda.empty_cache()
            total_PSNR = 0.
            total_SSIM = 0.
            total_num = 0.
            tqdm_test = tqdm(self.loader_test, ncols=120)
            for idx_img, (input, gt, filenames) in enumerate(tqdm_test):
                input_gpu = input.to(self.device)
                output_images = self.model(input_gpu)
                gt = gt[:, self.args.n_sequence // 2, :, :, :].to(self.device)
                PSNR = utils.calc_psnr(
                    gt, output_images, rgb_range=self.args.rgb_range)
                total_PSNR += float(PSNR)
                self.ckp.report_log(float(PSNR), type='PSNR')
                SSIM = ssim.ssim(gt, output_images, data_range=self.args.rgb_range)
                total_SSIM += SSIM
                self.ckp.report_log(float(SSIM), type='SSIM')
                total_num += 1
                frames_completed = (idx_img + 1) * self.args.batch_size_test
                if frames_completed > frames_total:
                    frames_completed = frames_total
                tqdm_test.set_postfix_str(s='[{}/{}][PSNR: {:.3f}][SSIM: {:.6f}]'.format(frames_completed,
                                                                           frames_total,
                                                                           total_PSNR/total_num,
                                                                           total_SSIM/total_num))

                if self.args.save_images:
                    for idx in range(0, self.args.batch_size_test):
                        filename = filenames[self.args.n_sequence // 2][idx]
                        input_center = input_gpu[idx, self.args.n_sequence // 2, :, :, :]
                        gtout = gt[idx, :, :, :]
                        output_image = output_images[idx, :, :, :]
                        gtout, input_center, output_image = \
                            utils.postprocess(gtout,
                                              input_center,
                                              output_image,
                                              rgb_range  = self.args.rgb_range,
                                              ycbcr_flag = False,
                                              device     = self.device)
                        save_list = [gtout, input_center, output_image]
                        self.ckp.save_images(filename, save_list, epoch)
            if self.args.StepLR:
                self.scheduler.step()
            tqdm_test.close()

            self.ckp.end_log(len(self.loader_test), type='PSNR')
            self.ckp.end_log(len(self.loader_test), type='SSIM')
            best = self.ckp.psnr_log.max(0)
            best_ssim = self.ckp.ssim_log.max(0)
            self.ckp.write_log('\t[{}]\taverage PSNR: {:.3f} (Best: {:.3f} @epoch {}) SSIM: {:.6f} (Best: {:.6f} @epoch {})'.format(
                self.args.data_test,
                self.ckp.psnr_log[-1],
                best[0],
                best[1] + 1,
                self.ckp.ssim_log[-1],
                best_ssim[0],
                best_ssim[1] + 1))
            self.args.total_test_time += datetime.now() - test_start_time
            self.args.epochs_completed += 1

            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))
