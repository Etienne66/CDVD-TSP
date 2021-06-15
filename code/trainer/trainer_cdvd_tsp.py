"""This is used to train."""
import decimal
import torch
import torch.optim as optim
from tqdm import tqdm
from utils import utils
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
            "Only support args.n_sequence=5; but get args.n_sequence={}".format(
                args.n_sequence)

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        return optim.Adam([{"params": self.model.get_model().recons_net.parameters()},
                           {"params": self.model.get_model().flow_net.parameters(), "lr": 1e-6}],
                          **kwargs)

    def train(self):
        torch.cuda.empty_cache()
        print("Now training")
        # Starting with PyTorch 1.1.0 and later optimizer.step() must be called before scheduler.step()
        self.optimizer.step()
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch
        job_epochs_completed = self.args.epochs_completed
        job_epochs_started = job_epochs_completed + 1
        job_total_epochs = self.args.epochs - (epoch - job_epochs_started)
        train_start_time = datetime.now()

        # get_lr was deprecated starting with PyTorch 1.4.0
        try:
            lr = self.scheduler.get_last_lr()[0]
        except:        
            lr = self.scheduler.get_lr()[0]

        self.ckp.write_log('\tEpoch {:4d} with Lr {:.2e}'.format(
            epoch,
            decimal.Decimal(lr)))

        if job_epochs_completed > 0:
            job_total_time = self.args.total_train_time + self.args.total_test_time
            epoch_end_time = self.args.start_time + (job_epochs_started * (job_total_time/job_epochs_completed))
            job_end_time = self.args.start_time + (job_total_epochs * (job_total_time/job_epochs_completed))
            self.ckp.write_log('\t[Now: {}][CE: {}][AE: {}]'.format(
                train_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                epoch_end_time.strftime("%Y-%m-%d %H:%M:%S"),
                job_end_time.strftime("%Y-%m-%d %H:%M:%S")))
        self.loss.start_log()
        self.model.train()
        self.ckp.start_log()
        #self.ckp.write_log('\t\t[L1:     Mean Absolute Error(L1 Loss)         ]')
        #self.ckp.write_log('\t\t[HEM:    Hard Example Mining                  ]')
        #self.ckp.write_log('\t\t[Total:  L1+HEM                               ]')
        #self.ckp.write_log('\t\t[now:    Current Time                         ]')
        #self.ckp.write_log('\t\t[CeFin:  Estimated Completion of Current epoch]')
        #self.ckp.write_log('\t\t[AeFin:  Estimated Completion of all epochs   ]')

        frames_total = len(self.loader_train.dataset)
        tqdm_train = tqdm(self.loader_train, ncols=140)
        #for batch, (input, gt, _) in enumerate(self.loader_train):
        for batch, (input, gt, _) in enumerate(tqdm_train):
            #torch.cuda.empty_cache()
            self.optimizer.zero_grad()
            input_gpu = input.to(self.device)
            gt_list = [gt[:, i, :, :, :] for i in range(self.args.n_sequence)]

            if self.args.n_sequence == 3:
                gt_cat = torch.cat([gt_list[1]], dim=1).to(self.device)
            elif self.args.n_sequence == 5:
                gt_cat = torch.cat([gt_list[1], gt_list[2], gt_list[3], gt_list[2]], dim=1).to(self.device)
            elif self.args.n_sequence == 7:
                gt_cat = torch.cat([gt_list[1], gt_list[2], gt_list[3], gt_list[2], gt_list[4],
                                gt_list[5], gt_list[3], gt_list[4], gt_list[3]], dim=1).to(self.device)
            del gt_list

            output_cat, _ = self.model(input_gpu)
            del input_gpu
            loss = self.loss(output_cat, gt_cat)
            del gt_cat
            del output_cat
            loss.backward()
            self.optimizer.step()
            self.ckp.report_log(float(loss.item()))
            frames_completed = (batch + 1) * self.args.batch_size
            if frames_completed > frames_total:
                frames_completed = frames_total
            tqdm_train.set_postfix_str(s='[{}/{}]{}'.format(frames_completed,
                                                            frames_total,
                                                            self.loss.display_loss(batch)))

            #if (batch + 1) % self.args.print_every == 0:
            #    current_time = datetime.now()
            #    elapsed_time = current_time - train_start_time
            #    frames_completed = (batch + 1) * self.args.batch_size
            #    train_end_time = train_start_time + (frames_total * (elapsed_time / frames_completed))
            #    epoch_end_time = train_end_time
            #
            #    if job_epochs_completed > 0:
            #        job_total_time = self.args.total_train_time + self.args.total_test_time
            #        epoch_end_time += self.args.total_test_time/job_epochs_completed
            #        job_end_time = self.args.start_time + (job_total_epochs * (job_total_time/job_epochs_completed))
            #        job_end_time += epoch_end_time - train_start_time
            #        
            #    else:
            #        job_end_time = self.args.start_time + (train_end_time - train_start_time)*job_total_epochs
            #
            #    self.ckp.write_log('[{}/{}]\tLoss : {}[now: {}][CeFin: {}][AeFin: {}]'.format(
            #        frames_completed,
            #        frames_total,
            #        self.loss.display_loss(batch),
            #        current_time.strftime("%H:%M:%S"),
            #        epoch_end_time.strftime("%Y-%m-%d %H:%M:%S"),
            #        job_end_time.strftime("%Y-%m-%d %H:%M:%S")
            #    ))
            del loss
        tqdm_train.close()
        del tqdm_train
        
        self.ckp.write_log('\tLoss: {}'.format(
            self.loss.display_loss(batch)))
        self.loss.end_log(len(self.loader_train))

        current_time = datetime.now()
        elapsed_time = current_time - train_start_time
        train_end_time = train_start_time + elapsed_time
        epoch_end_time = train_end_time

        if job_epochs_completed > 0:
            job_total_time = self.args.total_train_time + self.args.total_test_time
            epoch_end_time += self.args.total_test_time/job_epochs_completed
            job_end_time = self.args.start_time + (job_total_epochs * (job_total_time/job_epochs_completed))
            #job_end_time += epoch_end_time - train_start_time
        else:
            job_end_time = self.args.start_time + (train_end_time - train_start_time)*job_total_epochs
        
        self.ckp.write_log('\t[Now: {}][CE: {}][AE: {}]'.format(
            current_time.strftime("%Y-%m-%d %H:%M:%S"),
            epoch_end_time.strftime("%Y-%m-%d %H:%M:%S"),
            job_end_time.strftime("%Y-%m-%d %H:%M:%S")
        ))
        self.args.epochs_completed += 1
        self.args.total_train_time += (datetime.now() - train_start_time)

    def test(self):
        torch.cuda.empty_cache()
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('Now testing')
        self.ckp.write_log('\tTest Epoch {:3d}'.format(epoch))
        test_start_time = datetime.now()
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)
        frames_total = len(self.loader_test.dataset)
        with torch.no_grad():
            total_PSNR = 0.
            total_num = 0.
            tqdm_test = tqdm(self.loader_test, ncols=90)
            for idx_img, (input, gt, filenames) in enumerate(tqdm_test):
                #torch.cuda.empty_cache()
                input_gpu = input.to(self.device)
                _, output_images = self.model(input_gpu)
                gt_img = gt[:, self.args.n_sequence // 2, :, :, :].to(self.device)
                PSNR = utils.calc_psnr(
                    gt_img, output_images, rgb_range=self.args.rgb_range)
                total_PSNR += float(PSNR)
                self.ckp.report_log(float(PSNR), train=False)
                del PSNR
                total_num += 1
                frames_completed = (idx_img + 1) * self.args.batch_size
                if frames_completed > frames_total:
                    frames_completed = frames_total
                tqdm_test.set_postfix_str(s='[{}/{}][PSNR: {:.3f}]'.format(frames_completed,
                                                                           frames_total,
                                                                           total_PSNR/total_num))

                if self.args.save_images:
                    for idx in range(0, self.args.batch_size):
                        filename = filenames[self.args.n_sequence // 2][idx]
                        input_center = input_gpu[idx, self.args.n_sequence // 2, :, :, :]
                        gtout = gt_img[idx, :, :, :]
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
                del gt_img
                del output_images
                del input_gpu
            tqdm_test.close()
            del tqdm_test

            self.ckp.end_log(len(self.loader_test), train=False)
            best = self.ckp.psnr_log.max(0)
            self.ckp.write_log('\t[{}]\taverage PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                self.args.data_test,
                self.ckp.psnr_log[-1],
                best[0],
                best[1] + 1))
            self.args.total_test_time += datetime.now() - test_start_time

            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))
