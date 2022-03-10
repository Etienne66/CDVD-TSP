"""This is used to train."""
import decimal
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from utils import utils
from loss import ssim
from trainer.trainer import Trainer
from datetime import datetime, timedelta
import numpy as np

class Trainer_CDVD_TSP(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_CDVD_TSP, self).__init__(
            args, loader, my_model, my_loss, ckp)
        print("Using Trainer-CDVD-TSP")
        assert args.n_sequence in [3,5,7], \
            "Only support args.n_sequence in [3,5,7]; but get args.n_sequence={}".format(
                args.n_sequence)
    # end___init__


    def make_optimizer(self):
        print("Used custom AdamW")
        kwargs = {'lr':           self.args.lr,
                  'weight_decay': self.args.AdamW_weight_decay}
        return optim.AdamW([{"params": self.model.get_model().recons_net.parameters()},
                            {"params": self.model.get_model().flow_net.parameters()}],
        #return optim.AdamW([{"params": self.model.get_model().recons_net.parameters()},
        #                    {"params": self.model.get_model().flow_net.parameters(), "lr": 1e-10}],
        #return optim.AdamW([{"params": self.model.get_model().flow_net.parameters()},
        #                    {"params": self.model.get_model().recons_net.parameters(), "lr": 1e-10}],
                          **kwargs)

    def make_scheduler(self):
        print("Used custom scheduler")
        clr_fn = lambda x: self.args.CyclicLR_gamma**x
        #self.args.step_size_up = int(len(self.loader_train.dataset) * 1.25)
        kwargs = {'base_lr':        [self.args.lr, self.args.FlowNet_lr],
                  'max_lr':         [self.args.max_lr, self.args.FlowNet_max_lr],
        #kwargs = {'base_lr':        [self.args.lr, self.args.lr],
        #          'max_lr':         [self.args.max_lr, self.args.max_lr],
        #kwargs = {'base_lr':        self.args.lr,
        #          'max_lr':         self.args.max_lr,
                  'cycle_momentum': False, #needs to be False for Adam/AdamW
                  'step_size_up':   self.args.step_size_up,
                  'mode':           self.args.CyclicLR_mode,
                  'gamma':          self.args.CyclicLR_gamma
                  #'scale_fn':       clr_fn
                  }
        return lr_scheduler.CyclicLR(self.optimizer, **kwargs)

        
    def timedelta_format(self, deltatime):
        if str(deltatime).find('.') > 0:
            result = str(deltatime).split('.', 2)[0] + '.' + str(deltatime).split('.', 2)[1][:2]
        else:
            result = str(deltatime)
        
        return result

    
    def loss_backward(self, loss, lr, lr_array, loss_array, batch, stages):
        if self.args.plot_running_average:
            if len(loss_array) <= batch:
                lr_array.append(lr)
                if stages == 3:
                    loss_array = np.append(loss_array, [[0.,0.,0.]], axis=0)
                elif stages == 2:
                    loss_array = np.append(loss_array, [[0.,0.]], axis=0)
                elif stages == 1:
                    loss_array = np.append(loss_array, [[0.]], axis=0)
        if self.args.original_loss:
        # The original loss calculation was just an average of all values.
        # Adding weight to the averages to get the same value as the original.
            if self.args.plot_running_average:
                for stage in range(stages):
                    loss_array[batch,stage] = loss[stage].detach().cpu().numpy()
            if stages == 3:
                avg_loss = (loss[0] * 5 + loss[1] * 3 + loss[2]) / 9
            if stages == 2:
                avg_loss = (loss[0] * 3 + loss[1]) / 4
            if stages == 1:
                avg_loss = loss[0]
            avg_loss.backward()
        elif self.args.separate_loss:
        # https://stackoverflow.com/a/47174709
        # After reading the accepted answer it appears that having a separate calculation for each stage is appropriate
        # The loss for each frame in a single stage is about the same. The subsequent stage should have less loss.
        # For a stage 3 model
        #   loss[Stage1] = average loss of 5 frames derived from 7 blurred frames
        #   loss[Stage2] = average loss of 3 frames derived from Stage 1 restored frames
        #   loss[Stage3] = loss of 1 frame derived from Stage 2 restored frames
        # The drawback is that this uses more GPU Memory and takes a lot longer
            for stage in range(stages):
                if stage != (stages - 1):
                    loss[stage].backward(retain_graph = True)
                else:
                # Don't retain_graph on last backward so memory is freed
                    loss[stage].backward()
                
                if self.args.plot_running_average:
                    loss_array[batch,stage] = loss[stage].detach().cpu().numpy()
        else:
        # According to the following the total loss is the same as doing the backward separately
        # It also has the advantage of using less GPU Memory and takes less time than separate_loss
        # https://discuss.pytorch.org/t/what-exactly-does-retain-variables-true-in-loss-backward-do/3508/25
            if self.args.plot_running_average:
                for stage in range(stages):
                    loss_array[batch,stage] = loss[stage].detach().cpu().numpy()
            total_loss = sum(loss)
            total_loss.backward()
        
        if self.args.plot_running_average:
            return lr_array, loss_array
    # end_loss_backward

    
    def train(self):
        """Train the model
        Args:
            self (object): an object with many variables

        Returns:
            nothing
        """
        self.model.train()
        self.loss.start_log()
        self.ckp.start_log(type='LR')
        torch.cuda.empty_cache()
        self.ckp.write_log('\nNow training')
        self.loss.step()
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
        stages = self.args.n_sequence // 2
        number_of_batches = len(self.loader_train)
        start_time = train_start_time
        total_preprocess_time = total_forward_time = total_loss_time = timedelta(days=0)
        total_backward_time = total_optimizer_time = total_scheduler_time = total_time = timedelta(days=0)
        total_postprocess_time = timedelta(days=0)
        if self.args.plot_running_average:
        # Setup arrays to be used for plots
            loss_array = np.zeros((1,stages))
            lr_array = [lr]
        with tqdm(total=len(self.loader_train), position=1, bar_format='{desc}', desc='Waiting for first batch') as tqdm_desc:
            tqdm_train = tqdm(self.loader_train, position=0)
            for batch, (input, gt, filenames) in enumerate(tqdm_train):
                lr = self.scheduler.get_last_lr()[0]
                if self.args.OneCycleLR or self.args.StepLR:
                    self.ckp.report_log(float(lr), type='LR')
                input = input.to(device=self.device)
                gt = gt.to(device=self.device)
                if self.args.n_sequence == 3:
                    gt = torch.cat([gt[:, 1, :, :, :]], dim=1).to(device=self.device)
                elif self.args.n_sequence == 5:
                    gt = torch.cat([gt[:, 1, :, :, :], gt[:, 2, :, :, :], gt[:, 3, :, :, :],
                                    gt[:, 2, :, :, :]], dim=1).to(device=self.device)
                elif self.args.n_sequence == 7:
                    gt = torch.cat([gt[:, 1, :, :, :], gt[:, 2, :, :, :], gt[:, 3, :, :, :], gt[:, 4, :, :, :], gt[:, 5, :, :, :],
                                    gt[:, 2, :, :, :], gt[:, 3, :, :, :], gt[:, 4, :, :, :],
                                    gt[:, 3, :, :, :]], dim=1).to(device=self.device)
                preprocess_time = datetime.now()
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    output_cat = self.model(input)
                    forward_time = datetime.now()
                    loss = self.loss(output_cat, gt)
                    loss_time = datetime.now()
                    if self.args.plot_running_average:
                        lr_array, loss_array = self.loss_backward(loss, lr, lr_array, loss_array, batch, stages)
                    else:
                        self.loss_backward(loss, lr, None, None, batch, stages)
                    backward_time = datetime.now()
                    self.optimizer.step()
                    optimizer_time = datetime.now()
                    if not self.args.StepLR:
                        self.scheduler.step()
                    scheduler_time = datetime.now()
                frames_completed = (batch + 1) * self.args.batch_size
                if frames_completed > frames_total:
                    frames_completed = frames_total
                filename = filenames[self.args.n_sequence // 2][0].split('.')
                if self.args.StepLR:
                    tqdm_desc.set_description('{}[Image: {}][Folder: {}]'.format(self.loss.display_loss(batch),filename[1],filename[0]))
                    tqdm_train.set_postfix_str(s='[{}/{}]]'.format(
                        frames_completed,
                        frames_total))
                else:
                    tqdm_desc.set_description('{}[Image: {}][Folder: {}]'.format(self.loss.display_loss(batch),filename[1],filename[0]))
                    tqdm_train.set_postfix_str(s='[{}/{}][Lr: {:.5e}]'.format(
                        frames_completed,
                        frames_total,
                        decimal.Decimal(lr)))
                if (self.args.plot_running_average
                    and (   (batch + 1) % self.args.running_average == 0
                         or batch + 1 == len(self.loader_train))
                    ):
                    with torch.set_grad_enabled(False):
                        self.loss.plot_iteration_loss(
                            loss_array      = loss_array,
                            lr_array        = lr_array,
                            apath           = self.args.save_dir,
                            iterations      = len(self.loader_train),
                            epoch           = epoch,
                            stages          = stages,
                            running_average = self.args.running_average)
                postprocess_time = datetime.now()
                total_preprocess_time += preprocess_time - start_time
                total_forward_time += forward_time - preprocess_time
                total_loss_time += loss_time - forward_time
                total_backward_time += backward_time - loss_time
                total_optimizer_time += optimizer_time - backward_time
                total_scheduler_time += scheduler_time - optimizer_time
                total_postprocess_time += postprocess_time - scheduler_time
                total_time += postprocess_time - start_time
                start_time = postprocess_time

            tqdm_train.close()
        for stage in range(stages):
            self.ckp.write_log('\t[Stage {}][Loss: {}]'.format(stage + 1,
                                                               self.loss.display_loss_stage(batch, stage)))
        if stages > 1:
            self.ckp.write_log('\t[Average][Loss: {}]'.format(self.loss.display_loss(batch)))
        self.ckp.write_log('\t[Preprocess: {}][Forward: {}][Loss: {}][Backward: {}]'.format(
            self.timedelta_format(total_preprocess_time),
            self.timedelta_format(total_forward_time),
            self.timedelta_format(total_loss_time),
            self.timedelta_format(total_backward_time)))
        iterations_per_second = total_time/len(self.loader_train)
        self.ckp.write_log('\t[Optimizer: {}][Scheduler: {}][Post: {}]\t\t[Total: {}][{:.6f}s/it]'.format(
            self.timedelta_format(total_optimizer_time),
            self.timedelta_format(total_scheduler_time),
            self.timedelta_format(total_postprocess_time),
            self.timedelta_format(total_time),
            iterations_per_second.total_seconds()))

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
    # end_def_train


    def test(self):
        r"""Test the model
        Args:
            self (object): an object with many variables

        Returns:
            nothing
        """
        self.model.eval()
        epoch = self.args.epochs_completed + 1
        self.ckp.write_log('Now testing')
        self.ckp.write_log('\tTest Epoch {:3d}'.format(epoch))
        test_start_time = datetime.now()
        start_time = test_start_time
        total_preprocess_time = timedelta(days=0)
        total_forward_time = timedelta(days=0)
        total_postprocess_time = timedelta(days=0)
        total_time = timedelta(days=0)
        print('Evaluation:')
        self.ckp.start_log(type='PSNR')
        self.ckp.start_log(type='SSIM')
        frames_total = len(self.loader_test.dataset)
        stages = self.args.n_sequence // 2
    
        # This is to pick the same middle frame from each stage
        if stages == 3:
            image_list = [2, 6, 8]
            output_frames = 9
        elif stages == 2:
            image_list = [1, 3]
            output_frames = 4
        elif stages == 1:
            image_list = [0]
            output_frames = 1
    
        # No need for gradients during the test phase
        with torch.no_grad():
            torch.cuda.empty_cache()
            total_PSNR = 0.
            total_SSIM = 0.
            total_num = 0.
            with tqdm(total=len(self.loader_test), position=1, bar_format='{desc}', desc='Waiting for first batch') as tqdm_desc:
                tqdm_test = tqdm(self.loader_test, position=0)
                for idx_img, (input, gt, filenames) in enumerate(tqdm_test):
                    #if idx_img % 10 == 0:
                    #    torch.cuda.empty_cache()
                    SSIM = []
                    PSNR = []
                    input_gpu = input.to(device=self.device)
                    gt = gt[:, self.args.n_sequence // 2, :, :, :].to(device=self.device)
                    preprocess_time = datetime.now()
                    output_cat = self.model(input_gpu)
                    forward_time = datetime.now()
                    output_images = torch.chunk(output_cat, output_frames, dim=1)
                    for stage in range(stages):
                    # Compare each stage image to the one ground truth
                        PSNR.append(utils.calc_psnr(gt,
                                                    output_images[image_list[stage]],
                                                    rgb_range=self.args.rgb_range).detach().to('cpu').numpy())
                        SSIM.append(ssim.ssim(gt,
                                              output_images[image_list[stage]],
                                              data_range=self.args.rgb_range).detach().to('cpu').numpy())
                    total_PSNR += sum(PSNR)/len(PSNR)
                    self.ckp.report_log(PSNR, type='PSNR')
                    total_SSIM += sum(SSIM)/len(SSIM)
                    self.ckp.report_log(SSIM, type='SSIM')
                    total_num += 1
                    frames_completed = (idx_img + 1) * self.args.batch_size_test
                    
                    # This is only used on the last batch
                    if frames_completed > frames_total:
                        frames_completed = frames_total
                    filename = filenames[self.args.n_sequence // 2][0].split('.')
                    tqdm_desc.set_description('[PSNR: {:.3f}][SSIM: {:.6f}][Image: {}][Folder: {}]'.format(
                                               total_PSNR/total_num,
                                               total_SSIM/total_num,
                                               filename[1],filename[0]))
                    tqdm_test.set_postfix_str(s='[{}/{}]'.format(frames_completed,
                                                                                             frames_total))

                    if self.args.save_images:
                        for idx in range(0, self.args.batch_size_test):
                            filename = filenames[self.args.n_sequence // 2][idx]
                            input_center = input_gpu[idx, self.args.n_sequence // 2, :, :, :]
                            gtout = gt[idx, :, :, :]
                            output_image = output_images[image_list[stage]][idx, :, :, :]
                            gtout, input_center, output_image = \
                                utils.postprocess(gtout,
                                                  input_center,
                                                  output_image,
                                                  rgb_range  = self.args.rgb_range,
                                                  ycbcr_flag = False,
                                                  device     = self.device)
                            save_list = [gtout, input_center, output_image]
                            self.ckp.save_images(filename, save_list, epoch)
                        # end for
                    # end if
                    postprocess_time = datetime.now()
                    total_preprocess_time += preprocess_time - start_time
                    total_forward_time += forward_time - preprocess_time
                    total_postprocess_time += postprocess_time - forward_time
                    total_time += postprocess_time - start_time
                    start_time = postprocess_time
                # end for


                tqdm_test.close()

            # If using StepLR as the scheduler it needs to take a step after testing.
            if self.args.StepLR:
                self.scheduler.step()

            self.ckp.end_log(len(self.loader_test), type='PSNR')
            self.ckp.end_log(len(self.loader_test), type='SSIM')
            for stage in range(stages):
                best = self.ckp.psnr_log[:,stage].max(0)
                best_ssim = self.ckp.ssim_log[:,stage].max(0)
                self.ckp.write_log('\t[{}][Stage {}]\taverage PSNR: {:.3f} (Best: {:.3f} @epoch {}) SSIM: {:.6f} (Best: {:.6f} @epoch {})'.format(
                    self.args.data_test,
                    stage + 1,
                    self.ckp.psnr_log[-1,stage],
                    best[0],
                    best[1] + 1,
                    self.ckp.ssim_log[-1,stage],
                    best_ssim[0],
                    best_ssim[1] + 1))
            # end for
            self.args.epochs_completed += 1
            self.args.total_test_time += datetime.now() - test_start_time
            self.ckp.write_log('\t[Preprocess: {}][Forward: {}][Postprocess: {}]'.format(
                self.timedelta_format(total_preprocess_time),
                self.timedelta_format(total_forward_time),
                self.timedelta_format(total_postprocess_time)))
            iterations_per_second = total_time/len(self.loader_test)
            self.ckp.write_log('\t\t[Total: {}][{:.6f}s/it]'.format(
                self.timedelta_format(total_time),
                iterations_per_second.total_seconds()))

            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))
    # end_def_test
