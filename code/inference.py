import os
import torch
import glob
import numpy as np
import imageio
import cv2 #OpenCV
import math
import time
import argparse
import platform
from model.cdvd_tsp import CDVD_TSP
from pathlib import Path
from loss.ssim import ssim as ssim_2
from utils import utils

class Traverse_Logger:
    def __init__(self, result_dir, filename='inference_log.txt'):
        self.log_file_path = result_dir / filename
        open_type = 'a' if self.log_file_path.exists else 'w'
        self.log_file = self.log_file_path.open(mode=open_type, buffering=1)

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')


class Inference:
    def __init__(self, args):

        self.save_image = args.save_image
        self.border = args.border
        self.model_path = args.model_path
        self.data_path = args.data_path
        self.result_path = args.result_path
        self.n_seq = args.n_seq
        self.size_must_mode = 4
        self.device = 'cuda'

        if not self.result_path.is_dir():
            self.result_path.mkdir()
            print('mkdir: {}'.format(self.result_path))

        self.input_path = self.data_path / "blur"
        self.GT_path = self.data_path / "gt"

        now_time = time.gmtime()
        now_time_file = time.strftime("%Y-%m-%d", now_time) + "T" + time.strftime("%H%M%S", now_time)
        now_time_log = time.strftime("%Y-%m-%d %H:%M:%S", now_time)
        self.logger = Traverse_Logger(self.result_path, 'inference_log_{}.txt'.format(now_time_file))

        self.logger.write_log('Inference - {}'.format(now_time_log))
        self.logger.write_log('save_image: {}'.format(self.save_image))
        self.logger.write_log('border: {}'.format(self.border))
        self.logger.write_log('model_path: {}'.format(self.model_path))
        self.logger.write_log('data_path: {}'.format(self.data_path))
        self.logger.write_log('result_path: {}'.format(self.result_path))
        self.logger.write_log('n_seq: {}'.format(self.n_seq))
        self.logger.write_log('size_must_mode: {}'.format(self.size_must_mode))
        self.logger.write_log('device: {}'.format(self.device))

        self.net = CDVD_TSP(
            in_channels    = 3,
            n_sequence     = self.n_seq,
            out_channels   = 3,
            n_resblock     = 3,
            n_feat         = 32,
            is_mask_filter = True,
            device         = self.device)
        self.net.load_state_dict(torch.load(self.model_path), strict=False)
        self.net = self.net.to(self.device)
        self.logger.write_log('Loading model from {}'.format(self.model_path))
        self.net.eval()

    def infer(self):
        torch.backends.cudnn.benchmark = True
        stages = self.n_seq // 2
        
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
        with torch.no_grad():
            total_psnr = {}
            total_ssim = {}
            #total_ssim2 = {}
            #videos = sorted(os.listdir(self.input_path))
            videos = sorted(self.input_path.iterdir())
            for v in videos:
                video_psnr = []
                video_ssim = []
                #video_ssim2 = []
                
                #input_frames = sorted(glob.glob(self.input_path / v / "*"))
                #gt_frames = sorted(glob.glob(self.GT_path / v / "*"))
                input_frames = sorted((self.input_path / v.name).iterdir())
                gt_frames = sorted((self.GT_path / v.name).iterdir())

                if len(gt_frames) != 0:
                    input_seqs = self.gene_seq(input_frames, n_seq=self.n_seq)
                    gt_seqs = self.gene_seq(gt_frames, n_seq=self.n_seq)

                    for in_seq, gt_seq in zip(input_seqs, gt_seqs):
                        start_time = time.time()
                        #filename = os.path.basename(in_seq[self.n_seq // 2]).split('.')[0]
                        filename = in_seq[self.n_seq // 2].stem
                        #print(in_seq)
                        inputs = [imageio.imread(str(p)) for p in in_seq]

                        h, w, c = inputs[self.n_seq // 2].shape
                        new_h, new_w = h - h % self.size_must_mode, w - w % self.size_must_mode
                        inputs = [im[:new_h, :new_w, :] for im in inputs]

                        in_tensor = self.numpy2tensor(inputs).to(self.device)
                        preprocess_time = time.time()
                        output = self.net(in_tensor)
                        forward_time = time.time()
                        output_images = torch.chunk(output, output_frames, dim=1)
                        gt = imageio.imread(gt_seq[self.n_seq // 2])
                        gt = gt[:new_h, :new_w, :]
                        #gt_Tensor = torch.unsqueeze(utils.np2Tensor(gt, rgb_range=1.)[0].to(self.device), 0)

                        for stage in range(stages):
                            output_img = self.tensor2numpy(output_images[image_list[stage]])
                            psnr, ssim = self.get_PSNR_SSIM(output_img, gt)
                            video_psnr.append(psnr)
                            video_ssim.append(ssim)
                            total_psnr[v.name] = video_psnr
                            total_ssim[v.name] = video_ssim
                            self.logger.write_log('> {}-{} Stage:{} PSNR={:.5}, SSIM={:.4}'.format(v.name, filename, stage, psnr, ssim))

                        if self.save_image:
                            if not (self.result_path / v.name).is_dir():
                                (self.result_path / v.name).mkdir()
                                print("mkdir: ", self.result_path / v.name)
                            imageio.imwrite(self.result_path / v.name / '{}.png'.format(filename), output_img)
                        postprocess_time = time.time()

                        self.logger.write_log(
                            '> {}-{} pre_time:{:.3}s, forward_time:{:.3}s, post_time:{:.3}s, total_time:{:.3}s'
                                .format(v.name, filename,
                                        preprocess_time - start_time,
                                        forward_time - preprocess_time,
                                        postprocess_time - forward_time,
                                        postprocess_time - start_time))
                else:      
                    len_frames = len(input_frames)
                # Wrap frames around the ends so that the output has the same number of frames as the input
                # The process throws away n_seq // 2 from the start and finish of the video
                    if self.n_seq == 3:
                        temp_frames = [input_frames[2]] + input_frames\
                                      + [input_frames[len_frames-1]]
                    elif self.n_seq == 5:
                        temp_frames = [input_frames[3], input_frames[2]] + input_frames\
                                      + [input_frames[len_frames-1], input_frames[len_frames-2]]
                    elif self.n_seq == 7:
                        temp_frames = [input_frames[4], input_frames[3], input_frames[2]] + input_frames\
                                      + [input_frames[len_frames-1], input_frames[len_frames-2], input_frames[len_frames-3]]
                    input_frames = temp_frames
                    input_seqs = self.gene_seq(input_frames, n_seq=self.n_seq)
                    del temp_frames
                    del len_frames

                    for in_seq in input_seqs:
                        start_time = time.time()
                        filename = in_seq[self.n_seq // 2].stem
                        inputs = [imageio.imread(p) for p in in_seq]

                        h, w, c = inputs[self.n_seq // 2].shape
                        new_h, new_w = h - h % self.size_must_mode, w - w % self.size_must_mode
                        inputs = [im[:new_h, :new_w, :] for im in inputs]

                        in_tensor = self.numpy2tensor(inputs).to(self.device)
                        preprocess_time = time.time()
                        output = self.net(in_tensor)
                        forward_time = time.time()
                        output_images = torch.chunk(output, output_frames, dim=1)

                        output_img = self.tensor2numpy(output_images[image_list[-1]])
                        if not (self.result_path / v.name).is_dir():
                            (self.result_path / v.name).mkdir()
                        imageio.imwrite(self.result_path / v.name / '{}.png'.format(filename), output_img)
                        postprocess_time = time.time()

                        self.logger.write_log(
                            '> {}-{} pre_time:{:.3}s, forward_time:{:.3}s, post_time:{:.3}s, total_time:{:.3}s'
                                .format(v.name, filename,
                                        preprocess_time - start_time,
                                        forward_time - preprocess_time,
                                        postprocess_time - forward_time,
                                        postprocess_time - start_time))
            sum_psnr = 0.
            sum_ssim = 0.
            sum_ssim2 = 0.
            n_img = 0
            for k in total_psnr.keys():
                if len(total_psnr[k]) != 0:
                    self.logger.write_log("# Video:{} AVG-PSNR={:.5}, AVG-SSIM={:.4}".format(
                        k, sum(total_psnr[k]) / len(total_psnr[k]), sum(total_ssim[k]) / len(total_ssim[k])))
                    #self.logger.write_log("# Video:{} AVG-PSNR={:.5}, AVG-SSIM={:.4}, AVG-SSIM2={:.4}".format(
                    #    k, sum(total_psnr[k]) / len(total_psnr[k]), sum(total_ssim[k]) / len(total_ssim[k]), sum(total_ssim2[k]) / len(total_ssim2[k])))
                sum_psnr += sum(total_psnr[k])
                sum_ssim += sum(total_ssim[k])
                #sum_ssim2 += sum(total_ssim2[k])
                n_img += len(total_psnr[k])
            if n_img != 0:
                self.logger.write_log("# Total AVG-PSNR={:.5}, AVG-SSIM={:.4}, AVG-SSIM2={:.4}".format(sum_psnr / n_img, sum_ssim / n_img))
                #self.logger.write_log("# Total AVG-PSNR={:.5}, AVG-SSIM={:.4}, AVG-SSIM2={:.4}".format(sum_psnr / n_img, sum_ssim / n_img, sum_ssim2 / n_img))

    def gene_seq(self, img_list, n_seq):
        if self.border:
            half = n_seq // 2
            img_list_temp = img_list[:half]
            img_list_temp.extend(img_list)
            img_list_temp.extend(img_list[-half:])
            img_list = img_list_temp
        seq_list = []
        for i in range(len(img_list) - 2 * (n_seq // 2)):
            seq_list.append(img_list[i:i + n_seq])
        return seq_list

    def numpy2tensor(self, input_seq, rgb_range=1.):
        tensor_list = []
        for img in input_seq:
            img = np.array(img).astype('float64')
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
            tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
            tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)
            tensor_list.append(tensor)
        stacked = torch.stack(tensor_list).unsqueeze(0)
        return stacked

    def tensor2numpy(self, tensor, rgb_range=1.):
        rgb_coefficient = 255 / rgb_range
        img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
        img = img[0].data
        img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        return img

    def get_PSNR_SSIM(self, output, gt, crop_border=4):
        cropped_output = output[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_GT = gt[crop_border:-crop_border, crop_border:-crop_border, :]
        psnr = self.calc_PSNR(cropped_GT, cropped_output)
        ssim = self.calc_SSIM(cropped_GT, cropped_output)
        return psnr, ssim

    def calc_PSNR(self, img1, img2):
        '''
        img1 and img2 have range [0, 255]
        '''
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def calc_SSIM(self, img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
        '''

        def ssim(img1, img2):
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            kernel = cv2.getGaussianKernel(11, 1.5)
            window = np.outer(kernel, kernel.transpose())

            mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
            mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                    (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()

        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDVD-TSP-Inference')

    parser.add_argument('--save_image', action='store_true',
                        help='save image if true')
    parser.add_argument('--border', action='store_true',
                        help='restore border images of video if true')
    parser.add_argument('--n_seq', type=int, default=5,
                        help='number of frames to evaluate for 1 output frame')
    parser.add_argument('--default_data', type=str, default='.',
                        help='quick test, optional: DVD, GOPRO')
    parser.add_argument('--data_path', type=Path, default=Path('../dataset/DVD/test'),
                        help='the path of test data')
    parser.add_argument('--model_path', type=Path, default=Path('../pretrain_models/CDVD_TSP_DVD_Convergent.pt'),
                        help='the path of pretrain model')
    parser.add_argument('--result_path', type=Path, default=Path('../infer_results'),
                        help='the path of deblur result')
    args = parser.parse_args()

    if args.default_data == 'DVD':
        args.data_path = Path('../dataset/DVD/test')
        args.model_path = Path('../pretrain_models/CDVD_TSP_DVD_Convergent.pt')
        args.result_path = Path('../infer_results/DVD')
    elif args.default_data == 'GOPRO':
        args.data_path = Path('../dataset/GOPRO/test')
        args.model_path = Path('../pretrain_models/CDVD_TSP_GOPRO.pt')
        args.result_path = Path('../infer_results/GOPRO')
    elif args.default_data == 'REDS':
        # There is no model trained on the REDS dataset
        args.data_path = Path('../dataset/REDS/test')
        args.model_path = Path('../pretrain_models/CDVD_TSP_DVD_Convergent.pt')
        args.result_path = Path('../infer_results/REDS')

    args.use_checkpoint = False
    
    Infer = Inference(args)
    Infer.infer()
