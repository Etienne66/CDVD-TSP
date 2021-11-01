import os
import sys
import glob
import numpy as np
import imageio
import torch
import torch.utils.data as data
import utils.utils as utils
from pathlib import Path

class VIDEODATA(data.Dataset):
    def __init__(self, args, name='', train=True):
        self.args = args
        self.name = name
        self.train = train
        self.n_seq = args.n_sequence
        self.n_frames_per_video = args.n_frames_per_video
        print("n_seq:", args.n_sequence)
        print("n_frames_per_video:", args.n_frames_per_video)

        self.n_frames_video = []
        self.images_gt = []
        self.images_input = []

        if train:
            for image_dataset in args.dir_data.split(','):
                dir_data = Path('../dataset') / image_dataset / 'train'
                self._set_filesystem(dir_data, name=image_dataset)
                images_gt, images_input = self._scan()
                self.images_gt += images_gt
                self.images_input += images_input
        else:
            for image_dataset in args.dir_data_test.split(','):
                dir_data = Path('../dataset') / image_dataset / 'test'
                self._set_filesystem(dir_data, name=image_dataset)
                images_gt, images_input = self._scan()
                self.images_gt += images_gt
                self.images_input += images_input

        self.num_video = len(self.images_gt)
        self.num_frame = sum(self.n_frames_video) - (self.n_seq - 1) * len(self.n_frames_video)
        print("Number of videos to load:", self.num_video)
        print("Number of frames to load:", self.num_frame)

        if train:
            self.repeat = max(args.test_every // max((self.num_frame // self.args.batch_size), 1), 1)
            print("Dataset repeat:", self.repeat)

        if args.process:
            self.data_gt, self.data_input = self._load(self.images_gt, self.images_input)

    def _scan(self):
        vid_gt_names = sorted(self.dir_gt.iterdir())
        vid_input_names = sorted(self.dir_input.iterdir())
        assert len(vid_gt_names) == len(vid_input_names), "len(vid_gt_names) must equal len(vid_input_names)"

        images_gt = []
        images_input = []

        for vid_gt_name, vid_input_name in zip(vid_gt_names, vid_input_names):
            if self.train:
                gt_dir_names = sorted(vid_gt_name.iterdir())[:self.args.n_frames_per_video]
                input_dir_names = sorted(vid_input_name.iterdir())[:self.args.n_frames_per_video]
            else:
                gt_dir_names = sorted(vid_gt_name.iterdir())
                input_dir_names = sorted(vid_input_name.iterdir())
            images_gt.append(gt_dir_names)
            images_input.append(input_dir_names)
            self.n_frames_video.append(len(gt_dir_names))

        return images_gt, images_input

    def _load(self, images_gt, images_input):
        data_input = []
        data_gt = []

        n_videos = len(images_gt)
        for idx in range(n_videos):
            if idx % 10 == 0:
                print("Loading video %d" % idx)
            #gts = np.array([imageio.imread(hr_name) for hr_name in images_gt[idx]])
            #inputs = np.array([imageio.imread(lr_name) for lr_name in images_input[idx]])
            #data_input.append(inputs)
            #data_gt.append(gts)
            data_input.append(np.asarray([imageio.imread(hr_name) for hr_name in images_gt[idx]]))
            data_gt.append(np.asarray([imageio.imread(lr_name) for lr_name in images_input[idx]]))

        return data_gt, data_input

    def __getitem__(self, idx):
        if self.args.process:
            inputs, gts, filenames = self._load_file_from_loaded_data(idx)
        else:
            inputs, gts, filenames = self._load_file(idx)

        inputs, gts = self.get_patch_frames(inputs, gts, self.args.size_must_mode)
        
        input_tensors = utils.np2Tensor(*inputs, rgb_range=self.args.rgb_range)
        gt_tensors = utils.np2Tensor(*gts, rgb_range=self.args.rgb_range)
       
        return torch.stack(input_tensors), torch.stack(gt_tensors), filenames

    def __len__(self):
        if self.train:
            return self.num_frame * self.repeat
        else:
            return self.num_frame

    def _get_index(self, idx):
        if self.train:
            return idx % self.num_frame
        else:
            return idx

    def _find_video_num(self, idx, n_frame):
        for i, j in enumerate(n_frame):
            if idx < j:
                return i, idx
            else:
                idx -= j

    def _load_file(self, idx):
        idx = self._get_index(idx)

        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
        f_gts = self.images_gt[video_idx][frame_idx:frame_idx + self.n_seq]
        f_inputs = self.images_input[video_idx][frame_idx:frame_idx + self.n_seq]
        gts = np.asarray([imageio.imread(hr_name) for hr_name in f_gts])
        inputs = np.asarray([imageio.imread(lr_name) for lr_name in f_inputs])
        filenames = [name.parent.parts[-1] + '.' + name.stem
                     for name in f_inputs]

        return inputs, gts, filenames

    def _load_file_from_loaded_data(self, idx):
        idx = self._get_index(idx)

        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
        gts = self.data_gt[video_idx][frame_idx:frame_idx + self.n_seq]
        inputs = self.data_input[video_idx][frame_idx:frame_idx + self.n_seq]
        filenames = [name.parent.parts[-1] + '.' + name.stem
                     for name in self.images_gt[video_idx][frame_idx:frame_idx + self.n_seq]]

        return inputs, gts, filenames

    def get_patch(self, input, gt, size_must_mode=1):
        if self.train:
            if not self.args.no_patch:
                input, gt = utils.get_patch(input, gt, patch_size=self.args.patch_size)
            elif size_must_mode > 1:
                h, w, c = input.shape
                new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
                input, gt = input[:new_h, :new_w, :], gt[:new_h, :new_w, :]

            if not self.args.no_augment and not self.args.no_patch:
                input, gt = utils.data_augment(input, gt)
        else:
            h, w, c = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input, gt = input[:new_h, :new_w, :], gt[:new_h, :new_w, :]
        return input, gt

    def get_patch_frames(self, input, gt, size_must_mode=1):
        if self.train:
            if not self.args.no_patch:
                input, gt = utils.get_patch_frames(input, gt, patch_size=self.args.patch_size)
            elif size_must_mode > 1:
                _, h, w, c = input.shape
                new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
                input, gt = input[:, :new_h, :new_w, :], gt[:, :new_h, :new_w, :]

            if not self.args.no_augment and not self.args.no_patch:
                input, gt = utils.data_augment_frames(input, gt)
        else:
            _, h, w, c = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input, gt = input[:, :new_h, :new_w, :], gt[:, :new_h, :new_w, :]
        return input, gt
