# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

from ast import NotEq
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DistributedSampler
import torch.nn.functional as F
import torchvision.transforms as transforms

import os
import math
import random
from glob import glob
import os.path as osp
from pathlib import Path
import re

from sklearn.model_selection import train_test_split
import cv2

from .utils import frame_utils
from .utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from .utils.utils import print0

shift_info_printed = False

# sparse: sparse (kitti .png) format of flow data
class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, normalize=False):
        self.augmentor = None
        self.sparse = sparse
        self.normalize = normalize
        if aug_params is not None:
            if sparse:
                # Only KITTI, HD1k, VIPER are sparse.
                self.augmentor = SparseFlowAugmentor(self.ds_name, **aug_params)
            else:
                self.augmentor = FlowAugmentor(self.ds_name, **aug_params)

            global shift_info_printed
            if not shift_info_printed and aug_params['shift_prob']:
                print("Shift aug: {}, prob {}".format( \
                      self.augmentor.shift_sigmas, self.augmentor.shift_prob))
                shift_info_printed = True

        # if is_test, do not return flow (only for LB submission).
        self.is_test = False
        self.init_seed = True
        self.flow_list = []
        self.image_list = []
        self.extra_info = None
        self.occ_list = None
        self.seg_list = None
        self.seg_inv_list = None

    def __getitem__(self, index):
        if self.extra_info is not None:
            extra_info = self.extra_info[index]
        else:
            extra_info = 0
        if self.normalize:
            input_transform = transforms.Compose([
                transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]), # (0,255) -> (0,1)
                #transforms.Normalize(mean=[0.45,0.432,0.411], std=[1,1,1]) # (0,1) -> (-0.5,0.5)
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # from ImageNet dataset
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Standard normalization
            ])
        else:
            input_transform = transforms.Compose([
                transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255])#, # (0,255) -> (0,1)
            ])
        # if is_test, do not return flow (only for LB submission).
        # If there's groundtruth flow, then is_test=False, e.g. on chairs and things.
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, extra_info

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        # KITTI flow is saved as image files. 
        # KITTI, HD1K, VIPER are sparse format.
        if self.sparse:
            if self.flow_list[index].suffix == '.npz':
                flow, valid = frame_utils.readFlowNPZ(self.flow_list[index])
            elif self.flow_list[index].suffix == '.png':
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            # read_gen: general read? choose reader according to the file extension.
            if self.flow_list[index].suffix == '.npz':
                flow, valid = frame_utils.readFlowNPZ(self.flow_list[index])
            else:
                flow = frame_utils.read_gen(self.flow_list[index])

        if self.occ_list is not None:
            occ = frame_utils.read_gen(self.occ_list[index])
            occ = np.array(occ).astype(np.uint8)
            occ = torch.from_numpy(occ // 255).bool()

        if self.seg_list is not None:
            f_in = np.array(frame_utils.read_gen(self.seg_list[index]))
            seg_r = f_in[:, :, 0].astype('int32')
            seg_g = f_in[:, :, 1].astype('int32')
            seg_b = f_in[:, :, 2].astype('int32')
            seg_map = (seg_r * 256 + seg_g) * 256 + seg_b
            seg_map = torch.from_numpy(seg_map)

        if self.seg_inv_list is not None:
            seg_inv = frame_utils.read_gen(self.seg_inv_list[index])
            seg_inv = np.array(seg_inv).astype(np.uint8)
            seg_inv = torch.from_numpy(seg_inv // 255).bool()

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            #img1 = np.tile(img1[...,None], (1, 1, 3))
            img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
            #img2 = np.tile(img2[...,None], (1, 1, 3))
            img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
        else:
            # Remove alpha? 
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        #img1 = input_transform(torch.from_numpy(img1).permute(2, 0, 1).float()).to('cuda')
        #img2 = input_transform(torch.from_numpy(img2).permute(2, 0, 1).float()).to('cuda')
        #flow = torch.from_numpy(flow).permute(2, 0, 1).float().to('cuda')

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                # shift augmentation will return valid. Otherwise valid is None.
                img1, img2, flow, valid = self.augmentor(img1, img2, flow)

        img1 = input_transform(torch.from_numpy(img1).permute(2, 0, 1).float())
        img2 = input_transform(torch.from_numpy(img2).permute(2, 0, 1).float())
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        images = torch.stack((img1, img2))

        if valid is not None:
            if not torch.is_tensor(valid):
                valid = torch.from_numpy(valid)
        else:
            if torch.is_tensor(img1):
                #valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
                flow_u = flow[0]
                flow_v = flow[1]
                h,w = flow_u.shape
                invalid = torch.logical_or(torch.logical_or(torch.isnan(flow_u), torch.isnan(flow_v)),
                                           torch.logical_or(torch.isinf(flow_u), torch.isinf(flow_v)))
                flow_u[invalid] = 0
                flow_v[invalid] = 0
                invalid = np.logical_or(invalid, np.logical_or(np.abs(flow_u) >= w, np.abs(flow_v) >= h))
                valid = torch.logical_not(invalid)
            else:
                #valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
                flow_u = flow[0]
                flow_v = flow[1]
                h,w = flow_u.shape
                invalid = np.logical_or(np.logical_or(np.isnan(flow_u), np.isnan(flow_v)),
                                        np.logical_or(np.isinf(flow_u), np.isinf(flow_v)))
                flow_u[invalid] = 0
                flow_v[invalid] = 0
                # Things was getting GT maximum flows greater than 1100 and it was resulting in nan for EPE
                # Viper was getting GT maximum flows greater than 65365 and it was resulting in nan for EPE
                # A flow greater than h or w has gone out of view an impossible to calculate from just 2 images
                invalid = np.logical_or(invalid, np.logical_or(np.abs(flow_u) >= w, np.abs(flow_v) >= h))
                valid = np.logical_not(invalid)
        flow = torch.cat((flow, valid[None,:,:]), dim=0)

        if self.occ_list is not None:
            return img1, img2, flow, valid.float(), occ, self.occ_list[index]
        elif self.seg_list is not None and self.seg_inv_list is not None:
            return img1, img2, flow, valid.float(), seg_map, seg_inv
        else:
            return images, flow

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        if self.extra_info is not None:
            self.extra_info = v * self.extra_info
        return self
        
    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(self,
                 aug_params   = None,
                 split        = 'training',
                 root         = 'datasets/MPI_Sintel',
                 dstype       = 'clean',
                 occlusion    = False,
                 segmentation = False,
                 debug        = False):
        self.ds_name = f'sintel-{split}-{dstype}'
        super(MpiSintel, self).__init__(aug_params)

        #flow_root = osp.join(root, split, 'flow')
        #image_root = osp.join(root, split, dstype)
        #occ_root = osp.join(root, split, 'occlusions')
        flow_root = root / split / 'flow'
        image_root = root / split / dstype
        occ_root = root / split / 'occlusions'
        # occ_root = osp.join(root, split, 'occ_plus_out')
        # occ_root = osp.join(root, split, 'in_frame_occ')
        # occ_root = osp.join(root, split, 'out_of_frame')
        if debug:
            self.extra_info = []

        self.segmentation = segmentation
        self.occlusion = occlusion
        if self.occlusion:
            self.occ_list = []
        if self.segmentation:
            seg_root = root / split / 'segmentation'
            seg_inv_root = root / split / 'segmentation_invalid'
            self.seg_list = []
            self.seg_inv_list = []

        if split == 'test':
            self.is_test = True

        
        #for scene in sorted(os.listdir(image_root)):    
        for scene in sorted((root / split / dstype).iterdir()):
            #image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            image_list = sorted((image_root / scene.name).rglob('*.png'))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                # i: frame_id, the sequence number of the image.
                # The first image in this folder is numbered 0.
                if debug:
                    self.extra_info += [ (scene.name, i) ] # scene and frame_id

            if split != 'test':
                #self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))
                self.flow_list += sorted((flow_root / scene.name).rglob('*.flo'))
                if self.occlusion:
                    #self.occ_list += sorted(glob(osp.join(occ_root, scene, '*.png')))
                    self.occ_list += sorted((occ_root / scene.name).rglob('*.png'))
                if self.segmentation:
                    #self.seg_list += sorted(glob(osp.join(seg_root, scene, '*.png')))
                    self.seg_list += sorted((seg_root / scene.name).rglob('*.png'))
                    #self.seg_inv_list += sorted(glob(osp.join(seg_inv_root, scene, '*.png')))
                    self.seg_inv_list += sorted((seg_inv_root / scene.name).rglob('*.png'))

        print(f"{self.ds_name}: {len(self.image_list)} image pairs.")

class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/FlyingChairs_release/data'):
        self.ds_name = f'chairs-{split}'
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('FlyingChairs_train_val.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]

        print(f"{self.ds_name}: {len(self.image_list)} image pairs.")
        if len(self.image_list) == 0:
            exit()

class FlyingChairs2(FlowDataset):
    def __init__(self,
                 aug_params = None,
                 split      = 'training',
                 root       = Path('../datasets/FlyingChairs2'),
                 normalize  = False):
        self.ds_name = f'chairs2-{split}'
        super(FlyingChairs2, self).__init__(aug_params, normalize=normalize)


        if split == 'training':
            train_path = root / 'train'
            images = sorted(train_path.rglob('*img*.png'))
            flows = sorted(train_path.rglob('*.npz'))
            assert (len(images) == len(flows))
            for i in range(len(flows)):
                if flows[i].name.find('flow_01') > 0:
                    #print('flows[i]',flows[i].name)
                    #print('images[i]',images[i].name)
                    self.flow_list += [flows[i]]
                    self.image_list += [ [images[i], images[i+1]] ]
                elif flows[i].name.find('flow_10') > 0:
                    self.flow_list += [flows[i]]
                    self.image_list += [ [images[i], images[i-1]] ]
        else:
            val_path = root / 'val'
            images = sorted(val_path.rglob('*img*.png'))
            flows = sorted(val_path.rglob('*.npz'))
            assert (len(images) == len(flows))
            for i in range(len(flows)):
                if flows[i].name.find('flow_01') > 0:
                    self.flow_list += [flows[i]]
                    self.image_list += [ [images[i], images[i+1]] ]
                elif flows[i].name.find('flow_10') > 0:
                    self.flow_list += [flows[i]]
                    self.image_list += [ [images[i], images[i-1]] ]

        print(f"{self.ds_name}: {len(self.image_list)} image pairs.")
        if len(self.image_list) == 0:
            exit()



class FlyingThings3D(FlowDataset):
    def __init__(self,
                 aug_params = None,
                 root       = Path('../datasets/FlyingThings3D_subset'),
                 split      = 'training',
                 dstype     = 'image_final',
                 normalize  = False):
        ds_type_short = { 'image_clean': 'clean', 
                          'image_final': 'final' }
        self.ds_name = f'things-{split}-{ds_type_short[dstype]}'
        super(FlyingThings3D, self).__init__(aug_params, normalize=normalize)

        if split == 'training':
            for cam in ['left','right']:
                image_dir = root / 'train' / dstype / cam

                for direction in ['into_future', 'into_past']:
                #for direction in ['into_future']:
                    flow_dir = root / 'train' / 'flow' / cam / direction
                    for flow in sorted(flow_dir.rglob('*.npz')):
                        index = int(flow.stem)

                        if direction == 'into_future':
                            img1 = str(index).zfill(7)+'.png'
                            img2 = str(index+1).zfill(7)+'.png'
                            img1_path = image_dir / img1
                            img2_path = image_dir / img2

                            if  img1_path.is_file() \
                            and img2_path.is_file():
                                self.image_list += [ [img1_path, img2_path] ]
                                self.flow_list += [ flow ]
                        elif direction == 'into_past':
                            img1 = str(index).zfill(7)+'.png'
                            img2 = str(index-1).zfill(7)+'.png'
                            img1_path = image_dir / img1
                            img2_path = image_dir / img2

                            if  img1_path.is_file() \
                            and img2_path.is_file():
                                self.image_list += [ [img1_path, img2_path] ]
                                self.flow_list += [ flow ]

        elif split == 'validation':
            for cam in ['left','right']:
                image_dir = root / 'val' / dstype / cam

                #for direction in ['into_future', 'into_past']:
                for direction in ['into_future']:
                    flow_dir = root / 'train' / 'flow' / cam / direction
                    for flow in sorted(flow_dir.rglob('*.flo')):
                        index = int(flow.stem)

                        if direction == 'into_future':
                            img1 = str(index).zfill(7)+'.png'
                            img2 = str(index+1).zfill(7)+'.png'
                            img1_path = image_dir / img1
                            img2_path = image_dir / img2

                            if  img1_path.is_file() \
                            and img2_path.is_file():
                                self.image_list += [ [img1_path, img2_path] ]
                                self.flow_list += [ flow ]
                        elif direction == 'into_past':
                            img1 = str(index).zfill(7)+'.png'
                            img2 = str(index-1).zfill(7)+'.png'
                            img1_path = image_dir / img1
                            img2_path = image_dir / img2

                            if  img1_path.is_file() \
                            and img2_path.is_file():
                                self.image_list += [ [img1_path, img2_path] ]
                                self.flow_list += [ flow ]
      
        print(f"{self.ds_name}: {len(self.image_list)} image pairs.")
        if len(self.image_list) == 0:
            exit()



class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI',
                 debug=False):
        self.ds_name = f'kitti-{split}'
        super(KITTI, self).__init__(aug_params, sparse=True)

        if debug:
            self.extra_info = []

        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.image_list += [ [img1, img2] ]
            if debug:
                self.extra_info += [ [frame_id] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))

        print(f"{self.ds_name}: {len(self.image_list)} image pairs.")

# Further split KITTI training data into training and testing sets.
class KITTITrain(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI',
                 debug=False):
        self.ds_name = f'kittitrain-{split}'
        super(KITTITrain, self).__init__(aug_params, sparse=True)

        root = osp.join(root, "training")
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
        flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        extra_info = []
        image_list = []
        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            image_list += [ [img1, img2] ]
            extra_info += [ [frame_id] ]

        image_list_train, image_list_test, flow_list_train, flow_list_test, \
            extra_info_train, extra_info_test = \
                    train_test_split(image_list, flow_list, extra_info, test_size=0.3, random_state=42)

        if split == 'training':
            self.image_list = image_list_train
            self.flow_list = flow_list_train
            if debug:
                self.extra_info = extra_info_train
        else:
            self.image_list = image_list_test
            self.flow_list = flow_list_test
            if debug:
                self.extra_info = extra_info_test

        print(f"{self.ds_name}: {len(self.image_list)} image pairs.")

class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        self.ds_name = f'hd1k'
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1

        print(f"{self.ds_name}: {len(self.image_list)} image pairs.")

class Autoflow(FlowDataset):
    def __init__(self,
                 aug_params = None,
                 split      = 'training',
                 root       = Path('../datasets/AutoFlow'),
                 debug      = False,
                 normalize  = False):
        self.ds_name = f'autoflow-{split}'
        super(Autoflow, self).__init__(aug_params, normalize=normalize)

        scene_count = len(list(root.iterdir()))
        #training_size = int(scene_count * 0.9)
        training_size = int(scene_count * 1.1)
        if debug:
            self.extra_info = []
        
        #for i, scene in enumerate(sorted(os.listdir(root))):
        for i, scene in enumerate(sorted(root.iterdir())):
            if split == 'training' and i <= training_size or \
               split == 'test'     and i > training_size:
                image0_path = scene / 'im0.png'
                image1_path = scene / 'im1.png'
                flow_path   = scene / 'forward.flo'
                
                self.image_list += [ [image0_path, image1_path] ]
                self.flow_list  += [ flow_path ]
                if debug:
                    self.extra_info += [ [scene.name] ]

        print(f"{self.ds_name}: {len(self.image_list)} image pairs.")

# The VIPER .npz flow files have been converted to KITTI .png format.
class VIPER(FlowDataset):
    def __init__(self,
                 aug_params = None,
                 split      = 'training',
                 root       = Path('../datasets/VIPER/'),
                 filetype   = 'png',
                 debug      = False,
                 modulo     = 10,
                 normalize  = False):
        self.ds_name = f'viper-{split}'
        super(VIPER, self).__init__(aug_params, sparse=True, normalize=normalize)

        split_map = { 'training': 'train', 'validation': 'val', 'test': 'test' }
        split = split_map[split]
        split_img_root  = root / split / 'img'
        split_flow_root = root / split / 'flow'
        skip_count = 0
        if debug:
            self.extra_info = []

        if split == 'test':
            # 001_00001, 001_00076, ...
            TEST_FRAMES = open(root / "test_frames.txt")
            test_frames_dict = {}
            for frame_trunk in TEST_FRAMES:
                frame_trunk = frame_trunk.strip()
                test_frames_dict[frame_trunk] = 1
            print0("{} test frame names loaded".format(len(test_frames_dict)))
            self.is_test = True
            
        for i, scene in enumerate(sorted(split_flow_root.iterdir())):
            # scene: 001, 002, ...
            # dir: viper/train/img/001
            # img0_name: 001_00001.png, 001_00010.png, ...
            for flow_path in sorted(scene.iterdir()):
                #print('img0_name',img0_name.name)
                matches = re.search(r"(\d{3})_(\d{5})(\.npz)", flow_path.name)
                #if not matches:
                #    breakpoint()
                scene0   = matches.group(1)
                flow_idx = matches.group(2)
                suffix   = matches.group(3)
                assert scene.name == scene0
                #print('suffix',suffix,'flow_path.suffix',flow_path.suffix)
                assert suffix == flow_path.suffix
                # img0_trunk: flow_path without suffix.
                img0_trunk  = f"{scene0}_{flow_idx}"
                if ((split == 'train' or split == 'val') and int(flow_idx) % modulo == 0) \
                  or (split == 'test' and img0_trunk in test_frames_dict):
                    img1_idx    = "{:05d}".format(int(flow_idx) + 1)
                    img0_name   = f"{scene0}_{flow_idx}.{filetype}"
                    img1_name   = f"{scene0}_{img1_idx}.{filetype}"
                    #flow_name   = flow_path.stem + "." + filetype
                    image0_path = split_img_root / scene0 / img0_name
                    image1_path = split_img_root / scene0 / img1_name
                    #flow_path   = split_flow_root / scene0 / flow_name
                    # Sometimes image1 is missing. Skip this pair.
                    #if not os.path.isfile(image1_path):
                    if not image0_path.is_file():
                        #print('does not exist:',image0_path,' for ',flow_path)
                        skip_count += 1
                        continue
                    if not image1_path.is_file():
                        #print('does not exist:',image1_path,' for ',flow_path)
                        skip_count += 1
                        continue
                    # if both image0_path and image1_path exist, then flow_path should exist.
                    #if split != 'test' and not os.path.isfile(flow_path):
                    if split != 'test' and not flow_path.is_file():
                        print('does not exist:',flow_path)
                        skip_count += 1
                        continue
                # This file is not considered as the first frame. Skip.
                else:
                    skip_count += 1
                    continue
                        
                self.image_list += [ [image0_path, image1_path] ]
                self.flow_list  += [ flow_path ]
                if debug:
                    self.extra_info += [ [img0_trunk] ]

        print0(f"{self.ds_name}: {len(self.image_list)} image pairs. {skip_count} files skipped.")

class SlowFlow(FlowDataset):
    def __init__(self, aug_params=None, split='test', root='datasets/slowflow/', filetype='png', 
                 blur_mag=100, blur_num_frames=0, debug=True):
        self.ds_name = f'slowflow-{split}-{blur_mag}-{blur_num_frames}'
        super(SlowFlow, self).__init__(aug_params, sparse=False)

        sequence_folder = "sequence" if blur_num_frames == 0 else f"sequence_R0{blur_num_frames}"
        sequence_root = osp.join(root, str(blur_mag), sequence_folder)
        print0(sequence_root)
        flow_root = osp.join(root, str(blur_mag), 'flow')
        skip_count = 0
        if debug:
            self.extra_info = []

        for i, scene in enumerate(sorted(os.listdir(sequence_root))):
            # scene: Animals, Ball...
            # img0_name: seq5_0000000.png, seq5_0000001.png, ...
            for img0_name in sorted(os.listdir(osp.join(sequence_root, scene))):
                matches = re.match(r"seq(\d+)_(\d+).png", img0_name)
                #if not matches:
                #    breakpoint()
                subseq_idx  = matches.group(1)
                img0_idx    = matches.group(2)
                # This image is img1. Skip.
                if img0_idx[-1] == '1':
                    continue
                #if img0_idx[-1] != '0':
                #    breakpoint()
                # img0_trunk: img0_name without suffix.
                img0_trunk  = f"seq{subseq_idx}_{img0_idx}"
                img1_idx    = img0_idx[:-1] + '1'
                img1_name   = f"seq{subseq_idx}_{img1_idx}.png"
                flow_name   = img0_trunk + ".flo"
                image0_path = osp.join(sequence_root, scene, img0_name)
                image1_path = osp.join(sequence_root, scene, img1_name)
                flow_path   = osp.join(flow_root,     scene, flow_name)
                if not os.path.isfile(flow_path):
                    skip_count += 1
                    continue

                self.image_list += [ [image0_path, image1_path] ]
                self.flow_list  += [ flow_path ]
                if debug:
                    self.extra_info += [ [scene, img0_trunk] ]

        print0(f"{self.ds_name}: {len(self.image_list)} image pairs. {skip_count} skipped")


# 'crop_size' is first used to bound the minimal size of images after resizing. Then it's used to crop the image.
def fetch_dataloader(args, mini_batch_size=8, stage='autoflow', image_size=[488, 576], SINTEL_TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding training set """

    if stage == 'chairs':
        aug_params = {'crop_size': image_size, 'min_scale': -0.1, 'max_scale': 1.0, 
                      'do_flip': True,
                      'shift_prob': args.shift_aug_prob, 'shift_sigmas': args.shift_sigmas }
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif stage == 'chairs2':
        aug_params = {'crop_size': image_size, 'min_scale': -0.1, 'max_scale': 1.0, 
                      'do_flip': True,
                      'shift_prob': args.shift_aug_prob, 'shift_sigmas': args.shift_sigmas }
        train_dataset = FlyingChairs2(aug_params, split='training', normalize=args.normalize)
    
    elif stage == 'things':
        # For experiments to understand inborn vs. acquired robustness against image shifting, 
        # only do image shifting augmentation on Things.
        # Things is non-sparse. So only need to work on FlowAugmentor 
        # (no need to work on SparseFlowAugmentor).
        aug_params = {'crop_size': image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True,
                      'shift_prob': args.shift_aug_prob, 'shift_sigmas': args.shift_sigmas }
        things_clean  = FlyingThings3D(aug_params, dstype='image_clean', split='training', normalize=args.normalize)
        things_final  = FlyingThings3D(aug_params, dstype='image_final', split='training', normalize=args.normalize)
        train_dataset = things_clean + things_final

    elif stage == 'things_clean':
        # For experiments to understand inborn vs. acquired robustness against image shifting, 
        # only do image shifting augmentation on Things.
        # Things is non-sparse. So only need to work on FlowAugmentor 
        # (no need to work on SparseFlowAugmentor).
        aug_params = {'crop_size': image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True,
                      'shift_prob': args.shift_aug_prob, 'shift_sigmas': args.shift_sigmas }
        things_clean  = FlyingThings3D(aug_params, dstype='image_clean', split='training', normalize=args.normalize)
        train_dataset = things_clean

    elif stage == 'things_final':
        # For experiments to understand inborn vs. acquired robustness against image shifting, 
        # only do image shifting augmentation on Things.
        # Things is non-sparse. So only need to work on FlowAugmentor 
        # (no need to work on SparseFlowAugmentor).
        aug_params = {'crop_size': image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True,
                      'shift_prob': args.shift_aug_prob, 'shift_sigmas': args.shift_sigmas }
        things_final  = FlyingThings3D(aug_params, dstype='image_final', split='training', normalize=args.normalize)
        train_dataset = things_final

    elif stage == 'autoflow':
        # autoflow image size: (488, 576)
        # minimal scale = 2**0.42 = 1.338. 576*1.338=770.6 > 768. Otherwise there'll be exceptions.
        train_dataset = Autoflow({'crop_size': image_size, 'min_scale': -0.2, 'max_scale': 0.8, 
                                  'spatial_aug_prob': 1, 'do_flip': True,
                                  'shift_prob': args.shift_aug_prob, 'shift_sigmas': args.shift_sigmas }, normalize=args.normalize)

    elif stage == 'sintel':
        aug_params = {'crop_size': image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True,
                      'shift_prob': args.shift_aug_prob, 'shift_sigmas': args.shift_sigmas }
        things_clean = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')
        viper = VIPER(aug_params, split='training')

        if SINTEL_TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True,
                           'shift_prob': args.shift_aug_prob, 'shift_sigmas': args.shift_sigmas })
            hd1k = HD1K({'crop_size': image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True,
                         'shift_prob': args.shift_aug_prob, 'shift_sigmas': args.shift_sigmas })
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things_clean

        elif SINTEL_TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things_clean

    elif stage == 'SKHTV':
        aug_params = {'crop_size': image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True,
                      'shift_prob': args.shift_aug_prob, 'shift_sigmas': args.shift_sigmas }
        things_clean = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')
        kitti = KITTI({'crop_size': image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True,
                       'shift_prob': args.shift_aug_prob, 'shift_sigmas': args.shift_sigmas })
        hd1k = HD1K({'crop_size': image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True,
                     'shift_prob': args.shift_aug_prob, 'shift_sigmas': args.shift_sigmas })
        viper = VIPER(aug_params, split='training')
        train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things_clean
    
    elif stage == 'kitti':
        aug_params = {'crop_size': image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False,
                      'shift_prob': args.shift_aug_prob, 'shift_sigmas': args.shift_sigmas }
        train_dataset = KITTI(aug_params, split='training')

    elif stage == 'kittitrain':
        aug_params = {'crop_size': image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False,
                      'shift_prob': args.shift_aug_prob, 'shift_sigmas': args.shift_sigmas }
        train_dataset = KITTITrain(aug_params, split='training')

    elif stage == 'viper':
        aug_params = {'crop_size': image_size, 'min_scale': -1, 'max_scale': -0.5, 
                      'spatial_aug_prob': 1, 'do_flip': False,
                      'shift_prob': args.shift_aug_prob, 'shift_sigmas': args.shift_sigmas }
        train_dataset = VIPER(aug_params, split='training', normalize=args.normalize)
        #train_dataset = VIPER(aug_params, split='training', modulo=3, normalize=args.normalize)

    elif stage == 'viper1':
        aug_params = {'crop_size': image_size, 'min_scale': -1, 'max_scale': -0.5, 
                      'spatial_aug_prob': 1, 'do_flip': False,
                      'shift_prob': args.shift_aug_prob, 'shift_sigmas': args.shift_sigmas }
        train_dataset = VIPER(aug_params, split='training', modulo=1, normalize=args.normalize)

    elif stage == 'viper3':
        aug_params = {'crop_size': image_size, 'min_scale': -1, 'max_scale': -0.5, 
                      'spatial_aug_prob': 1, 'do_flip': False,
                      'shift_prob': args.shift_aug_prob, 'shift_sigmas': args.shift_sigmas }
        train_dataset = VIPER(aug_params, split='training', modulo=3, normalize=args.normalize)

    if args.ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = data.DataLoader(train_dataset,
                                       batch_size  = mini_batch_size,
                                       sampler     = train_sampler,
                                       pin_memory  = True,
                                       num_workers = args.num_workers,
                                       drop_last   = True)
    else:
        #print('mini_batch_size',mini_batch_size,'args.num_workers',args.num_workers)
        train_loader = data.DataLoader(train_dataset,
                                       batch_size  = mini_batch_size,
                                       pin_memory  = True,
                                       num_workers = args.num_workers,
                                       shuffle     = True,
                                       drop_last   = True)

    print0('Training with %d image pairs' % len(train_dataset))
    return train_loader

