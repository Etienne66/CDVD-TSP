""" Load Data from KTTI 2012 & 2015

From https://github.com/PruneTruong/GLU-Net/blob/master/datasets/KITTI_optical_flow.py

Updated for Python 3.9 and Pytorch 1.10

"""
__author__ = "Prune Truong, Martin Danelljan and Radu Timofte"
__copyright__ = "Copyright 2021, Computer Vision Lab, D-ITET, ETH Zürich, Switzerland"
__credits__ = ["Prune Truong", "Martin Danelljan", "Radu Timofte", "Steven Wilson"]
__license__ = "GPL-3.0"
__version__ = "1.0.1"
__maintainer__ = "Steven Wilson"
__email__ = "Etienne66"
__status__ = "Development"

import os.path
import glob
from data_flow.listdataset import ListDataset
from data_flow.util import split2list
from collections import deque
import numpy as np
from pathlib import Path
import imageio

try:
    import cv2
except ImportError as e:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn("failed to load openCV, which is needed"
                      "for KITTI which uses 16bit PNG images", ImportWarning)

'''
extracted from https://github.com/ClementPinard/FlowNetPytorch/tree/master/datasets
Dataset routines for KITTI_flow, 2012 and 2015.
http://www.cvlibs.net/datasets/kitti/eval_flow.php
OpenCV is needed to load 16bit png images
'''


def load_flow_from_png(png_path):
    # The -1 is here to specify not to change the image depth (16bit), and is compatible
    # with both OpenCV2 and OpenCV3
    flo_file = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    #print('flo_file: ', flo_file.dtype, flo_file.shape)
    flo_img = flo_file[:, :, 2:0:-1].astype(np.float32)
    #invalid = (flo_file[:,:,0] == 0) # in cv2 change the channel to the first, mask of invalid pixels
    valid = (flo_file[:,:,0] == 1)
    flo_img = flo_img - 32768
    flo_img = flo_img / 64
    #flo_img[np.abs(flo_img) < 1e-10] = 0#1e-10
    #flo_img[invalid, :] = 0
    #print('flow_image: ', flo_img.dtype, flo_img.shape)
    #flo_img2 = flo_img
    #flo_img2[:,:,0] *= invalid
    #flo_img2[:,:,1] *= invalid
    #np.testing.assert_array_equal(flo_img, flo_img2)
    return flo_img, valid.astype(np.uint8)


def make_dataset(root, train=True, occ=True):
    '''
    ATTENTION HERE I MODIFIED WHICH IMAGE IS THE TARGET OR NOT
    Will search in training folder for folders 'flow_noc' or 'flow_occ'
       and 'colored_0' (KITTI 2012) or 'image_2' (KITTI 2015) '''
    if train:
        root = root / 'training'
    else:
        root = root / 'testing'

    flow_dir = 'flow_occ' if occ else 'flow_noc'
    assert((root / flow_dir).is_dir())
    img_dir = 'colored_0'
    if not (root / img_dir).is_dir():
        img_dir = 'image_2'
    assert((root / img_dir).is_dir())

    images = []
    #print(root / flow_dir)
    for flow_map in (root / flow_dir).rglob('*.png'):
        f = flow_map.stem.split('_')
        root_filename = f[0] # name of image
        #print(root_filename)
        img1 = root / img_dir / (root_filename+'_10.png')
        img2 = root / img_dir / (root_filename+'_11.png') # this corresponds to target
        if not (    img1.is_file()
                and img2.is_file()):
            continue
        images.append([[img1, img2], flow_map])
    return images


def KITTI_flow_loader(root, path_imgs, path_flo):
    #print(path_flo)
    flow, mask = load_flow_from_png(path_flo)
    return [imageio.imread(img).astype(np.float32) for img in path_imgs], flow, mask
    #return [cv2.imread(str(img))[:,:,::-1].astype(np.float32) for img in path_imgs], flow, mask


def KITTI_2012_occ(root,
                   transform=None,
                   target_transform=None,
                   co_transform=None,
                   train=True,
                   lr_finder=False):
    train_list = make_dataset(root / 'KITTI_2012', train=train, occ=True)
    train_dataset = ListDataset(root / 'KITTI_2012', train_list, transform=transform,
                                target_transform=target_transform, co_transform=co_transform,
                                loader=KITTI_flow_loader, mask=True, lr_finder=False)
    return train_dataset


def KITTI_2012_noc(root,
                   transform=None,
                   target_transform=None,
                   co_transform=None,
                   train=True,
                   lr_finder=False):
    train_list = make_dataset(root / 'KITTI_2012', train=train, occ=False)
    train_dataset = ListDataset(root / 'KITTI_2012', train_list, transform=transform,
                                target_transform=target_transform, co_transform=co_transform,
                                loader=KITTI_flow_loader, mask=True, lr_finder=False)
    return train_dataset

def KITTI_2015_occ(root,
                   transform=None,
                   target_transform=None,
                   co_transform=None,
                   train=True,
                   lr_finder=False):
    train_list = make_dataset(root / 'KITTI_2015', train=train, occ=True)
    train_dataset = ListDataset(root / 'KITTI_2015', train_list, transform=transform,
                                target_transform=target_transform, co_transform=co_transform,
                                loader=KITTI_flow_loader, mask=True, lr_finder=False)
    return train_dataset


def KITTI_2015_noc(root,
                   transform=None,
                   target_transform=None,
                   co_transform=None,
                   train=True,
                   lr_finder=False):
    train_list = make_dataset(root / 'KITTI_2015', train=train, occ=False)
    train_dataset = ListDataset(root / 'KITTI_2015', train_list, transform=transform,
                                target_transform=target_transform, co_transform=co_transform,
                                loader=KITTI_flow_loader, mask=True, lr_finder=False)
    return train_dataset
