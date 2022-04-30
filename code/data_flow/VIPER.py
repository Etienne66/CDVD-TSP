""" VIPER data set

@InProceedings{Richter_2017,
            title = {Playing for Benchmarks},
            author = {Stephan R. Richter and Zeeshan Hayder and Vladlen Koltun},
            booktitle = {{IEEE} International Conference on Computer Vision, {ICCV} 2017, Venice, Italy, October 22-29, 2017},
            pages = {2232--2241},
            year = {2017},
            url = {https://doi.org/10.1109/ICCV.2017.243},
            doi = {10.1109/ICCV.2017.243},            
        }

"""
__author__ = "Steven Wilson"
__copyright__ = "Copyright 2022"
__credits__ = ["Steven Wilson"]
__license__ = "GPL-3.0"
__version__ = "1.0.0"
__maintainer__ = "Steven Wilson"
__email__ = "Etienne66"
__status__ = "Development"

from pathlib import Path
import numpy as np
from .listdataset import ListDataset
from .util import split2list
import imageio


def make_dataset(dataset, train=True, final=True):
    '''Will search for triplets that go by the pattern
        '[int].png  [int+1].ppm    [name]_flow01.png' forward flow
    or  '[int].png  [int-1].ppm    [name]_flow10.png' backward flow
    '''

    if train:
        flow_path = dataset / 'VIPER/train/flow'
        flowbw_path = dataset / 'VIPER/train/flowbw'
        img_path = dataset / 'VIPER/train/img'
    else:
        flow_path = dataset / 'VIPER/val/flow'
        flowbw_path = dataset / 'VIPER/val/flowbw'
        img_path = dataset / 'VIPER/val/image_final'

    images = []

    for scene_dir in (flow_path).iterdir():
        if scene_dir.is_dir():
            for flow_map in scene_dir.rglob('*.npz'):
                f = flow_map.stem.split('_')
                assert(f[1].isnumeric())
                frame = int(f[1])
                img1 = f[0]+'_'+str(frame).zfill(5)+'.png'
                img2 = f[0]+'_'+str(frame+1).zfill(5)+'.png'
                img1_path = img_path / scene_dir.name / img1
                img2_path = img_path / scene_dir.name / img2
                if not (    img1_path.is_file()
                        and img2_path.is_file()):
                    continue
                images.append([[img1_path,img2_path],flow_map])

    if flowbw_path.is_dir():
        for scene_dir in (flowbw_path).iterdir():
            if scene_dir.is_dir():
                for flow_map in scene_dir.rglob('*.npz'):
                    f = flow_map.stem.split('_')
                    assert(f[1].isnumeric())
                    frame = int(f[1])
                    img1 = f[0]+'_'+str(frame).zfill(5)+'.png'
                    img2 = f[0]+'_'+str(frame-1).zfill(5)+'.png'
                    img1_path = img_path / scene_dir.name / img1
                    img2_path = img_path / scene_dir.name / img2
                    if not (    img1_path.is_file()
                            and img2_path.is_file()):
                        continue
                    images.append([[img1_path,img2_path],flow_map])
    return images


def Viper_flow_loader(root, path_imgs, path_flo):
    with np.load(path_flo) as data: #, allow_pickle=True, encoding='bytes', fix_imports=True
        flow_u = data['u'].astype(np.float32)
        flow_v = data['v'].astype(np.float32)
    h,w = flow_u.shape
    invalid = np.logical_or(np.logical_or(np.isnan(flow_u), np.isnan(flow_v)),
                            np.logical_or(np.isinf(flow_u), np.isinf(flow_v)))
    flow_u[invalid] = 0
    flow_v[invalid] = 0

    invalid = np.logical_or(invalid, np.logical_or(np.abs(flow_u) >= h, np.abs(flow_v) >= w))
    flow_u[invalid] = 0
    flow_v[invalid] = 0
    
    mask = np.logical_not(invalid)
    
    #print('flow_u',flow_u)
    #print('flow_v',flow_v)
    #print('flow.type',type(flow))
    #print('flow.min',np.amin(flow))
    #print('flow.max',np.amax(flow))
    
    flow = np.stack((flow_u, flow_v), axis=-1)
    #print('flow.size',flow.shape)
    #assert(not(np.isnan(np.amin(flow)) or np.isnan(np.amax(flow))))
    return [imageio.imread(img).astype(np.uint8) for img in path_imgs], flow, mask


def viper(root,
          transform=None,
          target_transform=None,
          co_transform=None,
          train=True):
    train_list = make_dataset(root, train=train, final=False)
    train_dataset = ListDataset(root, train_list, transform=transform,
                                target_transform=target_transform, co_transform=co_transform,
                                loader=Viper_flow_loader, mask=True)

    return train_dataset


