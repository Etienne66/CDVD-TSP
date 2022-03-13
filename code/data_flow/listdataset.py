import torch.utils.data as data
import os
import os.path
from imageio import imread
import numpy as np
from data.util import load_flo
import torch
from distutils.version import LooseVersion


def get_gt_correspondence_mask(flow):
    # convert flow to mapping
    flow = np.nan_to_num(flow, nan=-1e10, posinf=1e10, neginf=-1e10)
    h,w = flow.shape[:2]
    X, Y = np.meshgrid(np.linspace(0, w - 1, w),
                       np.linspace(0, h - 1, h))
    map_x = (flow[:,:,0]+X).astype(np.float32)
    map_y = (flow[:,:,1]+Y).astype(np.float32)
    mask_x = np.logical_and(map_x>0, map_x< w)
    mask_y = np.logical_and(map_y>0, map_y< h)
    mask = np.logical_and(mask_x, mask_y).astype(np.uint8)
    return mask


def default_loader(root, path_imgs, path_flo):
    return [imread(img).astype(np.uint8) for img in path_imgs], load_flo(path_flo)


class ListDataset(data.Dataset):
    def __init__(self, root, path_list,
                 transform=None,
                 target_transform=None,
                 co_transform=None,
                 loader=default_loader,
                 mask=False,
                 size=False,
                 lr_finder=False):
        """

        :param root: directory containing the dataset images
        :param path_list: list containing the name of images and corresponding ground-truth flow files
        :param transform: transforms to apply to source images
        :param target_transform: transforms to apply to flow field
        :param co_transform: transforms to apply to both images and the flow field
        :param loader: loader function for the images and the flow
        :param mask: bool indicating is a mask of valid pixels needs to be loaded as well from root
        :param size: size of the original source image
        outputs:
            - source_image
            - target_image
            - flow_map
            - correspondence_mask
            - source_image_size
        """

        self.root = root
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader
        self.mask = mask
        self.size = size
        self.lr_finder = lr_finder

    def __getitem__(self, index):
        inputs, target = self.path_list[index]

        if self.mask:
            inputs, target, mask = self.loader(self.root, inputs, target)
        else:
            inputs, target = self.loader(self.root, inputs, target)
            mask = get_gt_correspondence_mask(target)

        target = np.concatenate((target, mask[:,:,None]), axis=2)
        
        if self.co_transform is not None:
            #inputs, target, mask = self.co_transform(inputs, target, mask)
            inputs, target = self.co_transform(inputs, target)
        if self.transform is not None:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])
        if self.target_transform is not None:
            target = self.target_transform(target)

        #mask = mask.astype(np.bool) if LooseVersion(torch.__version__) >= LooseVersion('1.3') else mask.astype(np.float32)
        inputs = torch.stack(inputs)
        
        return inputs, target
        #return inputs, target, mask

        """
        return {'from_images': inputs[0],
                'to_images': inputs[1],
                'flow_map': target,
                'correspondence_mask': mask.astype(np.bool) if float(torch.__version__[:3]) >= 1.1
                else mask.astype(np.uint8)
                }
        """

    def __len__(self):
        return len(self.path_list)



        """
    def __init__(self, root, path_list, source_image_transform=None, target_image_transform=None, flow_transform=None,
                 co_transform=None, loader=default_loader, mask=False, size=False):
        """
        """
        :param root: directory containing the dataset images
        :param path_list: list containing the name of images and corresponding ground-truth flow files
        :param source_image_transform: transforms to apply to source images
        :param target_image_transform: transforms to apply to target images
        :param flow_transform: transforms to apply to flow field
        :param co_transform: transforms to apply to both images and the flow field
        :param loader: loader function for the images and the flow
        :param mask: bool indicating is a mask of valid pixels needs to be loaded as well from root
        :param size: size of the original source image
        outputs:
            - source_image
            - target_image
            - flow_map
            - correspondence_mask
            - source_image_size
        """
        """
        self.root = root
        self.path_list = path_list
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.flow_transform = flow_transform
        self.co_transform = co_transform
        self.loader = loader
        self.mask = mask
        self.size = size

    def __getitem__(self, index):
        # for all inputs[0] must be the source and inputs[1] must be the target
        inputs, gt_flow = self.path_list[index]

        if not self.mask:
            if self.size:
                inputs, gt_flow, source_size = self.loader(self.root, inputs, gt_flow)
            else:
                inputs, gt_flow = self.loader(self.root, inputs, gt_flow)
                source_size = inputs[0].shape
            if self.co_transform is not None:
                inputs, gt_flow = self.co_transform(inputs, gt_flow)

            mask = get_gt_correspondence_mask(gt_flow)
        else:
            if self.size:
                inputs, gt_flow, mask, source_size = self.loader(self.root, inputs, gt_flow)
            else:
                # loader comes with a mask of valid correspondences
                inputs, gt_flow, mask = self.loader(self.root, inputs, gt_flow)
                source_size = inputs[0].shape
            # mask is shape hxw
            if self.co_transform is not None:
                inputs, gt_flow, mask = self.co_transform(inputs, gt_flow, mask)

        # here gt_flow has shape HxWx2

        # after co transform that could be reshapping the target
        # transforms here will always contain conversion to tensor (then channel is before)
        if self.transform is not None:
            inputs[0] = self.source_image_transform(inputs[0])
            inputs[1] = self.target_image_transform(inputs[1])
        if self.flow_transform is not None:
            gt_flow = self.flow_transform(gt_flow)

        return {'source_image': inputs[0],
                'target_image': inputs[1],
                'flow_map': gt_flow,
                'correspondence_mask': mask.astype(np.bool) if float(torch.__version__[:3]) >= 1.1
                else mask.astype(np.uint8),
                'source_image_size': source_size
                }
        """