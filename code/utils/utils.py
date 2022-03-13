import random
import torch
import torch.nn.functional as F
import torch.nn as nn
#import torchvision.transforms.functional as TVTF
import numpy as np
import math
#from torchvision import transforms
from PIL import Image
from distutils.version import LooseVersion

def get_patch(*args, patch_size=17, scale=1):
    """
    Get patch from an image for HWC order
    """
    ih, iw, _ = args[0].shape

    ip = patch_size
    tp = scale * ip

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret


def get_patch_frames(*args, patch_size=256, scale=1):
    """
    Get patch from an image for NHWC order
    """
    _, ih, iw, _ = args[0].shape

    ip = patch_size
    tp = scale * ip

    if ip < iw:
        ix = random.randrange(0, iw - ip + 1)
    else:
        ix = 0
    if ip < ih:
        iy = random.randrange(0, ih - ip + 1)
    else:
        iy = 0
    tx, ty = scale * ix, scale * iy

    ret = [
        args[0][:, iy:iy + ip, ix:ix + ip, :],
        args[1][:, ty:ty + tp, tx:tx + tp, :]
    ]

    return ret


def np2Tensor(*args, rgb_range=1., device='cpu'):
    def _np2Tensor(img):
        """Convert Numpy to Tensor but is still on CPU but ready for GPU

        Tried doing the np.transpose(torch.permute) and np.ascontiguousarray(torch.contiguous) as a tensor but it caused
        issues with strides and random errors like below. Plus there is no speed up since this is being done in the CPU.
        I think Pan converted the array to float64 to give it more room in memory for the transpose since it is converted to
        float32 before it is divided by 255.
            CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
        """
        #img = img.astype('float64')
        #np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # NHWC -> NCHW
        #tensor = torch.from_numpy(np_transpose).float()                # numpy -> tensor
        #tensor.mul_(rgb_range / 255.0)                                 # (0,255) -> (0,1)
        img = np.ascontiguousarray(img, dtype=np.float32)  # NHWC -> NCHW
        tensor = torch.from_numpy(img).permute((2, 0, 1))  # numpy -> tensor
        tensor.mul_(rgb_range / 255.0)                     # (0,255) -> (0,1)

        return tensor

    return [_np2Tensor(a) for a in args]

def Tensor2numpy(*args, rgb_range=1.):
    def _Tensor2np(img):
        img = img.mul(255.0 / rgb_range).clamp(0, 255).round().permute((1, 2, 0)) # NCHW -> NHWC
        img = img.cpu().numpy().astype(np.uint8)
        return img
    return [_Tensor2np(a) for a in args]


def data_augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        # Image in HWC order
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = np.rot90(img)

        return img

    return [_augment(a) for a in args]


def data_augment_frames(*args, hflip=True, vflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        # Image in NCHWC order
        if hflip:
            img = np.flip(img, axis=2)
        if vflip:
            img = np.flip(img, axis=1)
        if rot90:
            img = np.rot90(img, axes=(1,2))

        return img

    return [_augment(a) for a in args]


def data_augment_tensors(*args, hflip=True, vflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        # Image in BNCHW order
        #print()
        #print("img.size before: ", img.size())
        if hflip:
            img = torch.flip(img, dims=(4,))
        if vflip:
            img = torch.flip(img, dims=(3,))
        if rot90:
            img = torch.rot90(img, k=1, dims=[3,4])
            #pil_img = transforms.ToPILImage()(img[0][0]).convert("RGB")
            #pil_img.show()
        #print("img.size after: ", img.size())

        return img

    return [_augment(a) for a in args]


def postprocess(*images, rgb_range=1., ycbcr_flag, device):
    def _postprocess(img, rgb_coefficient, ycbcr_flag, device):
        if ycbcr_flag:
            out = img.mul(rgb_coefficient).clamp(16, 235)
        else:
            out = img.mul(rgb_coefficient).clamp(0, 255).round()

        return out

    rgb_coefficient = 255 / rgb_range
    return [_postprocess(img, rgb_coefficient, ycbcr_flag, device) for img in images]


def calc_psnr(img1, img2, rgb_range=1., shave=4):
    mse = torch.mean(torch.sub(img1[:, :, shave:-shave, shave:-shave].div_(rgb_range),
                               img2[:, :, shave:-shave, shave:-shave].div_(rgb_range)).pow_(2))
    if mse == 0:
        return 100
    return 20 * torch.log10(1 / torch.sqrt(mse))


def calc_grad_sobel(img, device='cuda'):
    if not isinstance(img, torch.Tensor):
        raise Exception("Now just support torch.Tensor. See the Type(img)={}".format(type(img)))
    if not img.ndimension() == 4:
        raise Exception("Tensor ndimension must equal to 4. See the img.ndimension={}".format(img.ndimension()))

    img = torch.mean(img, dim=1, keepdim=True)

    # img = calc_meanFilter(img, device=device)  # meanFilter

    sobel_filter_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).reshape((1, 1, 3, 3))
    sobel_filter_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).reshape((1, 1, 3, 3))
    sobel_filter_X = torch.from_numpy(sobel_filter_X).float().to(device)
    sobel_filter_Y = torch.from_numpy(sobel_filter_Y).float().to(device)
    grad_X = F.conv2d(img, sobel_filter_X, bias=None, stride=1, padding=1)
    grad_Y = F.conv2d(img, sobel_filter_Y, bias=None, stride=1, padding=1)
    grad = torch.sqrt(grad_X.pow(2) + grad_Y.pow(2))

    return grad_X, grad_Y, grad


def calc_grad_sobel_torch(img, device='cuda'):
    if not isinstance(img, torch.Tensor):
        raise Exception("Now just support torch.Tensor. See the Type(img)={}".format(type(img)))
    if not img.ndimension() == 4:
        raise Exception("Tensor ndimension must equal to 4. See the img.ndimension={}".format(img.ndimension()))

    img_mean = torch.mean(img, dim=1, keepdim=True)

    sobel_filter_X = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=device).reshape((1, 1, 3, 3)).float()
    sobel_filter_Y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=device).reshape((1, 1, 3, 3)).float()
    grad_X = F.conv2d(img_mean, sobel_filter_X, bias=None, stride=1, padding=1)
    grad_Y = F.conv2d(img_mean, sobel_filter_Y, bias=None, stride=1, padding=1)
    grad = torch.sqrt(grad_X.pow(2) + grad_Y.pow(2))

    return grad_X, grad_Y, grad


def calc_meanFilter(img, kernel_size=11, n_channel=1, device='cuda'):
    mean_filter_X = np.ones(shape=(1, 1, kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    mean_filter_X = torch.from_numpy(mean_filter_X).float().to(device)
    new_img = torch.zeros_like(img)
    for i in range(n_channel):
        new_img[:, i:i + 1, :, :] = F.conv2d(img[:, i:i + 1, :, :], mean_filter_X, bias=None,
                                             stride=1, padding=kernel_size // 2)
    return new_img


def calc_meanFilter_torch(img, kernel_size=11, n_channel=1, device='cuda'):
    mean_filter_X = torch.ones(size   = (1, 1, kernel_size, kernel_size),
                               dtype  = torch.float32,
                               device = device
                              ).div_(kernel_size * kernel_size).float()
    new_img = torch.zeros_like(img, device=device)
    for i in range(n_channel):
        new_img[:, i:i + 1, :, :] = F.conv2d(img[:, i:i + 1, :, :],
                                             mean_filter_X,
                                             bias    = None,
                                             stride  = 1,
                                             padding = kernel_size // 2)
    return new_img


class EPE(nn.Module):
    def __init__(self, device='cuda', mean=True):
        super(EPE, self).__init__()
        self.mean = mean

    def epe(self, input_flow, target_flow):
        """
        End-point-Error computation
        Args:
            input_flow: estimated flow [BxHxW,2]
            target_flow: ground-truth flow [BxHxW,2]
        Output:
            Averaged end-point-error (value)
        """
        if LooseVersion(torch.__version__) < LooseVersion('1.9.0'):
            EPE_loss = torch.norm(target_flow - input_flow, p=2, dim=1)
        else:
            EPE_loss = torch.linalg.vector_norm(target_flow - input_flow, ord=2, dim=1)
        if self.mean:
            EPE_loss = EPE_loss.mean()
        return EPE_loss

    def forward(self, input_flow, target_flow):
        if target_flow.shape[1]==3:
            #print('has mask')
            mask = target_flow[:,2,:,:]
            target_flow = target_flow[:,:2,:,:]
            if not self.training:
                target_flow *= mask[:,None,:,:]
                input_flow *= mask[:,None,:,:]
            epe_loss = self.epe(input_flow, target_flow)
        else:
            epe_loss = self.epe(input_flow, target_flow)

        return epe_loss


class Fl_KITTI_2015(nn.Module):
    def __init__(self, device='cuda', tau=[3.0, 0.05], use_mask=False):
        super(Fl_KITTI_2015, self).__init__()
        self.tau = tau
        self.use_mask=use_mask
        self.device=device

    def fl_kitti_2015(self, input_flow, target_flow, mask=None):
        """
        Computation number of outliers
        for which error > 3px(tau[0]) and error/magnitude(ground truth flow) > 0.05(tau[1])
        Args:
            input_flow: estimated flow [BxHxW,2]
            target_flow: ground-truth flow [BxHxW,2]
        Output:
            PCK metric
        """
        # input flow is shape (BxHgtxWgt,2)
        if target_flow.shape[1]==3:
            mask = target_flow[:,2,:,:]
            target_flow = target_flow[:,:2,:,:]
        if self.use_mask:
            if mask is None:
                raise ValueError("Mask must be specified if part of the Target")
            if not self.training:
                target_flow *= mask[:,None,:,:]
                input_flow *= mask[:,None,:,:]

        if LooseVersion(torch.__version__) < LooseVersion('1.9.0'):
            dist = torch.norm(target_flow - input_flow, p=2, dim=1)
            gt_magnitude = torch.norm(target_flow, p=2, dim=1)
        else:
            dist = torch.linalg.vector_norm(target_flow - input_flow, ord=2, dim=1)
            gt_magnitude = torch.linalg.vector_norm(target_flow, ord=2, dim=1)
        # dist is shape BxHgtxWgt
        n_err = dist.gt(self.tau[0]) & (dist/gt_magnitude).gt(self.tau[1])
        if self.use_mask:
            fl_loss = n_err.sum()/torch.sum(mask)
        else:
            fl_loss = n_err.sum()
        return fl_loss

    def forward(self, input_flow, target_flow, mask=None):
        fl_loss = self.fl_kitti_2015(input_flow, target_flow, mask)

        return fl_loss
