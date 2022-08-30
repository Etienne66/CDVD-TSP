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
                              ).div_(kernel_size * kernel_size)
    new_img = torch.zeros_like(img, device=device)
    for i in range(n_channel):
        new_img[:, i:i + 1, :, :] = F.conv2d(img[:, i:i + 1, :, :],
                                             mean_filter_X,
                                             bias    = None,
                                             stride  = 1,
                                             padding = kernel_size // 2)
    return new_img


class EPE(nn.Module):
    def __init__(self, ord=2, device='cuda', mean=True):
        super(EPE, self).__init__()
        self.mean = mean
        self.ord = ord

    def epe(self, input_flow, target_flow):
        """
        End-point-Error computation
        Args:
            input_flow: estimated flow [BxHxW,2]
            target_flow: ground-truth flow [BxHxW,2]
        Output:
            Averaged end-point-error (value)
        
        L1-Norm is "the sum of the absolute vector values, where the absolute value of a scalar uses the notation |a1|. In effect,
                    the norm is a calculation of the Manhattan distance from the origin of the vector space."
        >>> x.norm(dim=1, p=1)
        >>> torch.linalg.norm(x, dim=1, ord=1)
        >>> x.abs().sum(dim=1)
        
        L2-Norm is "the distance of the vector coordinate from the origin of the vector space. The L2 norm is calculated as the 
                    square root of the sum of the squared vector values."
        >>> x.norm(dim=1, p=2)
        >>> torch.linalg.norm(x, dim=1, ord=2)
        >>> x.pow(2).sum(dim=1).sqrt()
        """
        if LooseVersion(torch.__version__) < LooseVersion('1.9.0'):
            EPE_loss = torch.norm(target_flow - input_flow, p=self.ord, dim=1)
        else:
            EPE_loss = torch.linalg.vector_norm(target_flow - input_flow, ord=self.ord, dim=1)
        
        batch_size = EPE_loss.size(0)
        if self.mean:
            EPE_loss = EPE_loss.mean()
        else:
            EPE_loss = EPE_loss.sum()/batch_size
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

        #2-Norm is "the distance of the vector coordinate from the origin of the vector space. The L2 norm is calculated as the 
        #           square root of the sum of the squared vector values."
        #>>> x.norm(dim=1, p=2)
        #>>> torch.linalg.norm(x, dim=1, ord=2)
        #>>> x.pow(2).sum(dim=1).sqrt()
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


class multiscaleLoss(nn.Module):
    def __init__(self, device='cuda', loss='EPE', split_losses=True, weights=None):
        super(multiscaleLoss, self).__init__()
        self.device=device
        self.weights = weights
        self.loss = loss
        self.split_losses = split_losses
        self.realWAUC = WAUC()
        self.realEPE = EPE(mean=True)
        self.realEPE1 = EPE(ord=1)
        self.L1 = EPE(ord=1,mean=False)
        self.L2 = EPE(mean=False)

        
    def one_scale(self, output, target):
        b, _, h, w = output.size()
        _, _, th, tw = target.size()
        
        if h == th and w == tw:
            target_scaled = target
        else:
            # area interpolate is deterministic
            target_scaled = F.interpolate(target, (h, w), mode='bicubic')
            # Vectors need to be scaled down
            #target_scaled[:, 0, :, :] *= float(w) / float(tw)
            #target_scaled[:, 1, :, :] *= float(h) / float(th)
        
        if self.loss == 'WAUCl':
            loss = 100 - self.realWAUC(output, target_scaled)
        elif self.loss == 'EPE':
            loss = self.realEPE(output, target_scaled)
        elif self.loss == 'EPE1':
            loss = self.realEPE1(output, target_scaled)
        elif self.loss == 'L1':
            loss = self.L1(output, target_scaled)
        elif self.loss == 'L2':
            loss = self.L2(output, target_scaled)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found'.format(self.loss))
        return loss


    def forward(self, network_output, target_flow):
        #print('type network_output',type(network_output))
        if type(network_output) not in [tuple, list]:
            network_output = [network_output]
        #if self.weights is None:
        #    weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
        #assert(len(weights) == len(network_output))

        loss = 0
        if self.split_losses and self.loss in ('WAUCl','EPE','EPE1'):
            if self.weights is None:
                if len(network_output) == 6:
                    self.weights = [1, 0.5, 0.25, 0.25, 0.25, 0.25]  # as in original article but converted for mean loss
                else:
                    self.weights = [1, 0.5, 0.25, 0.25, 0.25]  # as in original article but converted for mean loss
                #weights = [1, 1, 1, 1, 1]  # as in original article but converted for mean loss
            assert(len(self.weights) == len(network_output))
            losses=[]
            for output, weight in zip(network_output, self.weights):
                if output is not None:
                    currentLoss = self.one_scale(output, target_flow)
                    losses.append(currentLoss)
                    loss += weight * currentLoss
                else:
                    losses.append(None)
            return loss, losses
        else:
            if self.weights is None:
                #These weights are based on mean=False
                self.weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
            assert(len(self.weights) == len(network_output))
            for output, weight in zip(network_output, self.weights):
                if output is not None:
                    currentLoss = self.one_scale(output, target_flow)
                    loss += weight * currentLoss
                else:
                    losses.append(None)
            return loss


class WAUC(nn.Module):
    def __init__(self, device='cuda'):
        super(WAUC, self).__init__()
        self.device=device

    def Average_Endpoint_Error(self, input_flow, target_flow, mask=None):
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
        if mask is None:
            raise ValueError("Mask must be specified if part of the Target")
        if not self.training:
            target_flow *= mask[:,None,:,:]
            input_flow *= mask[:,None,:,:]

        if LooseVersion(torch.__version__) < LooseVersion('1.9.0'):
            dist = torch.norm(target_flow - input_flow, p=2, dim=1)
        else:
            dist = torch.linalg.vector_norm(target_flow - input_flow, ord=2, dim=1)
        n_err = 0
        n_weights = 0

        for i in range(100):
            weight = (1 - ((i-1)/100))
            n_err += dist.le(i/20).sum() * weight
            n_weights += weight
        
        wauc_total = (100 * n_err) / (mask.sum() * n_weights)

        return wauc_total

    def forward(self, input_flow, target_flow, mask=None):
        fl_loss = self.Average_Endpoint_Error(input_flow, target_flow, mask)

        return fl_loss
