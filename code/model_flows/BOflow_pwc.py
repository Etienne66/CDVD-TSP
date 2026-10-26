import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import math
from utils import utils
import model_flows.correlation2 as correlation2
from model import correlation
import torch.utils.checkpoint as checkpoint
import inspect

from distutils.version import LooseVersion

def make_model(args):
    """ This does not appear to be used """
    device = 'cpu' if args.cpu else 'cuda'
    return BOFlow_PWC(device         = device,
                     use_checkpoint = args.use_checkpoint)


class BOFlow_PWC(nn.Module):
    """Derivative of `run.py` from
    [A reimplementation of PWC-Net in PyTorch that matches the official Caffe version](https://github.com/sniklaus/pytorch-pwc)
    
    """
    def __init__(self,
                 device         = 'cuda',
                 use_checkpoint = False):
        super(BOFlow_PWC, self).__init__()
        self.device = device
        self.use_checkpoint = use_checkpoint
        self.moduleNetwork = Network(device=device, use_checkpoint=use_checkpoint)
        print("Creating Flow PWC")


    def estimate_flow(self, tensorFirst, tensorSecond):
        b, c, intHeight, intWidth = tensorFirst.size()

        # Need input image resolution to always be a multiple of 64
        intPreprocessedWidth = int(math.ceil(intWidth / 64.0) * 64)
        intPreprocessedHeight = int(math.ceil(intHeight / 64.0) * 64)

        if True:
        #if intPreprocessedWidth == intWidth and intPreprocessedHeight == intHeight:
            # Faster but same memory utilization. Without detach it is slower but takes less memory.
            tensorPreprocessedFirst = tensorFirst.detach()
            tensorPreprocessedSecond = tensorSecond.detach()
        else:
            tensorPreprocessedFirst = nn.functional.interpolate(
                                        input         = tensorFirst,
                                        size          = (intPreprocessedHeight, intPreprocessedWidth),
                                        mode          = 'bilinear',
                                        align_corners = False,
                                        antialias     = True)
            tensorPreprocessedSecond = nn.functional.interpolate(
                                        input         = tensorSecond,
                                        size          = (intPreprocessedHeight, intPreprocessedWidth),
                                        mode          = 'bilinear',
                                        align_corners = False,
                                        antialias     = True)

        # The flow returned by moduleNetwork is always one quarter the resolution of the original and is upscaled to the original
        # resolution
        """
        tensorFlow = nn.functional.interpolate(
                        input         = self.moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond),
                        size          = (intHeight, intWidth),
                        mode          = 'bilinear',
                        align_corners = False)
        """
        
        tensorFlow = self.moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond)

        #tensorFlow[:, 0, :, :] *= 20.0 * float(intWidth) / float(intPreprocessedWidth)
        #tensorFlow[:, 1, :, :] *= 20.0 * float(intHeight) / float(intPreprocessedHeight)
        tensorFlow[:, 0, :, :] *= 20.0
        tensorFlow[:, 1, :, :] *= 20.0

        return tensorFlow

    def warp(self, x, flo):
        """Not part of `run.py` in pytorch-pwc
        warp an image/tensor (im2) back to im1, according to the optical flow
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow
            output: [B, C, H, W] (im1)
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W, device=self.device, dtype=torch.float32, requires_grad=True).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=self.device, dtype=torch.float32, requires_grad=True).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).add_(flo)

        # scale grid to [-1,1]
        grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        grid = grid.permute(0, 2, 3, 1) # B2HW -> BHW2
        # Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.
        # Code was developed for 0.4.1
        output = nn.functional.grid_sample(x,
                                           grid,
                                           mode='bilinear',
                                           padding_mode  = 'border',
                                           align_corners = True)
        mask = torch.ones_like(x,
                               device        = self.device,
                               requires_grad = True)
        mask = nn.functional.grid_sample(mask,
                                         grid,
                                         mode='bilinear',
                                         padding_mode  = 'border',
                                         align_corners = True)

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        output *= mask

        return output, mask


    def forward(self, frame_1, frame_2):
        """Not part of `run.py` in pytorch-pwc
        - Find the flow between Frame 1 and Frame 2
        - Warp Frame 2 to the same position as Frame 1
        """
        # flow
        flow = self.estimate_flow(frame_1, frame_2)

        # warp
        frame_2_warp, mask = self.warp(frame_2, flow)

        return frame_2_warp, flow, mask
    # end




##########################################################
def Backwarp(tensorInput, tensorFlow, requires_grad=False):
    """Warping layer
    Duplicate function from `run.py` from
    [A reimplementation of PWC-Net in PyTorch that matches the official Caffe version](https://github.com/sniklaus/pytorch-pwc)
    
    Need to replace with https://nonint.com/2020/09/17/accelerated-differentiable-image-warping-in-pytorch/
    Also refer to https://github.com/NVIDIA/flownet2-pytorch
    
    """
    tensorHorizontal = torch.linspace(-1.0,# + (1.0 / tensorFlow.shape[3]),
                                      1.0,# - (1.0 / tensorFlow.shape[3]),
                                      tensorFlow.shape[3],
                                      dtype=torch.float32,
                                      device=tensorFlow.device,
                                      requires_grad=requires_grad
                                     ).view(1,
                                            1,
                                            1,
                                            -1
                                           ).repeat(1,
                                                    1,
                                                    tensorFlow.shape[2],
                                                    1)
    tensorVertical = torch.linspace(-1.0,# + (1.0 / tensorFlow.shape[2]),
                                    1.0,# - (1.0 / tensorFlow.shape[2]),
                                    tensorFlow.shape[2],
                                    dtype=torch.float32,
                                    device=tensorFlow.device,
                                    requires_grad=requires_grad
                                   ).view(1,
                                          1,
                                          -1,
                                          1
                                         ).repeat(1,
                                                  1,
                                                  1,
                                                  tensorFlow.shape[3])
    Backwarp_tensorGrid = torch.cat([tensorHorizontal, tensorVertical], 1)
    #This is used to make the mask. Will be same device and dtype as tensorFlow
    Backwarp_tensorPartial = tensorFlow.new_ones([tensorFlow.shape[0],
                                                  1,
                                                  tensorFlow.shape[2],
                                                  tensorFlow.shape[3]],
                                                 dtype=torch.float32,
                                                 requires_grad=requires_grad)
    #tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.shape[3] - 1.0) / 2.0),
    #                        tensorFlow[:, 1:2, :, :] / ((tensorInput.shape[2] - 1.0) / 2.0)], 1)
    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :].float() * (2.0/(tensorInput.shape[3] - 1.0)),
                            tensorFlow[:, 1:2, :, :].float() * (2.0/(tensorInput.shape[2] - 1.0))], 1)
    # Add the mask as the last channel of the input
    tensorInput = torch.cat([tensorInput.float(), Backwarp_tensorPartial], 1)

    #Move to CPU to make it deterministic
    #Changed padding_mode from zeros to border because we shouldn't assume that the motion stops at the border
    tensorOutput = nn.functional.grid_sample(input         = tensorInput,
                                             grid          = (Backwarp_tensorGrid + tensorFlow).permute(0, 2, 3, 1),
                                             mode          = 'bilinear',
                                             padding_mode  = 'zeros',
                                             align_corners = True
                                             #align_corners = True
                                             )
    # Extract the mask from the last channel of the output.
    # The mask is to filter out the parts of the image that a flow can't be calculated for. Normally around the edges of the image.
    tensorMask = tensorOutput[:, -1:, :, :].nan_to_num();
    tensorMask[tensorMask < 0.9999] = 0;
    tensorMask[tensorMask > 0] = 1
    # Filter the mask from the tensor output and apply the mask to the remaining channels.
    return tensorOutput[:, :-1, :, :] * tensorMask
# end


class Extractor(nn.Module):
    """The feature pyramid extractor network. The first image (t = 1) and the second image (t =2) are encoded using the same
    Siamese network. Each convolution is followed by a leaky ReLU unit. The convolutional layer and the ×2 downsampling layer at
    each level is implemented using a single convolutional layer with a stride of 2. c denotes extracted features of image t at
    level l
    Using Feature ↑ with 3 layers instead of 2 as the paper says
    Paper says it is using 7 level pyramids but 6 layers are used and appears to be what was used in the paper.
    """
    def __init__(self, args):
        super(Extractor, self).__init__()
        self.useFlow1 = args.useFlow1

        if args.useOrginalModel:
            self.moduleOne = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1)
            )
            self.moduleTwo = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1)
            )
            self.moduleThr = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1)
            )
            self.moduleFou = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=m4Chan, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv2d(in_channels=m4Chan, out_channels=m4Chan, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv2d(in_channels=m4Chan, out_channels=m4Chan, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1)
            )
            self.moduleFiv = nn.Sequential(
                nn.Conv2d(in_channels=m4Chan, out_channels=m5Chan, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv2d(in_channels=m5Chan, out_channels=m5Chan, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv2d(in_channels=m5Chan, out_channels=m5Chan, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1)
            )
            # Author intended 192(32*6) but had used 196. 160(32*5) makes more sense since the prior levels were multiples of 32
            # 32*0.5, 32*1, 32*2, 32*3, 32*4
            self.moduleSix = nn.Sequential(
                nn.Conv2d(in_channels=m5Chan, out_channels=m6Chan, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv2d(in_channels=m6Chan, out_channels=m6Chan, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Conv2d(in_channels=m6Chan, out_channels=m6Chan, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1)
            )
        else:
            TempList = [
                nn.Conv2d(in_channels=3, out_channels=args.EC[0], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(args.EC[0]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[0],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[0]) if args.batchnorm2 else None,
                nn.Conv2d(in_channels=args.EC[0], out_channels=args.EC[0], kernel_size=3, stride=1, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[0]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[0],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[0]) if args.batchnorm2 else None,
                nn.Conv2d(in_channels=args.EC[0], out_channels=args.EC[0], kernel_size=3, stride=1, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[0]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[0],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[0]) if args.batchnorm2 else None,
                nn.Conv2d(in_channels=args.EC[0], out_channels=args.EC[0], kernel_size=3, stride=1, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[0]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[0],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[0]) if args.batchnorm2 else None
            ]
            TempListClean = [x for x in TempList if x is not None] #Remove blank items
            self.moduleOne = nn.Sequential(*TempListClean)
            TempList = (
                nn.Conv2d(in_channels=args.EC[0], out_channels=args.EC[1], kernel_size=3, stride=2, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[1]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[1],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[1]) if args.batchnorm2 else None,
                nn.Conv2d(in_channels=args.EC[1], out_channels=args.EC[1], kernel_size=3, stride=1, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[1]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[1],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[1]) if args.batchnorm2 else None,
                nn.Conv2d(in_channels=args.EC[1], out_channels=args.EC[1], kernel_size=3, stride=1, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[1]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[1],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[1]) if args.batchnorm2 else None,
                nn.Conv2d(in_channels=args.EC[1], out_channels=args.EC[1], kernel_size=3, stride=1, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[1]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[1],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[1]) if args.batchnorm2 else None
            )
            TempListClean = [x for x in TempList if x is not None] #Remove blank items
            self.moduleTwo = nn.Sequential(*TempListClean)
            TempList = (
                nn.Conv2d(in_channels=args.EC[1], out_channels=args.EC[2], kernel_size=3, stride=2, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[2]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[2],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[2]) if args.batchnorm2 else None,
                nn.Conv2d(in_channels=args.EC[2], out_channels=args.EC[2], kernel_size=3, stride=1, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[2]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[2],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[2]) if args.batchnorm2 else None,
                nn.Conv2d(in_channels=args.EC[2], out_channels=args.EC[2], kernel_size=3, stride=1, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[2]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[2],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[2]) if args.batchnorm2 else None,
                nn.Conv2d(in_channels=args.EC[2], out_channels=args.EC[2], kernel_size=3, stride=1, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[2]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[2],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[2]) if args.batchnorm2 else None
            )
            TempListClean = [x for x in TempList if x is not None] #Remove blank items
            self.moduleThr = nn.Sequential(*TempListClean)
            TempList = (
                nn.Conv2d(in_channels=args.EC[2], out_channels=args.EC[3], kernel_size=3, stride=2, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[3]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[3],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[3]) if args.batchnorm2 else None,
                nn.Conv2d(in_channels=args.EC[3], out_channels=args.EC[3], kernel_size=3, stride=1, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[3]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[3],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[3]) if args.batchnorm2 else None,
                nn.Conv2d(in_channels=args.EC[3], out_channels=args.EC[3], kernel_size=3, stride=1, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[3]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[3],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[3]) if args.batchnorm2 else None,
                nn.Conv2d(in_channels=args.EC[3], out_channels=args.EC[3], kernel_size=3, stride=1, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[3]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[3],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[3]) if args.batchnorm2 else None
            )
            TempListClean = [x for x in TempList if x is not None] #Remove blank items
            self.moduleFou = nn.Sequential(*TempListClean)
            TempList = (
                nn.Conv2d(in_channels=args.EC[3], out_channels=args.EC[4], kernel_size=3, stride=2, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[4]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[4],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[4]) if args.batchnorm2 else None,
                nn.Conv2d(in_channels=args.EC[4], out_channels=args.EC[4], kernel_size=3, stride=1, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[4]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[4],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[4]) if args.batchnorm2 else None,
                nn.Conv2d(in_channels=args.EC[4], out_channels=args.EC[4], kernel_size=3, stride=1, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[4]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[4],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[4]) if args.batchnorm2 else None,
                nn.Conv2d(in_channels=args.EC[4], out_channels=args.EC[4], kernel_size=3, stride=1, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[4]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[4],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[4]) if args.batchnorm2 else None
            )
            TempListClean = [x for x in TempList if x is not None] #Remove blank items
            self.moduleFiv = nn.Sequential(*TempListClean)
            # Author intended 192(32*6) but had used 196. 160(32*5) makes more sense since the prior levels were multiples of 32
            # 32*0.5, 32*1, 32*2, 32*3, 32*4
            TempList = (
                nn.Conv2d(in_channels=args.EC[4], out_channels=args.EC[5], kernel_size=3, stride=2, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[5]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[5],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[5]) if args.batchnorm2 else None,
                nn.Conv2d(in_channels=args.EC[5], out_channels=args.EC[5], kernel_size=3, stride=1, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[5]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[5],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[5]) if args.batchnorm2 else None,
                nn.Conv2d(in_channels=args.EC[5], out_channels=args.EC[5], kernel_size=3, stride=1, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[5]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[5],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[5]) if args.batchnorm2 else None,
                nn.Conv2d(in_channels=args.EC[5], out_channels=args.EC[5], kernel_size=3, stride=1, padding=1, bias=False if args.batchnorm else True),
                nn.BatchNorm2d(args.EC[5]) if args.batchnorm else None,
                nn.PReLU(num_parameters=args.EC[5],init=0.1) if args.usePReLu else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.BatchNorm2d(args.EC[5]) if args.batchnorm2 else None
            )
            TempListClean = [x for x in TempList if x is not None] #Remove blank items
            self.moduleSix = nn.Sequential(*TempListClean)
        
        # Original Caffe model used MSRA. This is also known as Kaiming
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0) 
        
    # end

    def forward(self, tensorInput):
        tensorOne = self.moduleOne(tensorInput)
        tensorTwo = self.moduleTwo(tensorOne)
        tensorThr = self.moduleThr(tensorTwo)
        tensorFou = self.moduleFou(tensorThr)
        tensorFiv = self.moduleFiv(tensorFou)
        tensorSix = self.moduleSix(tensorFiv)

        # Removed tensorOne from return because it wasn't used
        if self.useFlow1:
            return [tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix]
        else:
            return [tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix]
    # end
# end

class Decoder(nn.Module):
    """The optical flow estimator network. Each convolutional layer is followed by a leaky ReLU unit except the last one that
    outputs the optical flow.
    Paper indicates that 5 layers are used but appears to be 6 layers
    """
    def __init__(self,
                 intLevel,
                 args):
        super(Decoder, self).__init__()
        self.device=args.device
        self.div_flow = args.div_flow
        self.intLevel = intLevel
        self.useResidual = args.useResidual
        self.FullScale = args.FullScale
        self.useDenseNet = args.useDenseNet
        self.scale_supervision = args.scale_supervision
        self.max_displacement = args.max_displacement[intLevel - 1]
        correlation_channels_previous = (args.max_displacement[intLevel - 2] * 2 + 1) ** 2;
        correlation_channels = (args.max_displacement[intLevel - 1] * 2 + 1) ** 2;
        self.correlation_channels = correlation_channels
        xD = self.xD = args.DenseNetSize

        if args.useResidual:
            residualChannels = 2
        else:
            residualChannels = 0

        m1Chan = 128
        intChannels = [None,
                       correlation_channels_previous * 2 + args.EC[0] + 2 * 2 + residualChannels,
                       correlation_channels_previous * 2 + args.EC[1] + 2 * 2 + residualChannels, 
                       correlation_channels_previous * 2 + args.EC[2] + 2 * 2 + residualChannels,
                       correlation_channels_previous * 2 + args.EC[3] + 2 * 2 + residualChannels,
                       correlation_channels_previous * 2 + args.EC[4] + 2 * 2 + residualChannels,
                       correlation_channels_previous * 2 + args.EC[5],
                       None]
        intPrevious = intChannels[intLevel + 1]
        intChannels = [None,
                       correlation_channels * 2 + args.EC[0] + 2 * 2 + residualChannels,
                       correlation_channels * 2 + args.EC[1] + 2 * 2 + residualChannels, 
                       correlation_channels * 2 + args.EC[2] + 2 * 2 + residualChannels,
                       correlation_channels * 2 + args.EC[3] + 2 * 2 + residualChannels,
                       correlation_channels * 2 + args.EC[4] + 2 * 2 + residualChannels,
                       correlation_channels * 2 + args.EC[5],
                       None]
        intCurrent  = intChannels[intLevel + 0]

        if intLevel < 6:
            if args.useOrginalModel:
                self.moduleUpflow = nn.ConvTranspose2d(in_channels  = 2,
                                                       out_channels = 2,
                                                       kernel_size  = 4,
                                                       stride       = 2, # Double resolution
                                                       padding      = 1)
            else:
                self.moduleUpflow = nn.Sequential(nn.PReLU(num_parameters=2,init=0.1),
                                                  nn.ConvTranspose2d(in_channels  = 2,
                                                                     out_channels = 2,
                                                                     kernel_size  = 4,
                                                                     stride       = 2, # Double resolution
                                                                     padding      = 1))
            if args.useResidual:
                if args.useDenseNet:
                    if args.useOrginalModel:
                        self.moduleUpfeat = nn.ConvTranspose2d(in_channels  = intPrevious + m1Chan + 128 + 96 + 64 + 32,
                                                            out_channels = 2,
                                                            kernel_size  = 4,
                                                            stride       = 2, # Double resolution
                                                            padding      = 1)
                    else:
                        self.moduleUpfeat = nn.Sequential(nn.ConvTranspose2d(in_channels  = 16**xD * intPrevious + 8**xD * m1Chan + 4**xD * 128 + 2**xD * 96 + 64 + 32,
                                                                             out_channels = 2,
                                                                             kernel_size  = 4,
                                                                             stride       = 2, # Double resolution
                                                                             padding      = 1),
                                                          nn.PReLU(num_parameters=2,init=0.1))
                else:
                    if args.useOrginalModel:
                        self.moduleUpfeat = nn.ConvTranspose2d(in_channels  = intPrevious,
                                                               out_channels = 2,
                                                               kernel_size  = 4,
                                                               stride       = 2, # Double resolution
                                                               padding      = 1)
                    else:
                        self.moduleUpfeat = nn.Sequential(nn.ConvTranspose2d(in_channels  = 32,
                                                                             out_channels = 2,
                                                                             kernel_size  = 4,
                                                                             stride       = 2, # Double resolution
                                                                             padding      = 1),
                                                          nn.PReLU(num_parameters=2,init=0.1))
            # Start with inLevel 6(None), 5(20/32=0.625), 4(20/16=1.25), 3(20/8=2.5) and finally 2(20/4=5.0)
            # 6 doesn't have an objectPrevious so doesn't need a value. 2 is the last.
            # Add 1(20/2=10.0)
            #self.dblBackwarp = [None, None, 10.0, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]
            if self.scale_supervision:
                self.dblBackwarp = 1
            else:
                self.dblBackwarp = [None,
                                    self.div_flow/1,
                                    self.div_flow/2,
                                    self.div_flow/4,
                                    self.div_flow/8,
                                    self.div_flow/16,
                                    self.div_flow/32,
                                    self.div_flow/64][intLevel + 1]

        
        
        
        self.leakyrelu = nn.PReLU(num_parameters=2,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1)
        #print('self.max_displacement',self.max_displacement)
        #print('correlation_channels',correlation_channels)
        self.correlation = nn.Sequential(
            correlation2.ModuleCorrelation(padding = self.max_displacement,
                                           kernel_size = 1,
                                           max_displacement = self.max_displacement,
                                           stride1 = 1,
                                           stride2 = 1,
                                           device = args.device),
            nn.PReLU(num_parameters=correlation_channels,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
        self.moduleOne = nn.Sequential(
            nn.Conv2d(in_channels  = intCurrent,
                      out_channels = m1Chan,
                      kernel_size  = 3,
                      stride       = 1,
                      padding      = 1),
            nn.PReLU(num_parameters=m1Chan,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
            #nn.Conv2d(in_channels = m1Chan, out_channels = m1Chan, kernel_size = 3, stride = 1, padding = 1),
            #nn.PReLU(num_parameters=m1Chan,init=0.1)
        )
        self.moduleTwo = nn.Sequential(
            nn.Conv2d(in_channels  = intCurrent + m1Chan,
                      out_channels = 128,
                      kernel_size  = 3,
                      stride       = 1,
                      padding      = 1),
            nn.PReLU(num_parameters=128,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
            #nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            #nn.PReLU(num_parameters=128,init=0.1)
        )
        self.moduleThr = nn.Sequential(
            nn.Conv2d(in_channels  = 2**xD * intCurrent + m1Chan + 128,
                      out_channels = 96,
                      kernel_size  = 3,
                      stride       = 1,
                      padding      = 1),
            nn.PReLU(num_parameters=96,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
            #nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, stride = 1, padding = 1),
            #nn.PReLU(num_parameters=96,init=0.1)
        )
        self.moduleFou = nn.Sequential(
            nn.Conv2d(in_channels  = 4**xD * intCurrent + 2**xD * m1Chan + 128 + 96,
                      out_channels = 64,
                      kernel_size  = 3,
                      stride       = 1,
                      padding      = 1),
            nn.PReLU(num_parameters=64,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
            #nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            #nn.PReLU(num_parameters=64,init=0.1)
        )
        self.moduleFiv = nn.Sequential(
            nn.Conv2d(in_channels  = 8**xD * intCurrent + 4**xD * m1Chan + 2**xD * 128 + 96 + 64,
                      out_channels = 32,
                      kernel_size  = 3,
                      stride       = 1,
                      padding      = 1),
            nn.PReLU(num_parameters=32,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
            #nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            #nn.PReLU(num_parameters=32,init=0.1)
        )
        if args.FullScale and intLevel == 2:
            self.moduleSix = nn.Sequential(
                nn.Conv2d(in_channels  = 16**xD * intCurrent + 8**xD * m1Chan + 4**xD * 128 + 2**xD * 96 + 64 + 32,
                          out_channels = 128,
                          kernel_size  = 3,
                          stride       = 1,
                          padding      = 1),
                nn.PReLU(num_parameters=128,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
                nn.PReLU(num_parameters=128,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
                nn.PReLU(num_parameters=128,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size = 4, stride = 2, padding = 1), # Double resolution
                nn.PReLU(num_parameters=128,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
                nn.PReLU(num_parameters=64,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
                nn.PReLU(num_parameters=64,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 4, stride = 2, padding = 1), # Double resolution
                nn.PReLU(num_parameters=64,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
                nn.PReLU(num_parameters=32,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
                nn.PReLU(num_parameters=32,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels = 32, out_channels = 4, kernel_size = 3, stride = 1, padding = 1)

                #nn.ConvTranspose2d(in_channels  = 16**xD * intCurrent + 8**xD * m1Chan + 4**xD * 128 + 2**xD * 96 + 64 + 32,
                #                   out_channels = 64,
                #                   kernel_size  = 4,
                #                   stride       = 2,
                #                   padding      = 1),
                #nn.PReLU(num_parameters=64,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                #nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
                #nn.PReLU(num_parameters=64,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                #nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 4, stride = 2, padding = 1), # Double resolution
                #nn.PReLU(num_parameters=32,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                #nn.Conv2d(in_channels = 32, out_channels = 4, kernel_size = 3, stride = 1, padding = 1)
            )
        else:
            self.moduleSix = nn.Sequential(
                nn.Conv2d(in_channels  = 16**xD * intCurrent + 8**xD * m1Chan + 4**xD * 128 + 2**xD * 96 + 64 + 32,
                          out_channels = 64,
                          kernel_size  = 3,
                          stride       = 1,
                          padding      = 1),
                nn.PReLU(num_parameters=64,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
                nn.PReLU(num_parameters=64,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 4, stride = 2, padding = 1), # Double resolution
                nn.PReLU(num_parameters=64,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
                nn.PReLU(num_parameters=32,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
                nn.PReLU(num_parameters=32,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels = 32, out_channels = 4, kernel_size = 3, stride = 1, padding = 1)

                #nn.ConvTranspose2d(in_channels  = 16**xD * intCurrent + 8**xD * m1Chan + 4**xD * 128 + 2**xD * 96 + 64 + 32,
                #                   out_channels = 32,
                #                   kernel_size  = 4,
                #                   stride       = 2,
                #                   padding      = 1),
                #nn.PReLU(num_parameters=32,init=0.1) if args.usePReLuD else nn.LeakyReLU(inplace=False, negative_slope=0.1),
                #nn.Conv2d(in_channels = 32, out_channels = 4, kernel_size = 3, stride = 1, padding = 1)
            )

        # Original Caffe model used MSRA. This is also known as Kaiming
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0) 

    # end

    def forward(self, tensorBackward, tensorCenter, tensorForward, objectPrevious):
        tensorFlow = None

        if objectPrevious is None:
            tensorFlow = None
            tensorFeat0 = None

            #Cost volume layer. Output is always 81 channels
            tensorVolumeBackward = self.correlation(torch.stack((tensorCenter, tensorBackward), dim=1))
            tensorVolumeForward = self.correlation(torch.stack((tensorCenter, tensorForward), dim=1))
            tensorFeat0 = torch.cat([tensorVolumeBackward, tensorCenter, tensorVolumeForward], 1)

        elif objectPrevious is not None:
            tensorForwardFlow = objectPrevious['tensorForwardFlow'] # 2 channels
            tensorBackwardFlow = objectPrevious['tensorBackwardFlow'] # 2 channels
            # Warping layer of Flow Increment
            tensorBackwardWarp = Backwarp(tensorInput   = tensorBackward,
                                          tensorFlow    = tensorBackwardFlow.nan_to_num() * self.dblBackwarp, #self.dblBackwarp,self.div_flow
                                          requires_grad = self.training)
            tensorForwardWarp = Backwarp(tensorInput   = tensorForward,
                                         tensorFlow    = tensorForwardFlow.nan_to_num() * self.dblBackwarp, #self.dblBackwarp,self.div_flow
                                         requires_grad = self.training)

            #Cost volume layer. Output is always 81 channels
            tensorVolumeBackward = self.correlation(torch.stack((tensorCenter, tensorBackwardWarp), dim=1))
            tensorVolumeForward = self.correlation(torch.stack((tensorCenter, tensorForwardWarp), dim=1))
            tensorFeat0 = torch.cat([tensorVolumeBackward,
                                     self.leakyrelu(tensorBackwardFlow),
                                     tensorCenter,
                                     self.leakyrelu(tensorForwardFlow),
                                     tensorVolumeForward], 1)
        # end

        # Concatenating the previous features with the current features is a DenseNet connection
        # This is the CNN
        
        tensorFeat1 = torch.cat([self.moduleOne(tensorFeat0), tensorFeat0], 1)
        tensorFeat2 = torch.cat([self.moduleTwo(tensorFeat1), tensorFeat1], 1)
        tensorFeat3 = torch.cat([self.moduleThr(tensorFeat2), tensorFeat2], 1)
        tensorFeat4 = torch.cat([self.moduleFou(tensorFeat3), tensorFeat3], 1)
        tensorFeat = torch.cat([self.moduleFiv(tensorFeat4), tensorFeat4], 1)

        tensorFlow = torch.chunk(self.moduleSix(tensorFeat),chunks=2, dim=1)
        
        return {'tensorBackwardFlow': tensorFlow[0],
                'tensorForwardFlow':  tensorFlow[1],
                'tensorFeat': tensorFeat
               }
    # end
# end

class Refiner(nn.Module):
    """The context network. Each convolutional layer is followed by a leaky ReLU unit except the last one that outputs the
    optical flow. The last number in each convolutional layer denotes the dilation constant.
    """
    def __init__(self, args):
        super(Refiner, self).__init__()
        correlation_channels = (args.max_displacement[1] * 2 + 1) ** 2;
        xD = args.DenseNetSize
        self.scale_supervision = args.scale_supervision
        self.FullScale = args.FullScale
        print('correlation_channels:', correlation_channels)
        
        flowChannels = 0
        if args.useResidual:
            residualChannels = 2
        else:
            residualChannels = 0
            
        #features
        #feature_init_channels = 81 + 64 + 2 + 2
        #feature_cnn_channels = 128 + 128 + 96 + 64 + 32
        if (args.large_motion or args.full_motion) and args.useDenseNet and args.useFlow1:
            feature_init_channels = 16**xD * (correlation_channels + 16 + 2 + residualChannels)
            #DenseNet_channels = 8**xD * 160 + 4**xD * 128 + 2**xD * 96 + 64 + 32
            #m0Chan = 224
            #m1Chan = 192
            #m2Chan = 160
            DenseNet_channels = 8**xD * 128 + 4**xD * 128 + 2**xD * 96 + 64 + 32
            m0Chan = 128
            m1Chan = 128
            m2Chan = 128
        elif (args.large_motion or args.full_motion) and args.useDenseNet:
            feature_init_channels = 16**xD * (correlation_channels + 32 + 2 + residualChannels)
            #DenseNet_channels = 8**xD * 160 + 4**xD * 128 + 2**xD * 96 + 64 + 32
            #m0Chan = 224
            #m1Chan = 192
            #m2Chan = 160
            DenseNet_channels = 8**xD * 128 + 4**xD * 128 + 2**xD * 96 + 64 + 32
            m0Chan = 128
            m1Chan = 128
            m2Chan = 128
        elif args.useDenseNet and args.useFlow1:
            feature_init_channels = 16**xD * (correlation_channels + 16 + 2 + residualChannels)
            DenseNet_channels = 8**xD * 128 + 4**xD * 128 + 2**xD * 96 + 64 + 32
            m0Chan = 128
            m1Chan = 128
            m2Chan = 128
        elif args.useDenseNet:
            feature_init_channels = 16**xD * (correlation_channels + 32 + 2 + residualChannels)
            DenseNet_channels = 8**xD * 128 + 4**xD * 128 + 2**xD * 96 + 64 + 32
            m0Chan = 128
            m1Chan = 128
            m2Chan = 128
        else:
            feature_init_channels = 0
            DenseNet_channels = 32
            m0Chan = 128
            m1Chan = 128
            m2Chan = 128
        print('refiner in_channels:',feature_init_channels + DenseNet_channels + flowChannels)

        if args.useOrginalModel:
            self.moduleMain = nn.Sequential(
                nn.Conv2d(in_channels  = feature_init_channels + DenseNet_channels + flowChannels,
                          out_channels = 128,
                          kernel_size  = 3,
                          stride       = 1,
                          padding      = 1,
                          dilation     = 1,
                          bias=False),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8, bias=False),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16, bias=False),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
            )
        elif args.useDenseNet and args.usePReLu:
            if args.useResidual:
                self.moduleMain = nn.Sequential(
                    nn.Conv2d(in_channels  = feature_init_channels + DenseNet_channels + flowChannels,
                              out_channels = m1Chan,
                              kernel_size  = 3,
                              stride       = 1,
                              padding      = 1,
                              dilation     = 1),
                    nn.PReLU(num_parameters=m1Chan,init=0.1),
                    nn.Conv2d(in_channels=m1Chan, out_channels=m2Chan, kernel_size=3, stride=1, padding=2, dilation=2),
                    nn.PReLU(num_parameters=m2Chan,init=0.1),
                    nn.Conv2d(in_channels=m2Chan, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    nn.PReLU(num_parameters=128,init=0.1),
                    nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    nn.PReLU(num_parameters=96,init=0.1),
                    nn.Conv2d(in_channels=96,  out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    nn.PReLU(num_parameters=64,init=0.1),
                    nn.Conv2d(in_channels=64,  out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.PReLU(num_parameters=32,init=0.1),
                    #nn.Upsample(scale_factor = 2, mode = 'bilinear'), # Double Resolution
                    nn.Conv2d(in_channels=32,  out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
                )
            elif args.FullScale:
                self.moduleMain = nn.Sequential(
                    nn.ConvTranspose2d(in_channels  = feature_init_channels + DenseNet_channels + flowChannels,
                                       out_channels = m0Chan,
                                       kernel_size  = 4,
                                       stride       = 2, # Double resolution
                                       padding      = 1),
                    nn.PReLU(num_parameters=m0Chan,init=0.1),
                    nn.Conv2d(in_channels=m0Chan, out_channels=m1Chan, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.PReLU(num_parameters=m1Chan,init=0.1),
                    nn.Conv2d(in_channels=m1Chan, out_channels=m2Chan, kernel_size=3, stride=1, padding=2, dilation=2),
                    nn.PReLU(num_parameters=m2Chan,init=0.1),
                    nn.Conv2d(in_channels=m2Chan, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    nn.PReLU(num_parameters=128,init=0.1),
                    nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    nn.PReLU(num_parameters=96,init=0.1),
                    nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    nn.PReLU(num_parameters=64,init=0.1),
                    #nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1), # Double resolution
                    nn.PReLU(num_parameters=32,init=0.1),
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.PReLU(num_parameters=32,init=0.1),
                    nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)#, nn.BatchNorm2d(2)
                )
            else:
                self.moduleMain = nn.Sequential(
                    nn.ConvTranspose2d(in_channels  = feature_init_channels + DenseNet_channels + flowChannels,
                                       out_channels = m0Chan,
                                       kernel_size  = 4,
                                       stride       = 2, # Double resolution
                                       padding      = 1),
                    nn.PReLU(num_parameters=m0Chan,init=0.1),
                    nn.Conv2d(in_channels=m0Chan, out_channels=m1Chan, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.PReLU(num_parameters=m1Chan,init=0.1),
                    nn.Conv2d(in_channels=m1Chan, out_channels=m2Chan, kernel_size=3, stride=1, padding=2, dilation=2),
                    nn.PReLU(num_parameters=m2Chan,init=0.1),
                    nn.Conv2d(in_channels=m2Chan, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    nn.PReLU(num_parameters=128,init=0.1),
                    nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    nn.PReLU(num_parameters=96,init=0.1),
                    nn.Conv2d(in_channels=96,  out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    nn.PReLU(num_parameters=64,init=0.1),
                    nn.Conv2d(in_channels=64,  out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.PReLU(num_parameters=32,init=0.1),
                    nn.Conv2d(in_channels=32,  out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
                )
        elif args.useDenseNet:
            if args.useResidual:
                self.moduleMain = nn.Sequential(
                    nn.Conv2d(in_channels  = feature_init_channels + DenseNet_channels + flowChannels,
                              out_channels = m1Chan,
                              kernel_size  = 3,
                              stride       = 1,
                              padding      = 1,
                              dilation     = 1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=m1Chan, out_channels=m2Chan, kernel_size=3, stride=1, padding=2, dilation=2),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=m2Chan, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=96,  out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=64,  out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    #nn.Upsample(scale_factor = 2, mode = 'bilinear'), # Double Resolution
                    nn.Conv2d(in_channels=32,  out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
                )
            elif args.FullScale:
                self.moduleMain = nn.Sequential(
                    nn.ConvTranspose2d(in_channels  = feature_init_channels + DenseNet_channels + flowChannels,
                                       out_channels = m0Chan,
                                       kernel_size  = 4,
                                       stride       = 2, # Double resolution
                                       padding      = 1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=m0Chan, out_channels=m1Chan, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=m1Chan, out_channels=m2Chan, kernel_size=3, stride=1, padding=2, dilation=2),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=m2Chan, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    #nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1), # Double resolution
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
                )
            else:
                self.moduleMain = nn.Sequential(
                    nn.ConvTranspose2d(in_channels  = feature_init_channels + DenseNet_channels + flowChannels,
                                       out_channels = m0Chan,
                                       kernel_size  = 4,
                                       stride       = 2, # Double resolution
                                       padding      = 1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=m0Chan, out_channels=m1Chan, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=m1Chan, out_channels=m2Chan, kernel_size=3, stride=1, padding=2, dilation=2),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=m2Chan, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=96,  out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=64,  out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=32,  out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
                )
        else:
            self.moduleMain = nn.Sequential(
                nn.Conv2d(in_channels  = feature_init_channels + DenseNet_channels + flowChannels,
                          out_channels = m1Chan,
                          kernel_size  = 3,
                          stride       = 1,
                          padding      = 1,
                          dilation     = 1),
                nn.PReLU(num_parameters=m1Chan,init=0.1),
                nn.Conv2d(in_channels=m1Chan, out_channels=m2Chan, kernel_size=3, stride=1, padding=2, dilation=2),
                nn.PReLU(num_parameters=m2Chan,init=0.1),
                nn.Conv2d(in_channels=m2Chan, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                nn.PReLU(num_parameters=128,init=0.1),
                nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                nn.PReLU(num_parameters=96,init=0.1),
                nn.Conv2d(in_channels=96,  out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                nn.PReLU(num_parameters=64,init=0.1),
                nn.Conv2d(in_channels=64,  out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.PReLU(num_parameters=32,init=0.1),
                nn.Conv2d(in_channels=32,  out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
            )
        # Original Caffe model used MSRA. This is also known as Kaiming
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0) 
    # end

    def forward(self, tensorInput):
        #print('tensorInput.shape',tensorInput.shape)
        if self.FullScale and self.scale_supervision:
            return self.moduleMain(tensorInput)*4
        elif self.scale_supervision:
            return self.moduleMain(tensorInput)*2
        else:
            return self.moduleMain(tensorInput)
    # end
# end


class Network(nn.Module):
    """ Duplicate class from `run.py` of
    [A reimplementation of PWC-Net in PyTorch that matches the official Caffe version](https://github.com/sniklaus/pytorch-pwc)
    
    """
    def __init__(self,
                 args):
        super(Network, self).__init__()
        self.use_checkpoint = args.use_checkpoint
        self.useFlow1 = args.useFlow1
        self.FullScale = args.FullScale
        self.scale_output = args.scale_output
        self.useContext = args.useContext
        self.useResidual = args.useResidual
        self.useDenseNet = args.useDenseNet
        self.UpScale = args.UpScale
        self.max_displacement = args.max_displacement[5]
        self.scale_supervision = args.scale_supervision
        self.device = args.device
        self.lr_finder = args.lr_finder
        self.div_flow = torch.as_tensor(args.div_flow)

        self.moduleExtractor = Extractor(args=args)

        self.moduleOne = Decoder(1, args=args)
        self.moduleTwo = Decoder(2, args=args)
        self.moduleThr = Decoder(3, args=args)
        self.moduleFou = Decoder(4, args=args)
        self.moduleFiv = Decoder(5, args=args)
        self.moduleSix = Decoder(6, args=args)

        if args.useContext:
            self.moduleRefiner = Refiner(args=args)

        #self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey,
        #                      tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-pwc/network-'
        #                                                                          + arguments_strModel + '.pytorch',
        #                                                                      file_name='pwc-' + arguments_strModel).items() })
    # end__init__

    def custom(self, module):
        """This is used for Checkpointing
        See the following website for more information
        https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
        """
        def custom_forward(*inputs):
            if type(module) in [type(self.moduleExtractor), type(self.moduleRefiner)]:
                inputs = module(inputs[0])
            else:
                inputs = module(inputs[0], inputs[1], inputs[2])
            return inputs
        return custom_forward
    # end_custom

    def forward(self, x):
        
        # Need input image resolution to always be a multiple of 64
        #
        #if intPreprocessedWidth == intWidth and intPreprocessedHeight == intHeight:
        #    # Faster but same memory utilization. Without detach it is slower but takes less memory.


        tensorBackward = x[:, 0, :, :, :]
        tensorCenter   = x[:, 1, :, :, :]
        tensorForward  = x[:, 2, :, :, :]


        if self.training:
            outputF=[None,None,None,None,None,None]
            outputB=[None,None,None,None,None,None]

        tensorBackward = self.moduleExtractor(tensorBackward)
        tensorCenter   = self.moduleExtractor(tensorCenter)
        tensorForward  = self.moduleExtractor(tensorForward)
        
        objectEstimate6 = self.moduleSix(tensorBackward[-1], tensorCenter[-1], tensorForward[-1], None)
        if self.training:
            outputF[5] = objectEstimate6['tensorForwardFlow']
            outputB[5] = objectEstimate6['tensorBackwardFlow']
        objectEstimate5 = self.moduleFiv(tensorBackward[-2], tensorCenter[-2], tensorForward[-2], objectEstimate6)
        if self.training:
            outputF[4] = objectEstimate5['tensorForwardFlow']
            outputB[4] = objectEstimate5['tensorBackwardFlow']
        objectEstimate4 = self.moduleFou(tensorBackward[-3], tensorCenter[-3], tensorForward[-3], objectEstimate5)
        if self.training:
            outputF[3] = objectEstimate4['tensorForwardFlow']
            outputB[3] = objectEstimate4['tensorBackwardFlow']
        objectEstimate3 = self.moduleThr(tensorBackward[-4], tensorCenter[-4], tensorForward[-4], objectEstimate4)
        if self.training:
            outputF[2] = objectEstimate3['tensorForwardFlow']
            outputB[2] = objectEstimate3['tensorBackwardFlow']
        objectEstimate2 = self.moduleTwo(tensorBackward[-5], tensorCenter[-5], tensorForward[-5], objectEstimate3)
        if self.training:
            outputF[1] = objectEstimate2['tensorForwardFlow']
            outputB[1] = objectEstimate2['tensorBackwardFlow']
        if self.useFlow1:
            objectEstimate1 = self.moduleOne(tensorBackward[-6], tensorCenter[-6], tensorForward[-6], objectEstimate2)
            objectEstimateB = objectEstimate1['tensorBackwardFlow']
            objectEstimateF = objectEstimate1['tensorForwardFlow']
            
        else:
            objectEstimateB = objectEstimate2['tensorBackwardFlow']
            objectEstimateF = objectEstimate2['tensorForwardFlow']


        if self.training :
            if self.useResidual:
                scale_factor = [2,4,8,16,32,64]
            else:
                if self.FullScale:
                    scale_factor = [1,1,4,8,16,32,64]
                elif self.useFlow1:
                    scale_factor = [1,2,4,8,16,32,64]
                else:
                    scale_factor = [2,2,4,8,16,32,64]
            # bilinear interpolate is nondeterministic. However the objectEstimate2 is 0.25 the size of the original image
            # Move to CPU to make it deterministic
            if self.useFlow1 or self.useContext:
                outputB[0] = objectEstimateB
                outputF[0] = objectEstimateF
            else:
                outputB[0] = None
                outputF[0] = None
            for i in range(len(outputB)):
                if outputB[i] is not None:
                    #if scale_factor[i] != 1:
                    if self.scale_output:
                        outputB[i] = nn.functional.interpolate(
                                            input         = outputB[i],
                                            scale_factor  = scale_factor[i],
                                            mode          = 'bilinear',
                                            align_corners = False,
                                            antialias     = True)
                        outputF[i] = nn.functional.interpolate(
                                            input         = outputF[i],
                                            scale_factor  = scale_factor[i],
                                            mode          = 'bilinear',
                                            align_corners = False,
                                            antialias     = True)
                    if self.scale_supervision:
                        outputB[i] *= scale_factor[i]
                        outputF[i] *= scale_factor[i]
                    else:
                        outputB[i] = outputB[i] * self.div_flow
                        outputF[i] = outputF[i] * self.div_flow
                    #output[i] *= scale_factor[i]
            return outputB, outputF
        else:
            if self.useFlow1:
                if self.useResidual:
                    scale_factor = 2
                else:
                    scale_factor = 1
            else:
                if self.useResidual:
                    scale_factor = 4
                elif self.FullScale:
                    scale_factor = 1
                else:
                    scale_factor = 2
                # bilinear interpolate is nondeterministic. However the objectEstimate2 is 0.25 the size of the original image
                # Move to CPU to make it deterministic
                if scale_factor != 1:
                    objectEstimateB = nn.functional.interpolate(
                                        input         = objectEstimateB,
                                        #size          = (intHeight, intWidth),
                                        scale_factor  = scale_factor,
                                        mode          = 'bilinear',
                                        align_corners = False,
                                        antialias     = True)
                    objectEstimateF = nn.functional.interpolate(
                                        input         = objectEstimateF,
                                        #size          = (intHeight, intWidth),
                                        scale_factor  = scale_factor,
                                        mode          = 'bilinear',
                                        align_corners = False,
                                        antialias     = True)
            if self.scale_supervision:
                objectEstimateB *= scale_factor
                objectEstimateF *= scale_factor
            else:
                objectEstimateB *= self.div_flow
                objectEstimateF *= self.div_flow
            #objectEstimate[:, 0, :, :] *= self.div_flow * intWidth / intPreprocessedWidth
            #objectEstimate[:, 1, :, :] *= self.div_flow * intHeight / intPreprocessedHeight
            return objectEstimateB, objectEstimateF
    # end_forward
# end_Network


def boflow_pwc(data             = None,
              args             = None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)
    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    moduleNetwork = Network(args = args)
    
    if data is not None:
        if 'state_dict' in data.keys():
            moduleNetwork.load_state_dict(data['state_dict'])
    return moduleNetwork
