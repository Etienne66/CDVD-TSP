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
    return Flow_PWC2(device         = device,
                     use_checkpoint = args.use_checkpoint)


class Flow_PWC2(nn.Module):
    """Derivative of `run.py` from
    [A reimplementation of PWC-Net in PyTorch that matches the official Caffe version](https://github.com/sniklaus/pytorch-pwc)
    
    """
    def __init__(self,
                 device         = 'cuda',
                 use_checkpoint = False):
        super(Flow_PWC2, self).__init__()
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
                                        mode          = 'bicubic',
                                        align_corners = False)
            tensorPreprocessedSecond = nn.functional.interpolate(
                                        input         = tensorSecond,
                                        size          = (intPreprocessedHeight, intPreprocessedWidth),
                                        mode          = 'bicubic',
                                        align_corners = False)

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
                                           mode='bicubic',
                                           padding_mode  = 'border',
                                           align_corners = True)
        mask = torch.ones_like(x,
                               device        = self.device,
                               requires_grad = True)
        mask = nn.functional.grid_sample(mask,
                                         grid,
                                         mode='bicubic',
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
    tensorHorizontal = torch.linspace(-1.0 + (1.0 / tensorFlow.shape[3]),
                                      1.0 - (1.0 / tensorFlow.shape[3]),
                                      tensorFlow.shape[3],
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
    tensorVertical = torch.linspace(-1.0 + (1.0 / tensorFlow.shape[2]),
                                    1.0 - (1.0 / tensorFlow.shape[2]),
                                    tensorFlow.shape[2],
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
                                                 requires_grad=requires_grad)
    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.shape[3] - 1.0) / 2.0),
                            tensorFlow[:, 1:2, :, :] / ((tensorInput.shape[2] - 1.0) / 2.0)], 1)
    # Add the mask as the last channel of the input
    tensorInput = torch.cat([tensorInput, Backwarp_tensorPartial], 1)

    #Move to CPU to make it deterministic
    #Changed padding_mode from zeros to border because we shouldn't assume that the motion stops at the border
    tensorOutput = nn.functional.grid_sample(input         = tensorInput,
                                             grid          = (Backwarp_tensorGrid + tensorFlow).permute(0, 2, 3, 1),
                                             mode          = 'bicubic',
                                             padding_mode  = 'zeros',
                                             align_corners = False)
    # Extract the mask from the last channel of the output.
    # The mask is to filter out the parts of the image that a flow can't be calculated for. Normally around the edges of the image.
    tensorMask = tensorOutput[:, -1:, :, :];
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
    def __init__(self, batchnorm=False, large_motion=False):
        super(Extractor, self).__init__()

        if large_motion:
            m4Chan = 128
            m5Chan = 256
            m6Chan = 512
        else:
            m4Chan = 96
            m5Chan = 128
            m6Chan = 192
        
        if batchnorm:
            self.moduleOne = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            self.moduleTwo = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            self.moduleThr = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            self.moduleFou = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=m4Chan, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(m4Chan),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=m4Chan, out_channels=m4Chan, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(m4Chan),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=m4Chan, out_channels=m4Chan, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(m4Chan),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            self.moduleFiv = nn.Sequential(
                nn.Conv2d(in_channels=m4Chan, out_channels=m5Chan, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(m5Chan),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=m5Chan, out_channels=m5Chan, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(m5Chan),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=m5Chan, out_channels=m5Chan, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(m5Chan),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            # Author intended 192(32*6) but had used 196. 160(32*5) makes more sense since the prior levels were multiples of 32
            # 32*0.5, 32*1, 32*2, 32*3, 32*4
            self.moduleSix = nn.Sequential(
                nn.Conv2d(in_channels=m5Chan, out_channels=m6Chan, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(m6Chan),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=m6Chan, out_channels=m6Chan, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(m6Chan),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=m6Chan, out_channels=m6Chan, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(m6Chan),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
        else:
            self.moduleOne = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            self.moduleTwo = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            self.moduleThr = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            self.moduleFou = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=m4Chan, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=m4Chan, out_channels=m4Chan, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=m4Chan, out_channels=m4Chan, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            self.moduleFiv = nn.Sequential(
                nn.Conv2d(in_channels=m4Chan, out_channels=m5Chan, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=m5Chan, out_channels=m5Chan, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=m5Chan, out_channels=m5Chan, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            # Author intended 192(32*6) but had used 196. 160(32*5) makes more sense since the prior levels were multiples of 32
            # 32*0.5, 32*1, 32*2, 32*3, 32*4
            self.moduleSix = nn.Sequential(
                nn.Conv2d(in_channels=m5Chan, out_channels=m6Chan, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=m6Chan, out_channels=m6Chan, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=m6Chan, out_channels=m6Chan, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
        
        # Original Caffe model used MSRA. This is also known as Kaiming
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    # end

    def forward(self, tensorInput):
        tensorOne = self.moduleOne(tensorInput)
        tensorTwo = self.moduleTwo(tensorOne)
        tensorThr = self.moduleThr(tensorTwo)
        tensorFou = self.moduleFou(tensorThr)
        tensorFiv = self.moduleFiv(tensorFou)
        tensorSix = self.moduleSix(tensorFiv)

        # Removed tensorOne from return because it wasn't used
        return [tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix]
        #return [tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix]
    # end
# end

class Decoder(nn.Module):
    """The optical flow estimator network. Each convolutional layer is followed by a leaky ReLU unit except the last one that
    outputs the optical flow.
    Paper indicates that 5 layers are used but appears to be 6 layers
    """
    def __init__(self,
                 intLevel,
                 device           = 'cuda',
                 div_flow         = 20,
                 max_displacement = 4,
                 batchnorm        = False,
                 large_motion     = False,
                 useResidual      = False,
                 useDenseNet      = False,
                 mixed_precision  = False):
        super(Decoder, self).__init__()
        self.device=device
        self.max_displacement = 4
        self.div_flow = div_flow
        self.useResidual = useResidual
        self.useDenseNet = useDenseNet
        self.max_displacement = max_displacement
        correlation_channels = (max_displacement * 2 + 1) ** 2;

        if useResidual:
            residualChannels = 2
        else:
            residualChannels = 0

        if False:
            m1Chan = 128
            intChannels = [None,
                           correlation_channels +  32 + 2 + residualChannels,
                           correlation_channels +  64 + 2 + residualChannels, 
                           correlation_channels +  96 + 2 + residualChannels,
                           correlation_channels + 128 + 2 + residualChannels,
                           correlation_channels + 160 + 2 + residualChannels,
                           correlation_channels + 192,
                           None]
        elif large_motion:
            m1Chan = 160
            intChannels = [None,
                           correlation_channels +  16 + 2 + residualChannels, #101
                           correlation_channels +  32 + 2 + residualChannels, #117 
                           correlation_channels +  64 + 2 + residualChannels, #149
                           correlation_channels + 128 + 2 + residualChannels, #213
                           correlation_channels + 256 + 2 + residualChannels, #341
                           correlation_channels + 512,         #593
                           None]
        else:
            m1Chan = 128
            intChannels = [None,
                           correlation_channels +  16 + 2 + residualChannels, #101
                           correlation_channels +  32 + 2 + residualChannels, #117
                           correlation_channels +  64 + 2 + residualChannels, #149
                           correlation_channels +  96 + 2 + residualChannels, #181
                           correlation_channels + 128 + 2 + residualChannels, #213
                           correlation_channels + 192,         #273
                           None]

        intPrevious = intChannels[intLevel + 1]
        intCurrent  = intChannels[intLevel + 0]

        corr_channels = [None, 16, 32, 64, 96, 128, 192, None][intLevel + 0]

        if intLevel < 6:
            if useDenseNet:
                #self.moduleUpflow = nn.ConvTranspose2d(in_channels  = 2,
                #                                       out_channels = 2,
                #                                       kernel_size  = 4,
                #                                       stride       = 2, # Double resolution
                #                                       padding      = 1)
                #    
                #self.moduleUpfeat = nn.ConvTranspose2d(in_channels  = intPrevious + m1Chan + 128 + 96 + 64 + 32,
                #                                       out_channels = 2,
                #                                       kernel_size  = 4,
                #                                       stride       = 2, # Double resolution
                #                                       padding      = 1)
                self.moduleUpflow = nn.Sequential(nn.Upsample(scale_factor  = 2,
                                                              mode          = 'bicubic'),
                                                  nn.Conv2d(in_channels  = 2,
                                                            out_channels = 2,
                                                            kernel_size  = 3,
                                                            stride       = 1,
                                                            padding      = 1))
                self.moduleUpfeat = nn.Sequential(nn.Upsample(scale_factor  = 2,
                                                              mode          = 'bicubic'),
                                                  nn.Conv2d(in_channels  = intPrevious + m1Chan + 128 + 96 + 64 + 32,
                                                            out_channels = 2,
                                                            kernel_size  = 3,
                                                            stride       = 1,
                                                            padding      = 1))
            else:
                #self.moduleUpflow = nn.ConvTranspose2d(in_channels  = 2,
                #                                       out_channels = 2,
                #                                       kernel_size  = 4,
                #                                       stride       = 2, # Double resolution
                #                                       padding      = 1)
                self.moduleUpflow = nn.Upsample(scale_factor  = 2,
                                                mode          = 'bicubic')
                    
                self.moduleUpfeat = nn.Sequential(nn.Upsample(scale_factor  = 2,
                                                              mode          = 'bicubic'),
                                                  nn.Conv2d(in_channels  = 32,
                                                            out_channels = 2,
                                                            kernel_size  = 3,
                                                            stride       = 1,
                                                            padding      = 1))
                #self.moduleUpfeat = nn.ConvTranspose2d(in_channels  = intPrevious,
                #                                       out_channels = 2,
                #                                       kernel_size  = 4,
                #                                       stride       = 2, # Double resolution
                #                                       padding      = 1)
            # Start with inLevel 6(None), 5(20/32=0.625), 4(20/16=1.25), 3(20/8=2.5) and finally 2(20/4=5.0)
            # 6 doesn't have an objectPrevious so doesn't need a value. 2 is the last.
            # Add 1(20/2=10.0)
            #self.dblBackwarp = [None, None, 10.0, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]
            self.dblBackwarp = [None, div_flow/1, div_flow/2, div_flow/4, div_flow/8, div_flow/16, div_flow/32, div_flow/64][intLevel + 1]

        
        self.leakyrelu = nn.LeakyReLU(inplace=False, negative_slope=0.1)
        
        
        if batchnorm:
            self.correlation = nn.Sequential(
                correlation2.ModuleCorrelation(padding          = max_displacement,
                                               kernel_size      = 1,
                                               max_displacement = max_displacement,
                                               stride1          = 1,
                                               stride2          = 1,
                                               device           = device,
                                               mixed_precision  = mixed_precision),
                nn.BatchNorm2d(correlation_channels),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            if useDenseNet:
                self.moduleOne = nn.Sequential(
                    nn.Conv2d(in_channels  = intCurrent,
                              out_channels = m1Chan,
                              kernel_size  = 3,
                              stride       = 1,
                              padding      = 1,
                              bias         = False),
                    nn.BatchNorm2d(m1Chan),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.moduleTwo = nn.Sequential(
                    nn.Conv2d(in_channels  = intCurrent + m1Chan,
                              out_channels = 128,
                              kernel_size  = 3,
                              stride       = 1,
                              padding      = 1,
                              bias         = False),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.moduleThr = nn.Sequential(
                    nn.Conv2d(in_channels  = intCurrent + m1Chan + 128,
                              out_channels = 96,
                              kernel_size  = 3,
                              stride       = 1,
                              padding      = 1,
                              bias         = False),
                    nn.BatchNorm2d(96),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.moduleFou = nn.Sequential(
                    nn.Conv2d(in_channels  = intCurrent + m1Chan + 128 + 96,
                              out_channels = 64,
                              kernel_size  = 3,
                              stride       = 1,
                              padding      = 1,
                              bias         = False),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.moduleFiv = nn.Sequential(
                    nn.Conv2d(in_channels  = intCurrent + m1Chan + 128 + 96 + 64,
                              out_channels = 32,
                              kernel_size  = 3,
                              stride       = 1,
                              padding      = 1,
                              bias         = False),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.moduleSix = nn.Sequential(
                    #nn.Upsample(scale_factor = 2,
                    #            mode         = 'bicubic'). # Double Resolution
                    nn.Conv2d(in_channels  = intCurrent + m1Chan + 128 + 96 + 64 + 32,
                              out_channels = 2,
                              kernel_size  = 3,
                              stride       = 1,
                              padding      = 1,
                              bias         = False),
                    nn.BatchNorm2d(2)
                )
            else:
                self.noDenseNet = torch.nn.Sequential(
                    nn.Conv2d(in_channels=intCurrent, out_channels=m1Chan, kernel_size=3, stride=1, padding=1, bias = False),
                    nn.BatchNorm2d(m1Chan),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=m1Chan, out_channels=128, kernel_size=3, stride=1, padding=1, bias = False),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1, bias = False),
                    nn.BatchNorm2d(96),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1, bias = False),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias = False),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.moduleSix = torch.nn.Sequential(
                    #nn.Upsample(scale_factor = 2, mode = 'bicubic'), # Double Resolution
                    nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, bias = False),
                    nn.BatchNorm2d(2)
                )
        elif useDenseNet:
            self.correlation = nn.Sequential(
                correlation2.ModuleCorrelation(padding = max_displacement,
                                               kernel_size = 1,
                                               max_displacement = max_displacement,
                                               stride1 = 1,
                                               stride2 = 1,
                                               device = device),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            self.moduleOne = nn.Sequential(
                nn.Conv2d(in_channels  = intCurrent,
                          out_channels = m1Chan,
                          kernel_size  = 3,
                          stride       = 1,
                          padding      = 1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            self.moduleTwo = nn.Sequential(
                nn.Conv2d(in_channels  = intCurrent + m1Chan,
                          out_channels = 128,
                          kernel_size  = 3,
                          stride       = 1,
                          padding      = 1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            self.moduleThr = nn.Sequential(
                nn.Conv2d(in_channels  = intCurrent + m1Chan + 128,
                          out_channels = 96,
                          kernel_size  = 3,
                          stride       = 1,
                          padding      = 1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            self.moduleFou = nn.Sequential(
                nn.Conv2d(in_channels  = intCurrent + m1Chan + 128 + 96,
                          out_channels = 64,
                          kernel_size  = 3,
                          stride       = 1,
                          padding      = 1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            self.moduleFiv = nn.Sequential(
                nn.Conv2d(in_channels  = intCurrent + m1Chan + 128 + 96 + 64,
                          out_channels = 32,
                          kernel_size  = 3,
                          stride       = 1,
                          padding      = 1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            self.moduleSix = nn.Sequential(
                nn.Upsample(scale_factor = 2,
                            mode         = 'bicubic'), # Double Resolution
                nn.Conv2d(in_channels  = intCurrent + m1Chan + 128 + 96 + 64 + 32,
                          out_channels = 2,
                          kernel_size  = 3,
                          stride       = 1,
                          padding      = 1)
               
            )
        else:
            self.correlation = nn.Sequential(
                correlation2.ModuleCorrelation(padding = max_displacement,
                                               kernel_size = 1,
                                               max_displacement = max_displacement,
                                               stride1 = 1,
                                               stride2 = 1,
                                               device = device),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            self.noDenseNet = torch.nn.Sequential(
                nn.Conv2d(in_channels=intCurrent, out_channels=m1Chan, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=m1Chan, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            self.moduleSix = torch.nn.Sequential(
                #nn.Upsample(scale_factor = 2, mode = 'bicubic'), # Double Resolution
                nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
            )

        # Original Caffe model used MSRA. This is also known as Kaiming
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # end

    def compute_cost_volume(self, feat1, feat2):
        """
        only implemented for:
            kernel_size = 1
            stride1 = 1
            stride2 = 1
        """

        max_disp = self.max_displacement

        _, _, height, width = feat1.size()
        num_shifts = 2 * max_disp + 1
        feat2_padded = F.pad(feat2, (max_disp, max_disp, max_disp, max_disp), "constant", 0)

        cost_list = []
        for i in range(num_shifts):
            for j in range(num_shifts):
                corr = torch.mean(feat1 * feat2_padded[:, :, i:(height + i), j:(width + j)], dim=1, keepdim=True)
                cost_list.append(corr)
        cost_volume = torch.cat(cost_list, dim=1)
        return cost_volume


    def forward(self, tensorFirst, tensorSecond, objectPrevious):
        tensorFlow = None
        tensorFeat = None

        if objectPrevious is None:
            tensorFlow = None
            tensorFeat = None

            #Cost volume layer. Output is always 81 channels
            tensorCor = torch.stack((tensorFirst, tensorSecond), dim=1)
            tensorVolume = self.correlation(tensorCor)
            #tensorVolume = self.leakyrelu(self.compute_cost_volume(tensorFirst, tensorSecond))
            #Cost volume layer. Output is always 81 channels
            #tensorVolume =  self.leakyrelu(correlation.FunctionCorrelation(tensorFirst  = tensorFirst,
            #                                                               tensorSecond = tensorSecond))

            tensorFeat = torch.cat([tensorVolume, tensorFirst], 1)

        elif objectPrevious is not None:
            if self.useResidual:
                #if self.useDenseNet:
                #else:
                tensorFlow = objectPrevious['tensorFlow'] # 2 channels & double resolution
                #tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow']) # 2 channels & double resolution
                #Residual Connections Flow Increment. Reduced to 2 channels & double resolution
                tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat'])
                #tensorFeat = objectPrevious['tensorFeat']
                # Warping layer of Flow Increment
                tensorWarp = Backwarp(tensorInput   = tensorSecond,
                                      tensorFlow    = tensorFlow * self.dblBackwarp, #self.dblBackwarp,
                                      requires_grad = self.training)
            else:
                #if self.useDenseNet:
                #    tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow']) # 2 channels & double resolution
                #else:
                tensorFlow = objectPrevious['tensorFlow'] # 2 channels
                # Warping layer of Flow Increment
                tensorWarp = Backwarp(tensorInput   = tensorSecond,
                                      tensorFlow    = tensorFlow * self.dblBackwarp, #self.dblBackwarp,self.div_flow
                                      #tensorFlow    = objectPrevious['tensorFlow'] * self.dblBackwarp,
                                      requires_grad = self.training)

            #Cost volume layer. Output is always 81 channels
            tensorCor = torch.stack((tensorFirst, tensorWarp), dim=1)
            tensorVolume = self.correlation(tensorCor)
            #tensorVolume = self.leakyrelu(self.compute_cost_volume(tensorFirst, tensorWarp))
            #tensorVolume = self.leakyrelu(correlation.FunctionCorrelation(tensorFirst  = tensorFirst,
            #                                                              tensorSecond = tensorWarp))

            if self.useResidual:
                tensorFeat = torch.cat([tensorFirst, tensorVolume, tensorFlow, tensorFeat], 1)
            else:
                tensorFeat = torch.cat([tensorFirst, tensorVolume, tensorFlow], 1)
        # end

        # Concatenating the previous features with the current features is a DenseNet connection
        # This is the CNN
        
        if self.useDenseNet:
            tensorFeat = torch.cat([self.moduleOne(tensorFeat), tensorFeat], 1)
            tensorFeat = torch.cat([self.moduleTwo(tensorFeat), tensorFeat], 1)
            tensorFeat = torch.cat([self.moduleThr(tensorFeat), tensorFeat], 1)
            tensorFeat = torch.cat([self.moduleFou(tensorFeat), tensorFeat], 1)
            tensorFeat = torch.cat([self.moduleFiv(tensorFeat), tensorFeat], 1)
            tensorFlow = self.moduleSix(tensorFeat) #* 1 #* 2
        else:
            tensorFeat = self.noDenseNet(tensorFeat)
            tensorFlow = self.moduleSix(tensorFeat) #* 1 #* 2

        return {
            'tensorFlow': tensorFlow,
            'tensorFeat': tensorFeat
        }
    # end
# end

class Refiner(nn.Module):
    """The context network. Each convolutional layer is followed by a leaky ReLU unit except the last one that outputs the
    optical flow. The last number in each convolutional layer denotes the dilation constant.
    """
    def __init__(self,
                 batchnorm        = False,
                 useFlow1         = False,
                 large_motion     = False,
                 useResidual      = False,
                 useDenseNet      = False,
                 max_displacement = 4):
        super(Refiner, self).__init__()
        correlation_channels = (max_displacement * 2 + 1) ** 2;
        
        flowChannels = 0
        if useResidual:
            residualChannels = 2
        else:
            residualChannels = 0
            
        #features
        #feature_init_channels = 81 + 64 + 2 + 2
        #feature_cnn_channels = 128 + 128 + 96 + 64 + 32
        if large_motion and useDenseNet and useFlow1:
            feature_init_channels = correlation_channels + 16 + 2 + residualChannels
            DenseNet_channels = 160 + 128 + 96 + 64 + 32
            m1Chan = 192
            m2Chan = 160
        elif large_motion and useDenseNet:
            feature_init_channels = correlation_channels + 32 + 2 + residualChannels
            DenseNet_channels = 160 + 128 + 96 + 64 + 32
            m1Chan = 192
            m2Chan = 160
        elif useDenseNet and useFlow1:
            feature_init_channels = correlation_channels + 16 + 2 + residualChannels
            DenseNet_channels = 128 + 128 + 96 + 64 + 32
            m1Chan = 128
            m2Chan = 128
        elif useDenseNet:
            feature_init_channels = correlation_channels + 32 + 2 + residualChannels
            DenseNet_channels = 128 + 128 + 96 + 64 + 32
            m1Chan = 128
            m2Chan = 128
        else:
            feature_init_channels = 0
            DenseNet_channels = 32
            m1Chan = 128
            m2Chan = 128
        print('feature_init_channels:',feature_init_channels)

        if batchnorm:
            self.moduleMain = nn.Sequential(
                nn.Conv2d(in_channels  = feature_init_channels + DenseNet_channels + flowChannels,
                          out_channels = m1Chan,
                          kernel_size  = 3,
                          stride       = 1,
                          padding      = 1,
                          dilation     = 1,
                          bias=False),
                nn.BatchNorm2d(m1Chan),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=m1Chan, out_channels=m2Chan, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(m2Chan),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=m2Chan, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8, bias=False),
                nn.BatchNorm2d(96),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                nn.BatchNorm2d(2)
            )
        elif useDenseNet:
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
                nn.Upsample(scale_factor = 2, mode = 'bicubic'), # Double Resolution
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
                nn.Upsample(scale_factor = 2, mode = 'bicubic'), # Double Resolution
                nn.Conv2d(in_channels=32,  out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
            )
        # Original Caffe model used MSRA. This is also known as Kaiming
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    # end

    def forward(self, tensorInput):
        #print('tensorInput.shape',tensorInput.shape)
        return self.moduleMain(tensorInput)
    # end
# end


class Network(nn.Module):
    """ Duplicate class from `run.py` of
    [A reimplementation of PWC-Net in PyTorch that matches the official Caffe version](https://github.com/sniklaus/pytorch-pwc)
    
    """
    def __init__(self,
                 device           = 'cuda',
                 use_checkpoint   = False,
                 useFlow1         = False,
                 useContext       = False,
                 useResidual      = False,
                 useDenseNet      = False,
                 max_displacement = 4,
                 lr_finder        = False,
                 div_flow         = 1,
                 batchnorm        = False,
                 large_motion     = False,
                 mixed_precision  = False):
        super(Network, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.useFlow1 = useFlow1
        self.useContext = useContext
        self.useResidual = useResidual
        self.useDenseNet = useDenseNet
        self.device = device
        self.lr_finder = lr_finder
        self.div_flow = torch.as_tensor(div_flow)

        self.moduleExtractor = Extractor(batchnorm=batchnorm, large_motion=large_motion)

        self.moduleOne = Decoder(1, device=device, div_flow=self.div_flow, batchnorm=batchnorm, large_motion=large_motion, useResidual=useResidual, useDenseNet=useDenseNet, max_displacement=max_displacement, mixed_precision=mixed_precision)
        self.moduleTwo = Decoder(2, device=device, div_flow=self.div_flow, batchnorm=batchnorm, large_motion=large_motion, useResidual=useResidual, useDenseNet=useDenseNet, max_displacement=max_displacement, mixed_precision=mixed_precision)
        self.moduleThr = Decoder(3, device=device, div_flow=self.div_flow, batchnorm=batchnorm, large_motion=large_motion, useResidual=useResidual, useDenseNet=useDenseNet, max_displacement=max_displacement, mixed_precision=mixed_precision)
        self.moduleFou = Decoder(4, device=device, div_flow=self.div_flow, batchnorm=batchnorm, large_motion=large_motion, useResidual=useResidual, useDenseNet=useDenseNet, max_displacement=max_displacement, mixed_precision=mixed_precision)
        self.moduleFiv = Decoder(5, device=device, div_flow=self.div_flow, batchnorm=batchnorm, large_motion=large_motion, useResidual=useResidual, useDenseNet=useDenseNet, max_displacement=max_displacement, mixed_precision=mixed_precision)
        self.moduleSix = Decoder(6, device=device, div_flow=self.div_flow, batchnorm=batchnorm, large_motion=large_motion, useResidual=useResidual, useDenseNet=useDenseNet, max_displacement=max_displacement, mixed_precision=mixed_precision)

        if useContext:
            self.moduleRefiner = Refiner(batchnorm=batchnorm, useFlow1=useFlow1, large_motion=large_motion, useResidual=useResidual, useDenseNet=useDenseNet, max_displacement=max_displacement)

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
        #b, N, c, intHeight, intWidth = x.size()
        
        # Need input image resolution to always be a multiple of 64
        #intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        #intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))
        #
        #if intPreprocessedWidth == intWidth and intPreprocessedHeight == intHeight:
        #    # Faster but same memory utilization. Without detach it is slower but takes less memory.

        tensorFirst = x[:, 0, :, :, :]
        tensorSecond = x[:, 1, :, :, :]

        #else:
        #    tensorFirst = nn.functional.interpolate(
        #                    input         = x[:,0,:,:,:],
        #                    size          = (intPreprocessedHeight, intPreprocessedWidth),
        #                    mode          = 'bicubic',
        #                    align_corners = False,
        #                    antialias     = True)
        #    tensorSecond = nn.functional.interpolate(
        #                    input         = x[:,1,:,:,:],
        #                    size          = (intPreprocessedHeight, intPreprocessedWidth),
        #                    mode          = 'bicubic',
        #                    align_corners = False,
        #                    antialias     = True)


        if self.training:
            output=[None,None,None,None,None,None]

        tensorFirst  = self.moduleExtractor(tensorFirst)
        tensorSecond = self.moduleExtractor(tensorSecond)
        
        objectEstimate6 = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None)
        if self.training:
            output[5] = objectEstimate6['tensorFlow']
        objectEstimate5 = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate6)
        if self.training:
            output[4] = objectEstimate5['tensorFlow']
        objectEstimate4 = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate5)
        if self.training:
            output[3] = objectEstimate4['tensorFlow']
        objectEstimate3 = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate4)
        if self.training:
            output[2] = objectEstimate3['tensorFlow']
        objectEstimate2 = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate3)
        if self.useFlow1:
            if self.training:
                output[1] = objectEstimate2['tensorFlow']
            objectEstimate1 = self.moduleOne(tensorFirst[-6], tensorSecond[-6], objectEstimate2)
            if self.useContext:
                objectEstimate = objectEstimate1['tensorFlow'] + self.moduleRefiner(objectEstimate1['tensorFeat'])
                #objectEstimate = self.moduleRefiner(torch.cat(objectEstimate1['tensorFlow'], objectEstimate1['tensorFeat'], dim=1))
            else:
                objectEstimate = objectEstimate1['tensorFlow']
            
        else:
            if self.useContext:
                objectEstimate = objectEstimate2['tensorFlow'] + self.moduleRefiner(objectEstimate2['tensorFeat'])
                #objectEstimate = self.moduleRefiner(torch.cat(objectEstimate2['tensorFlow'], objectEstimate2['tensorFeat'], dim=1))
            else:
                objectEstimate = objectEstimate2['tensorFlow']

        #objectEstimate = objectEstimate2['tensorFlow'] + self.moduleRefiner(objectEstimate2['tensorFeat'])
        #objectEstimate = objectEstimate1['tensorFlow'] + self.moduleRefiner(objectEstimate1['tensorFeat'])
        #objectEstimate = self.moduleRefiner(torch.cat([objectEstimate2['tensorFlow'], objectEstimate2['tensorFeat']], dim=1))
            
        if self.training :
            scale_factor = [1,2,4,8,16,32,64]
            #scale_factor = [2,4,8,16,32,64]
            # bilinear interpolate is nondeterministic. However the objectEstimate2 is 0.25 the size of the original image
            # Move to CPU to make it deterministic
            if self.useFlow1:
                output[0] = objectEstimate
            else:
                output[1] = objectEstimate
                output[0] = None
            for i in range(len(output)):
                if output[i] is not None:
                    if scale_factor[i] != 1:
                        output[i] = nn.functional.interpolate(
                                            input         = output[i],
                                            #size          = (intHeight, intWidth),
                                            scale_factor  = scale_factor[i],
                                            mode          = 'bicubic',
                                            align_corners = False)
                    #output[i] *= scale_factor[i]
                    #output[i] *= self.div_flow * scale_factor[i]
                    output[i] *= self.div_flow
                    #output[i][:, 0, :, :] *= self.div_flow * intWidth / intPreprocessedWidth
                    #output[i][:, 1, :, :] *= self.div_flow * intHeight / intPreprocessedHeight
            return output
        else:
            if self.useFlow1:
                #scale_factor = 2
                scale_factor = 1
            else:
                #scale_factor = 4
                scale_factor = 2
                # bilinear interpolate is nondeterministic. However the objectEstimate2 is 0.25 the size of the original image
                # Move to CPU to make it deterministic
                objectEstimate = nn.functional.interpolate(
                                    input         = objectEstimate,
                                    #size          = (intHeight, intWidth),
                                    scale_factor  = scale_factor,
                                    mode          = 'bicubic',
                                    align_corners = False)
            #objectEstimate *= self.div_flow * scale_factor
            objectEstimate *= self.div_flow
            #objectEstimate[:, 0, :, :] *= self.div_flow * intWidth / intPreprocessedWidth
            #objectEstimate[:, 1, :, :] *= self.div_flow * intHeight / intPreprocessedHeight
            return objectEstimate
    # end_forward
# end_Network


def flow_pwc2(data             = None,
              device           = 'cuda',
              use_checkpoint   = False,
              useFlow1         = False,
              useContext       = False,
              useResidual      = False,
              useDenseNet      = False,
              max_displacement = 4,
              lr_finder        = False,
              div_flow         = 1,
              batchnorm        = False,
              large_motion     = False,
              mixed_precision  = False):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)
    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    moduleNetwork = Network(device           = device,
                            use_checkpoint   = use_checkpoint,
                            useFlow1         = useFlow1,
                            useContext       = useContext,
                            useResidual      = useResidual,
                            useDenseNet      = useDenseNet,
                            max_displacement = max_displacement,
                            lr_finder        = lr_finder,
                            div_flow         = div_flow,
                            batchnorm        = batchnorm,
                            large_motion     = large_motion,
                            mixed_precision  = mixed_precision)
    if data is not None:
        if 'state_dict' in data.keys():
            moduleNetwork.load_state_dict(data['state_dict'])
    return moduleNetwork
