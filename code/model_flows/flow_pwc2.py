import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import math
from utils import utils
from model import correlation
import torch.utils.checkpoint as checkpoint


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
            tensorPreprocessedFirst = torch.nn.functional.interpolate(
                                        input         = tensorFirst,
                                        size          = (intPreprocessedHeight, intPreprocessedWidth),
                                        mode          = 'bilinear',
                                        align_corners = False)
            tensorPreprocessedSecond = torch.nn.functional.interpolate(
                                        input         = tensorSecond,
                                        size          = (intPreprocessedHeight, intPreprocessedWidth),
                                        mode          = 'bilinear',
                                        align_corners = False)

        # The flow returned by moduleNetwork is always one quarter the resolution of the original and is upscaled to the original
        # resolution
        """
        tensorFlow = torch.nn.functional.interpolate(
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
                                           padding_mode  = 'border',
                                           align_corners = False)
        mask = torch.ones_like(x,
                               device        = self.device,
                               requires_grad = True)
        mask = nn.functional.grid_sample(mask,
                                         grid,
                                         align_corners = False)

        mask[mask < 0.999] = 0
        mask[mask > 0] = 1

        output = output * mask

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

Backwarp_tensorGrid = {}
Backwarp_tensorPartial = {}

def Backwarp(tensorInput, tensorFlow, device='cuda'):
    """Warping layer
    Duplicate function from `run.py` from
    [A reimplementation of PWC-Net in PyTorch that matches the official Caffe version](https://github.com/sniklaus/pytorch-pwc)
    
    """
    if str(tensorFlow.shape) not in Backwarp_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0 + (1.0 / tensorFlow.shape[3]),
                                          1.0 - (1.0 / tensorFlow.shape[3]),
                                          tensorFlow.shape[3],
                                          device=device
                                         ).view(1,
                                                1,
                                                1,
                                                -1
                                               ).repeat(1,
                                                        1,
                                                        tensorFlow.shape[2],
                                                        1)
        tensorVertical = torch.linspace(-1.0  + (1.0 / tensorFlow.shape[2]),
                                        1.0 - (1.0 / tensorFlow.shape[2]),
                                        tensorFlow.shape[2],
                                        device=device
                                       ).view(1,
                                              1,
                                              -1,
                                              1
                                             ).repeat(1,
                                                      1,
                                                      1,
                                                      tensorFlow.shape[3])
        #Backwarp_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1).to(device)
        Backwarp_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1)
    # end

    if str(tensorFlow.size()) not in Backwarp_tensorPartial:
        Backwarp_tensorPartial[str(tensorFlow.size())] = tensorFlow.new_ones([tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3)])
    # end

    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                            tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)
    tensorInput = torch.cat([tensorInput, Backwarp_tensorPartial[str(tensorFlow.size())]], 1)

    tensorOutput = torch.nn.functional.grid_sample(input=tensorInput,
                                                   grid=(Backwarp_tensorGrid[str(tensorFlow.size())] + tensorFlow
                                                        ).permute(0, 2, 3, 1),
                                                   mode='bilinear',
                                                   padding_mode='zeros',
                                                   align_corners=False)
    tensorMask = tensorOutput[:, -1:, :, :];
    tensorMask[tensorMask > 0.999] = 1.0;
    tensorMask[tensorMask < 1.0] = 0.0

    return tensorOutput[:, :-1, :, :] * tensorMask
# end

##########################################################


class Extractor(torch.nn.Module):
    """The feature pyramid extractor network. The first image (t = 1) and the second image (t =2) are encoded using the same
    Siamese network. Each convolution is followed by a leaky ReLU unit. The convolutional layer and the ×2 downsampling layer at
    each level is implemented using a single convolutional layer with a stride of 2. c denotes extracted features of image t at
    level l
    """
    def __init__(self):
        super(Extractor, self).__init__()

        self.moduleOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleThr = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFou = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFiv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSix = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
    # end

    def forward(self, tensorInput):
        tensorOne = self.moduleOne(tensorInput)
        tensorTwo = self.moduleTwo(tensorOne)
        tensorThr = self.moduleThr(tensorTwo)
        tensorFou = self.moduleFou(tensorThr)
        tensorFiv = self.moduleFiv(tensorFou)
        tensorSix = self.moduleSix(tensorFiv)

        # Removed tensorOne from return because it wasn't used and isn't a multiple of 32
        return [tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix]
    # end
# end

class Decoder(torch.nn.Module):
    """The optical flow estimator network. Each convolutional layer is followed by a leaky ReLU unit except the last one that
    outputs the optical flow.
    """
    def __init__(self,
                 intLevel,
                 device         = 'cuda'):
        super(Decoder, self).__init__()
        self.device=device

        intPrevious = [None,
                       80 + 16 + 2 + 2,
                       81 + 32 + 2 + 2,
                       81 + 64 + 2 + 2,
                       81 + 96 + 2 + 2,
                       81 + 128 + 2 + 2,
                       81,
                       None][intLevel + 1]
        intCurrent = [None,
                      80 + 16 + 2 + 2,
                      81 + 32 + 2 + 2,
                      81 + 64 + 2 + 2,
                      81 + 96 + 2 + 2,
                      81 + 128 + 2 + 2,
                      81,
                      None][intLevel + 0]

        if intLevel < 6:
            self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels  = 2,
                                                         out_channels = 2,
                                                         kernel_size  = 4,
                                                         stride       = 2,
                                                         padding      = 1)
            self.moduleUpfeat = torch.nn.ConvTranspose2d(in_channels  = intPrevious + 128 + 128 + 96 + 64 + 32,
                                                         out_channels = 2,
                                                         kernel_size  = 4,
                                                         stride       = 2,
                                                         padding      = 1)
            # Start with inLevel 6(None), 5(20/32=0.625), 4(20/16=1.25), 3(20/8=2.5) and finally 2(20/4=5.0)
            # 6 doesn't have an objectPrevious so doesn't need a value. 2 is the last.
            # Add 1(20/2=10.0)
            self.dblBackwarp = [None, None, 10.0, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]

        self.moduleOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels  = intCurrent,
                            out_channels = 128,
                            kernel_size  = 3,
                            stride       = 1,
                            padding      = 1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels  = intCurrent + 128,
                            out_channels = 128,
                            kernel_size  = 3,
                            stride       = 1,
                            padding      = 1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleThr = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels  = intCurrent + 128 + 128,
                            out_channels = 96,
                            kernel_size  = 3,
                            stride       = 1,
                            padding      = 1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFou = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels  = intCurrent + 128 + 128 + 96,
                            out_channels = 64,
                            kernel_size  = 3,
                            stride       = 1,
                            padding      = 1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFiv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels  = intCurrent + 128 + 128 + 96 + 64,
                            out_channels = 32,
                            kernel_size  = 3,
                            stride       = 1,
                            padding      = 1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSix = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels  = intCurrent + 128 + 128 + 96 + 64 + 32,
                            out_channels = 2,
                            kernel_size  = 3,
                            stride       = 1,
                            padding      = 1)
        )
    # end

    def forward(self, tensorFirst, tensorSecond, objectPrevious):
        tensorFlow = None
        tensorFeat = None

        if objectPrevious is None:
            tensorFlow = None
            tensorFeat = None

            #Cost volume layer. Output is always 81 channels
            tensorVolume = torch.nn.functional.leaky_relu(
                input          = correlation.FunctionCorrelation(tensorFirst  = tensorFirst,
                                                                 tensorSecond = tensorSecond),
                negative_slope = 0.1,
                inplace        = False)

            tensorFeat = torch.cat([tensorVolume], 1)

        elif objectPrevious is not None:
            tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow']) # 2 channels
            tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat']) # reduced to 2 channels

            #Cost volume layer + Warping layer. Output is always 81 channels
            tensorVolume = torch.nn.functional.leaky_relu(
                input          = correlation.FunctionCorrelation(
                                    tensorFirst  = tensorFirst,
                                    tensorSecond = Backwarp(tensorInput = tensorSecond,
                                                            tensorFlow  = tensorFlow * self.dblBackwarp,
                                                            device      = self.device)),
                negative_slope = 0.1,
                inplace        = False)

            tensorFeat = torch.cat([tensorVolume, tensorFirst, tensorFlow, tensorFeat], 1)
        # end

        tensorFeat = torch.cat([self.moduleOne(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleTwo(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleThr(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleFou(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleFiv(tensorFeat), tensorFeat], 1)

        tensorFlow = self.moduleSix(tensorFeat)

        return {
            'tensorFlow': tensorFlow,
            'tensorFeat': tensorFeat
        }
    # end
# end

class Refiner(torch.nn.Module):
    """The context network. Each convolutional layer is followed by a leaky ReLU unit except the last one that outputs the
    optical flow. The last number in each convolutional layer denotes the dilation constant.
    """
    def __init__(self):
        super(Refiner, self).__init__()

        self.moduleMain = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels  = 81 + 16 + 2 + 2 + 128 + 128 + 96 + 64 + 32,
                            out_channels = 128,
                            kernel_size  = 3,
                            stride       = 1,
                            padding      = 1,
                            dilation     = 1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
        )
    # end

    def forward(self, tensorInput):
        return self.moduleMain(tensorInput)
    # end
# end


class Network(torch.nn.Module):
    """ Duplicate class from `run.py` of
    [A reimplementation of PWC-Net in PyTorch that matches the official Caffe version](https://github.com/sniklaus/pytorch-pwc)
    
    """
    def __init__(self,
                 device = 'cuda',
                 use_checkpoint = False):
        super(Network, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.device = device

        self.moduleExtractor = Extractor()

        self.moduleOne = Decoder(1, device=device)
        self.moduleTwo = Decoder(2, device=device)
        self.moduleThr = Decoder(3, device=device)
        self.moduleFou = Decoder(4, device=device)
        self.moduleFiv = Decoder(5, device=device)
        self.moduleSix = Decoder(6, device=device)

        self.moduleRefiner = Refiner()

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

    def forward(self, tensorFirst, tensorSecond):
        if self.use_checkpoint and self.training:
            # Using a dummy tensor to avoid the error:
            #       UserWarning: None of the inputs have requires_grad=True. Gradients will be None
            # NOTE: In case of checkpointing, if all the inputs don't require grad but the outputs do, then if the inputs are
            #       passed as is, the output of Checkpoint will be variable which don't require grad and autograd tape will
            #       break there. To get around, you can pass a dummy input which requires grad but isn't necessarily used in
            #       computation.
            dummy_tensor = torch.zeros(1, device=self.device, requires_grad=True)
            
            tensorFirst  = checkpoint.checkpoint(self.custom(self.moduleExtractor),
                                                 tensorFirst,
                                                 dummy_tensor)
            tensorSecond = checkpoint.checkpoint(self.custom(self.moduleExtractor),
                                                 tensorSecond,
                                                 dummy_tensor)

            objectEstimate = checkpoint.checkpoint(self.custom(self.moduleSix),
                                                   tensorFirst[-1],
                                                   tensorSecond[-1],
                                                   None,
                                                   dummy_tensor)
            objectEstimate = checkpoint.checkpoint(self.custom(self.moduleFiv),
                                                   tensorFirst[-2],
                                                   tensorSecond[-2],
                                                   objectEstimate,
                                                   dummy_tensor)
            objectEstimate = checkpoint.checkpoint(self.custom(self.moduleFou),
                                                   tensorFirst[-3],
                                                   tensorSecond[-3],
                                                   objectEstimate,
                                                   dummy_tensor)
            objectEstimate = checkpoint.checkpoint(self.custom(self.moduleThr),
                                                   tensorFirst[-4],
                                                   tensorSecond[-4],
                                                   objectEstimate,
                                                   dummy_tensor)
            objectEstimate = checkpoint.checkpoint(self.custom(self.moduleTwo),
                                                   tensorFirst[-5],
                                                   tensorSecond[-5],
                                                   objectEstimate,
                                                   dummy_tensor)
            objectEstimate = checkpoint.checkpoint(self.custom(self.moduleOne),
                                                   tensorFirst[-6],
                                                   tensorSecond[-6],
                                                   objectEstimate,
                                                   dummy_tensor)

            objectEstimate['tensorFeat'] = checkpoint.checkpoint(self.custom(self.moduleRefiner),
                                                                 objectEstimate['tensorFeat'],
                                                                 dummy_tensor)
        else:
            tensorFirst  = self.moduleExtractor(tensorFirst)
            tensorSecond = self.moduleExtractor(tensorSecond)
            
            objectEstimate = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None)
            objectEstimate = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate)
            objectEstimate = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate)
            objectEstimate = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate)
            objectEstimate = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate)
            objectEstimate = self.moduleOne(tensorFirst[-6], tensorSecond[-6], objectEstimate)

            objectEstimate['tensorFeat'] = self.moduleRefiner(objectEstimate['tensorFeat'])

        return objectEstimate['tensorFlow'] + objectEstimate['tensorFeat']
    # end_forward
# end_Network


def flow_pwc2(data=None, device='cuda', use_checkpoint=False):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)
    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    moduleNetwork = Network(device=device, use_checkpoint=use_checkpoint)
    if data is not None:
        moduleNetwork.load_state_dict(data['state_dict'])
    return moduleNetwork
