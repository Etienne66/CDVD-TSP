import torch
import torch.nn.functional as F
from imageio import imread, imwrite
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms


def Backwarp(tensorInput, tensorFlow, requires_grad=False):
    """Warping layer
    Duplicate function from `run.py` from
    [A reimplementation of PWC-Net in PyTorch that matches the official Caffe version](https://github.com/sniklaus/pytorch-pwc)
    
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
    tensorVertical = torch.linspace(-1.0  + (1.0 / tensorFlow.shape[2]),
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
    #This is used to make the mask
    Backwarp_tensorPartial = tensorFlow.new_ones([tensorFlow.shape[0],
                                                  1,
                                                  tensorFlow.shape[2],
                                                  tensorFlow.shape[3]],
                                                  requires_grad=requires_grad)
    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.shape[3] - 1.0) / 2.0),
                            tensorFlow[:, 1:2, :, :] / ((tensorInput.shape[2] - 1.0) / 2.0)], 1)
    # Add the mask as the last channel of the input
    tensorInput = torch.cat([tensorInput, Backwarp_tensorPartial], 1)
    print('min {}, max {}'.format(torch.min(Backwarp_tensorGrid), torch.max(Backwarp_tensorGrid)))
    print('min {}, max {}'.format(torch.min(Backwarp_tensorGrid + tensorFlow), torch.max(Backwarp_tensorGrid + tensorFlow)))

    #Move to CPU to make it deterministic
    #Changed padding_mode from zeros to border because we shouldn't assume that the motion stops at the border
    #bicubic
    tensorOutput = F.grid_sample(input         = tensorInput,
                                 grid          = (Backwarp_tensorGrid + tensorFlow).permute(0, 2, 3, 1),
                                 mode          = 'bicubic',
                                 padding_mode  = 'zeros',
                                 align_corners = False)
    print('min {}, max {}'.format(torch.min(tensorOutput), torch.max(tensorOutput)))
    print('min {}, max {}'.format(torch.min(tensorOutput[:, -1:, :, :]), torch.max(tensorOutput[:, -1:, :, :])))
    # Extract the mask from the last channel of the output
    tensorMask = tensorOutput[:, -1:, :, :];
    tensorMask[tensorMask < 0.9999] = 0;
    tensorMask[tensorMask > 0] = 1
    # Filter the mask from the tensor output and apply the mask to the remaining channels.
    return tensorOutput[:, :-1, :, :] * tensorMask
    #return tensorOutput[:, :-1, :, :]


##########################################################
def backwarp(tenIn, tenFlow, requires_grad=False):
    tenHor = torch.linspace(start         = -1.0,
                            end           = 1.0,
                            steps         = tenFlow.shape[3],
                            dtype         = tenFlow.dtype,
                            device        = tenFlow.device,
                            requires_grad = requires_grad).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
    tenVer = torch.linspace(start         = -1.0,
                            end           = 1.0,
                            steps         = tenFlow.shape[2],
                            dtype         = tenFlow.dtype,
                            device        = tenFlow.device,
                            requires_grad = requires_grad).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

    backwarp_tenGrid = torch.cat([tenHor, tenVer], 1)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenIn.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenIn.shape[2] - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tenIn, grid=(backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1), mode='bicubic', padding_mode='zeros', align_corners=True)
# end


def load_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)

    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (h, w, 2))
    return data2D

if __name__ == '__main__':
    input_transform = transforms.Compose([
        transforms.ToTensor()#,
        #transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]), # (0,255) -> (0,1)
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # from ImageNet dataset
    ])
    
    png_normalize = transforms.Compose([
        #transforms.Normalize(mean= [0., 0., 0.], std = [1/0.229, 1/0.224, 1/0.225]),
        #transforms.Normalize(mean= [-0.485, -0.456, -0.406], std = [1., 1., 1.]),
        transforms.Normalize(mean=[0, 0, 0], std=[1/255, 1/255, 1/255]) # (0,1) -> (0,255)
    ])
    
    
    img1 = imread(Path('../datasets/MPI_Sintel/training/final/ambush_5/frame_0011.png'))
    #print('img1.shape.1',img1.shape)
    #imwrite('./ambush_5.frame_010.1.png', img1)
    flo1 = torch.from_numpy(load_flo(Path('../datasets/MPI_Sintel/training/flow/ambush_5/frame_0010.flo'))).permute((2, 0, 1))
    img1 = input_transform(img1)
    img1 = img1[None,:,:,:]
    flo1 = flo1[None,:,:,:]
    #print('img1.shape.2',img1.shape)
    #print('flo1.shape',flo1.shape)
    imgout = Backwarp(img1, flo1)
    #imgout = backwarp(img1, flo1)
    imgout = imgout[0,:,:,:]
    img1in = img1[0,:,:,:]
    imgout = png_normalize(imgout)
    img1in = png_normalize(img1in)
    imgout = imgout.permute((1, 2, 0)).numpy().astype(np.uint8)
    img1in = img1in.permute((1, 2, 0)).numpy().astype(np.uint8)
    #print('img1in.shape.2',img1in.shape)
    
    imwrite('./ambush_5.frame_010.png', img1in)
    imwrite('./ambush_5.frame_011.png', imgout)

