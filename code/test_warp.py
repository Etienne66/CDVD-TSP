import torch
import torch.nn.functional as F
from imageio.v2 import imread, imwrite
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
    #print('min {}, max {}'.format(torch.min(Backwarp_tensorGrid), torch.max(Backwarp_tensorGrid)))
    #print('min {}, max {}'.format(torch.min(Backwarp_tensorGrid + tensorFlow), torch.max(Backwarp_tensorGrid + tensorFlow)))

    #Move to CPU to make it deterministic
    #Changed padding_mode from zeros to border because we shouldn't assume that the motion stops at the border
    #bicubic
    tensorOutput = F.grid_sample(input         = tensorInput,
                                 grid          = (Backwarp_tensorGrid + tensorFlow).permute(0, 2, 3, 1),
                                 mode          = 'bicubic',
                                 padding_mode  = 'zeros',
                                 align_corners = False)
    #print('min {}, max {}'.format(torch.min(tensorOutput), torch.max(tensorOutput)))
    #print('min {}, max {}'.format(torch.min(tensorOutput[:, -1:, :, :]), torch.max(tensorOutput[:, -1:, :, :])))
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

def readFlowNPZ(filename):
    with np.load(filename) as data: #, allow_pickle=True, encoding='bytes', fix_imports=True
        flow_u = data['u']#.astype(np.float32)
        flow_v = data['v']#.astype(np.float32)
        if 'mask' in data.keys():
            valid = data['mask'].astype(bool)
        else:
            h,w = flow_u.shape
            #print('h: ',h,', w: ',w)
            invalid = np.logical_or(np.logical_or(np.isnan(flow_u), np.isnan(flow_v)),
                                    np.logical_or(np.isinf(flow_u), np.isinf(flow_v)))
            flow_u[invalid] = 0
            flow_v[invalid] = 0
            invalid = np.logical_or(invalid,
                                    np.logical_or(np.abs(flow_u) >= w,
                                                  np.abs(flow_v) >= h))
            flow_u[invalid] = 0
            flow_v[invalid] = 0
            valid = np.logical_not(invalid)
    flow = np.stack((flow_u, flow_v), axis=-1)
    return flow, valid
    
if __name__ == '__main__':
    things = False
    viper = False
    sintel = False
    erase = True
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

    if erase:
        #img1 = imread(Path('../datasets/MPI_Sintel/training/final/ambush_5/frame_0011.png'))
        #img1 = [:,:,:]
        img1 = np.array([[[1,1,1,1,1],
                          [1,1,1,1,1],
                          [1,1,1,1,1],
                          [1,1,1,1,1]],
                         [[2,2,2,2,2],
                          [2,2,2,2,2],
                          [2,2,2,2,2],
                          [2,2,2,2,2]],
                         [[3,3,3,3,3],
                          [3,3,3,3,3],
                          [3,3,3,3,3],
                          [3,3,3,3,3]]]) #.transpose(1, 2, 0)
        # Numpy HWC -> CHW Torch
        print('img1',img1.shape)
        print('img1',img1)
        print('img1.reshape',img1.reshape(-1, 3).shape)
        print('img1.reshape',img1.reshape(-1, 3))
        print('img1.mean',np.mean(img1.reshape(-1, 3), axis=0).shape)
        print('img1.mean',np.mean(img1.reshape(-1, 3), axis=0))
        
        #if torch.is_tensor(img2):
        #    _, ht, wd = img2.size()
        #else:
        #    ht, wd = img2.shape[:2]
        #if np.random.rand() < self.eraser_aug_prob:
        #    #print('img1.size',img1.shape)
        #    #print('img1.reshape(-1, 3).size',img1.reshape(-1, 3).shape)
        #    #exit()
        #    if torch.is_tensor(img2):
        #        mean_colorF = torch.mean(img1.reshape(-1, 3), axis=0)
        #    else:
        #        mean_colorF = np.mean(img1.reshape(-1, 3), axis=0)
        #    for _ in range(np.random.randint(1, 3)):
        #        x0 = np.random.randint(0, wd)
        #        y0 = np.random.randint(0, ht)
        #        dx = np.random.randint(50, 100)
        #        dy = np.random.randint(50, 100)
        #        if torch.is_tensor(img1):
        #            img1[0, y0:y0+dy, x0:x0+dx] = mean_colorF[0]
        #            img1[1, y0:y0+dy, x0:x0+dx] = mean_colorF[1]
        #            img1[2, y0:y0+dy, x0:x0+dx] = mean_colorF[2]
        #        else:
        #            img1[y0:y0+dy, x0:x0+dx, :] = mean_colorF
        #    for _ in range(np.random.randint(1, 3)):
        #        x0 = np.random.randint(0, wd)
        #        y0 = np.random.randint(0, ht)
        #        dx = np.random.randint(50, 100)
        #        dy = np.random.randint(50, 100)
        #        if torch.is_tensor(img3):
        #            img3[0, y0:y0+dy, x0:x0+dx] = mean_colorF[0]
        #            img3[1, y0:y0+dy, x0:x0+dx] = mean_colorF[1]
        #            img3[2, y0:y0+dy, x0:x0+dx] = mean_colorF[2]
        #        else:
        #            img3[y0:y0+dy, x0:x0+dx, :] = mean_colorB    
    
    if sintel:
        img1 = imread(Path('../datasets/MPI_Sintel/training/final/ambush_5/frame_0011.png'))
        #print('img1.shape.1',img1.shape)
        #imwrite('./ambush_5.frame_010.1.png', img1)
        flo1 = torch.from_numpy(load_flo(Path('../datasets/MPI_Sintel/training/flow/ambush_5/frame_0010.flo'))).permute((2, 0, 1))
        #flow, mask = readFlowNPZ(Path('../datasets/FlyingThings3D_subset/train/flow/left/into_past/0000001.npz'))
        #flow, mask = readFlowNPZ(Path('../datasets/FlyingThings3D_subset/train/flow/left/into_future/0000001.npz'))
        #flo1 = torch.from_numpy(flow).permute((2, 0, 1))
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
        
        imwrite('./frame_0011.png', img1in)
        imwrite('./frame_0010.png', imgout)
    if things:
        img1 = imread(Path('../datasets/FlyingThings3D_subset/train/image_clean/left/0000002.png'))
        flow, mask = readFlowNPZ(Path('../datasets/FlyingThings3D_subset/train/flow/left/into_future/0000001.npz'))
        flo1 = torch.from_numpy(flow*mask[:,:,None]).permute((2, 0, 1))
        img1 = input_transform(img1)
        img1 = img1[None,:,:,:]
        flo1 = flo1[None,:,:,:]
        imgout = Backwarp(img1, flo1)
        #imgout = backwarp(img1, flo1)
        imgout = imgout[0,:,:,:]
        img1in = img1[0,:,:,:]
        imgout = png_normalize(imgout)
        img1in = png_normalize(img1in)
        imgout = imgout.permute((1, 2, 0)).numpy().astype(np.uint8)
        img1in = img1in.permute((1, 2, 0)).numpy().astype(np.uint8)
        #print('img1in.shape.2',img1in.shape)
        
        imwrite('./a0000002.png', img1in)
        imwrite('./a0000001.fw.png', imgout)

        img1 = imread(Path('../datasets/FlyingThings3D_subset/train/image_clean/left/0000000.png'))
        flow, mask = readFlowNPZ(Path('../datasets/FlyingThings3D_subset/train/flow/left/into_past/0000001.npz'))
        flo1 = torch.from_numpy(flow*mask[:,:,None]).permute((2, 0, 1))
        img1 = input_transform(img1)
        img1 = img1[None,:,:,:]
        flo1 = flo1[None,:,:,:]
        imgout = Backwarp(img1, flo1)
        #imgout = backwarp(img1, flo1)
        imgout = imgout[0,:,:,:]
        img1in = img1[0,:,:,:]
        imgout = png_normalize(imgout)
        img1in = png_normalize(img1in)
        imgout = imgout.permute((1, 2, 0)).numpy().astype(np.uint8)
        img1in = img1in.permute((1, 2, 0)).numpy().astype(np.uint8)
        
        imwrite('./a0000000.png', img1in)
        imwrite('./a0000001.bw.png', imgout)
    if viper:
        #E:\workspaces\CDVD-TSP\datasets\VIPER\train\img\001
        img0 =            imread(Path('../datasets/VIPER/train/img/001/001_00001.png'))
        flow, mask = readFlowNPZ(Path('../datasets/VIPER/train/flowbw/001/001_00002.npz'))
        flow = torch.from_numpy(flow*mask[:,:,None]).permute((2, 0, 1))
        img0 = input_transform(img0)
        img0 = img0[None,:,:,:]
        flow = flow[None,:,:,:]
        imgout = Backwarp(img0, flow)
        imgout = imgout[0,:,:,:]
        img1in = img0[0,:,:,:]
        img1in = png_normalize(img1in)
        imgout = png_normalize(imgout).clip(min=0,max=255)
        img1in = img1in.permute((1, 2, 0)).numpy().astype(np.uint8)
        imgout = imgout.permute((1, 2, 0)).numpy().astype(np.uint8)
        
        imwrite('./001_00001.png', img1in)
        imwrite('./001_00002.bw.png', imgout)

        img2 =            imread(Path('../datasets/VIPER/train/img/001/001_00003.png'))
        flow, mask = readFlowNPZ(Path('../datasets/VIPER/train/flow/001/001_00002.npz'))
        flow = torch.from_numpy(flow*mask[:,:,None]).permute((2, 0, 1))
        img2 = input_transform(img2)
        img2 = img2[None,:,:,:]
        flow = flow[None,:,:,:]
        imgout = Backwarp(img2, flow)
        imgout = imgout[0,:,:,:]
        img1in = img2[0,:,:,:]
        img1in = png_normalize(img1in)
        imgout = png_normalize(imgout).clip(min=0,max=255)
        img1in = img1in.permute((1, 2, 0)).numpy().astype(np.uint8)
        imgout = imgout.permute((1, 2, 0)).numpy().astype(np.uint8)
        
        imwrite('./001_00003.png', img1in)
        imwrite('./001_00002.fw.png', imgout)
