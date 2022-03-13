import numpy as np
import cv2 #OpenCV
import torch
import imageio


def load_flow_from_png(png_path, mask=True):
    # The -1 is here to specify not to change the image depth (16bit), and is compatible
    # with both OpenCV2 and OpenCV3
    flo_file = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    print('flo_file: ', flo_file.dtype, flo_file.shape)
    flo_img = flo_file[:, :, 2:0:-1].astype(np.float32)
    valid = (flo_file[:,:,0] == 1)
    flo_img = flo_img - 32768
    flo_img = flo_img / 64
    if mask:
        invalid = (flo_file[:,:,0] == 0) # in cv2 change the channel to the first, mask of invalid pixels
        flo_img[np.abs(flo_img) < 1e-10] = 0#1e-10
        flo_img[invalid, :] = 0
    print('flow_image: ', flo_img.dtype, flo_img.shape)
    return flo_img, valid.astype(np.uint8)


def write_flow_png(filename, uv, v=None, mask=None):
    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv
        #v = v.astype(np.float64)

    assert (u.shape == v.shape)

    height_img, width_img = u.shape
    if mask is None:
        valid_mask = np.ones([height_img, width_img])
    else:
        valid_mask = mask
    # range(-512, 512) -> range(0.0, 65535.0)
    flow_u = np.clip((u * 64 + 2 ** 15), 0.0, 65535.0).astype(np.uint16)
    flow_v = np.clip((v * 64 + 2 ** 15), 0.0, 65535.0).astype(np.uint16)
    
    #Because OpenCV uses BGR instead of RGB so the layers must be switched
    output = np.stack((valid_mask, flow_v, flow_u), axis=-1)

    """ OpenCV defaults to PNG compression level 1
    https://machinelearningknowledge.ai/tips-and-tricks-of-opencv-cv2-imwrite-that-nobody-told-you/#Managing_PNG_Image_Compression_in_cv2_imwrite
    https://docs.opencv.org/3.4/d8/d6a/group__imgcodecs__flags.html#gga292d81be8d76901bff7988d18d2b42acad2548321c69ab9c0582fd51e75ace1d0
    """
    cv2.imwrite(filename, output, [int(cv2.IMWRITE_PNG_COMPRESSION),9])

    # imageio is having difficulties writing a 16 bit png file
    #imageio.imwrite(filename, output, prefer_uint8=False)


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


def split2list(images, split, default_split=0.9):
    if isinstance(split, str):
        with open(split) as f:
            split_values = [x.strip() == '1' for x in f.readlines()]
        assert(len(images) == len(split_values))
    elif split is None:
        split_values = np.random.uniform(0,1,len(images)) < default_split
    else:
        try:
            split = float(split)
        except TypeError:
            print("Invalid Split value, it must be either a filepath or a float")
            raise
        split_values = np.random.uniform(0,1,len(images)) < split
    train_samples = [sample for sample, split in zip(images, split_values) if split]
    test_samples = [sample for sample, split in zip(images, split_values) if not split]
    return train_samples, test_samples


def center_crop(img, size):
    """
    Get the center crop of the input image
    Args:
        img: input image [HxWxC]
        size: size of the center crop (tuple) (width, height)
    Output:
        img_pad: center crop
        x, y: coordinates of the crop
    """

    if not isinstance(size, tuple):
        size = (size, size)
        #size is W,H

    img = img.copy()
    h, w = img.shape[:2]

    pad_w = 0
    pad_h = 0
    if w < size[0]:
        pad_w = np.int(np.ceil((size[0] - w) / 2))
    if h < size[1]:
        pad_h = np.int(np.ceil((size[1] - h) / 2))
    img_pad = cv2.copyMakeBorder(img,
                                 pad_h,
                                 pad_h,
                                 pad_w,
                                 pad_w,
                                 cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])
    h, w = img_pad.shape[:2]

    x1 = w // 2 - size[0] // 2
    y1 = h // 2 - size[1] // 2

    img_pad = img_pad[y1:y1 + size[1], x1:x1 + size[0], :]

    return img_pad, x1, y1


def convert_flow_to_mapping(flow, output_channel_first=True):
    if not isinstance(flow, np.ndarray):
        #torch tensor
        if len(flow.shape) == 4:
            if flow.shape[1] != 2:
                # size is BxHxWx2
                flow = flow.permute(0, 3, 1, 2)

            B, C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if flow.is_cuda:
                grid = grid.cuda()
            map = flow + grid # here also channel first
            if not output_channel_first:
                map = map.permute(0,2,3,1)
        else:
            if flow.shape[0] != 2:
                # size is HxWx2
                flow = flow.permute(2, 0, 1)

            C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if flow.is_cuda:
                grid = grid.cuda()
            map = flow + grid # here also channel first
            if not output_channel_first:
                map = map.permute(1,2,0).float()
        return map.float()
    else:
        # here numpy arrays
        if len(flow.shape) == 4:
            if flow.shape[3] != 2:
                # size is Bx2xHxW
                flow = flow.permute(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = flow.shape[:3]
            map = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                map[i, :, :, 0] = flow[i, :, :, 0] + X
                map[i, :, :, 1] = flow[i, :, :, 1] + Y
            if output_channel_first:
                map = map.transpose(0,3,1,2)
        else:
            if flow.shape[0] == 2:
                # size is 2xHxW
                flow = flow.permute(1,2,0)
            # HxWx2
            h_scale, w_scale = flow.shape[:2]
            map = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            map[:,:,0] = flow[:,:,0] + X
            map[:,:,1] = flow[:,:,1] + Y
            if output_channel_first:
                map = map.transpose(2,0,1).float()
        return map.astype(np.float32)


def convert_mapping_to_flow(map, output_channel_first=True):
    if not isinstance(map, np.ndarray):
        # torch tensor
        if len(map.shape) == 4:
            if map.shape[1] != 2:
                # size is BxHxWx2
                map = map.permute(0, 3, 1, 2)

            B, C, H, W = map.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if map.is_cuda:
                grid = grid.cuda()
            flow = map - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(0,2,3,1)
        else:
            if map.shape[0] != 2:
                # size is HxWx2
                map = map.permute(2, 0, 1)

            C, H, W = map.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if map.is_cuda:
                grid = grid.cuda()

            flow = map - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(1,2,0).float()
        return flow.float()
    else:
        # here numpy arrays
        if len(map.shape) == 4:
            if map.shape[3] != 2:
                # size is Bx2xHxW
                map = map.permute(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = map.shape[:3]
            flow = np.copy(map)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                flow[i, :, :, 0] = map[i, :, :, 0] - X
                flow[i, :, :, 1] = map[i, :, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(0,3,1,2)
        else:
            if map.shape[0] == 2:
                # size is 2xHxW
                map = map.permute(1,2,0)
            # HxWx2
            h_scale, w_scale = map.shape[:2]
            flow = np.copy(map)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            flow[:,:,0] = map[:,:,0]-X
            flow[:,:,1] = map[:,:,1]-Y
            if output_channel_first:
                flow = flow.transpose(2,0,1).float()
        return flow.astype(np.float32)