import os.path
import glob
from .listdataset import ListDataset
from .util import split2list


def load_flow_from_png(png_path):
    # The -1 is here to specify not to change the image depth (16bit), and is compatible
    # with both OpenCV2 and OpenCV3
    flo_file = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    flo_img = flo_file[:, :, 2:0:-1].astype(np.float32)
    valid = (flo_file[:,:,0] == 1)
    flo_img = flo_img - 32768
    flo_img = flo_img / 64
    return flo_img, valid.astype(np.uint8)


def Chairs_flow_loader(root, path_imgs, path_flo):
    #print(path_flo)
    flow, mask = load_flow_from_png(path_flo)
    return [imageio.imread(img).astype(np.float32) for img in path_imgs], flow, mask
    #return [cv2.imread(str(img))[:,:,::-1].astype(np.float32) for img in path_imgs], flow, mask


def make_dataset(root, train=True):
    '''Will search for triplets that go by the pattern '[name]_img1.ppm  [name]_img2.ppm    [name]_flow.flo' '''
    if train:
        flow_path = root / 'FlyingChairs2/train/flow_png'
        img_path = root / 'FlyingChairs2/train/image2'
    else:
        flow_path = root / 'FlyingChairs2/val/flow_png'
        img_path = root / 'FlyingChairs2/val/image2'

    images = []
    for flow_map in flow_path.rglob('*-flow_01.png'):
        f = flow_map.stem.split('-')
        root_filename = f[0]

        img1 = img_path / (root_filename+'-img_0.png')
        img2 = img_path / (root_filename+'-img_1.png')
        if not (    img1.is_file()
                and img2.is_file()):
            continue
        images.append([[img1,img2],flow_map])

        flow_map = flow_path / (root_filename+'-flow_10.png')
        img1 = img_path / (root_filename+'-img_1.png')
        img2 = img_path / (root_filename+'-img_0.png')
        if not (    img1.is_file()
                and img2.is_file()
                and flow_map.is_file()):
            continue
        images.append([[img1,img2],flow_map])

    return images


def flying_chairs2(root,
                   transform=None,
                   target_transform=None,
                   co_transform=None,
                   train=True):
    train_list = make_dataset(root, train=train)
    train_dataset = ListDataset(root, train_list, transform=transform,
                                target_transform=target_transform, co_transform=co_transform,
                                loader=Chairs_flow_loader, mask=True)

    return train_dataset
