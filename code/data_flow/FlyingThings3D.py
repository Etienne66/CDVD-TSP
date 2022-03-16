"""
@InProceedings{MIFDB16,
  author    = "N. Mayer and E. Ilg and P. H{\"a}usser and P. Fischer and D. Cremers and A. Dosovitskiy and T. Brox",
  title     = "A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation",
  booktitle = "IEEE International Conference on Computer Vision and Pattern Recognition (CVPR)",
  year      = "2016",
  note      = "arXiv:1512.02134",
  url       = "http://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16"
}
"""
from pathlib import Path
import glob
from .listdataset import ListDataset
from .util import split2list



def load_flow_from_png(png_path):
    # The -1 is here to specify not to change the image depth (16bit), and is compatible
    # with both OpenCV2 and OpenCV3
    flo_file = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    flo_img = flo_file[:, :, 2:0:-1].astype(np.float32)
    valid = (flo_file[:,:,0] == 1)
    flo_img = flo_img - 32768
    flo_img = flo_img / 64
    return flo_img, valid.astype(np.uint8)


def make_dataset(dataset, train=True, final=True):
    '''Will search for triplets that go by the pattern
        '[int].png  [int+1].ppm    [name]_flow01.png' forward flow
    or  '[int].png  [int-1].ppm    [name]_flow10.png' backward flow
    '''

    if train:
        flow_path = dataset / 'FlyingThings3D_subset/train/flow'
        if final:
            img_path = dataset / 'FlyingThings3D_subset/train/image_final'
        else:
            img_path = dataset / 'FlyingThings3D_subset/train/image_clean'
    else:
        flow_path = dataset / 'FlyingThings3D_subset/val/flow'
        if final:
            img_path = dataset / 'FlyingThings3D_subset/val/image_final'
        else:
            img_path = datasets / 'FlyingThings3D_subset/val/image_clean'

    images = []

    for eye_path in sorted(flow_path.iterdir()):
        if eye_path.is_dir() and eye_path.name in ['left','right']:
            direction_path = eye_path / 'forward'
            for flow_map in direction_path.rglob('*-flow_01.png'):
                f = flow_map.stem.split('-')
                assert(f[0].isnumeric())
                iteration = int(f[0])
                img1 = str(iteration).zfill(7)+'.png'
                img2 = str(iteration+1).zfill(7)+'.png'
                img1_path = img_path / eye_path.name / img1
                img2_path = img_path / eye_path.name / img2
                if not (    img1_path.is_file()
                        and img2_path.is_file()):
                    continue
                images.append([[img1_path,img2_path],flow_map])

            direction_path = eye_path / 'backward'
            for flow_map in direction_path.rglob('*-flow_10.png'):
                f = flow_map.stem.split('-')
                assert(f[0].isnumeric())
                iteration = int(f[0])
                img1 = str(iteration).zfill(7)+'.png'
                img2 = str(iteration-1).zfill(7)+'.png'
                img1_path = img_path / eye_path.name / img1
                img2_path = img_path / eye_path.name / img2
                if not (    img1_path.is_file()
                        and img2_path.is_file()):
                    continue
                images.append([[img1_path,img2_path],flow_map])
    return images


def Things_flow_loader(root, path_imgs, path_flo):
    flow, mask = load_flow_from_png(path_flo)
    return [imread(img).astype(np.uint8) for img in path_imgs], flow, mask


def flying_things_clean(root,
                        transform=None,
                        target_transform=None,
                        co_transform=None,
                        train=True,
                        lr_finder=False):
    train_list = make_dataset(root, train=train, final=False)
    train_dataset = ListDataset(root, train_list, transform=transform,
                                target_transform=target_transform, co_transform=co_transform,
                                loader=Things_flow_loader, mask=True, lr_finder=False)

    return train_dataset


def flying_things_final(root,
                        transform=None,
                        target_transform=None,
                        co_transform=None,
                        train=True):
    train_list = make_dataset(root, train=train)
    train_dataset = ListDataset(root, train_list, transform=transform,
                                target_transform=target_transform, co_transform=co_transform,
                                loader=Things_flow_loader, mask=True)

    return train_dataset


def flying_things_both(root,
                       transform=None,
                       target_transform=None,
                       co_transform=None,
                       train=True,
                       lr_finder=False):
    train_list = make_dataset(root, train=train)
    train_list2 = make_dataset(root, train=train, final=False)
    train_dataset = ListDataset(root, train_list, transform=transform,
                                target_transform=target_transform, co_transform=co_transform,
                                loader=Things_flow_loader, mask=True)

    return train_dataset

