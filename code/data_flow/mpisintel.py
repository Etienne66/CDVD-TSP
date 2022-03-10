import os.path
import glob
from .listdataset import ListDataset
from .util import split2list
from data import flow_transforms

'''
Dataset routines for MPI Sintel.
http://sintel.is.tue.mpg.de/
clean version imgs are without shaders, final version imgs are fully rendered
The dataset is not very big, you might want to only pretrain on it for flownet
'''
def make_dataset(root, dataset_type='clean', train=True):
    flow_dir = 'flow'
    train_test_dir = 'training' if train else 'test'
    assert((root / train_test_dir / flow_dir).is_dir())
    img_dir = dataset_type
    assert((root / train_test_dir / img_dir).is_dir())

    images = []
    for scene_dir in (root / train_test_dir / flow_dir).iterdir():
        if scene_dir.is_dir():
            for flow_map in scene_dir.rglob('*.flo'):
                f = flow_map.stem.split('_')
                prefix = f[0]
                frame_nb = int(f[1])
                img1 = root / train_test_dir / img_dir / scene_dir.name / '{}_{:04d}.png'.format(prefix, frame_nb)
                img2 = root / train_test_dir / img_dir / scene_dir.name / '{}_{:04d}.png'.format(prefix, frame_nb + 1)
                if not (    img1.is_file()
                        and img2.is_file()):
                    continue
                images.append([[img1,img2],flow_map])
    return images


def mpi_sintel_clean(root,
                     transform=None,
                     target_transform=None,
                     co_transform=None,
                     train=True):
    train_list = make_dataset(root / 'MPI_Sintel', 'clean', train=train)
    train_dataset = ListDataset(root / 'MPI_Sintel', train_list, transform, target_transform, co_transform)
    #test_dataset = ListDataset(root, test_list, transform, target_transform, flow_transforms.CenterCrop((384,1024)))

    return train_dataset#, test_dataset


def mpi_sintel_final(root,
                     transform=None,
                     target_transform=None,
                     co_transform=None,
                     train=True):
    train_list = make_dataset(root / 'MPI_Sintel', 'final', train=train)
    train_dataset = ListDataset(root / 'MPI_Sintel', train_list, transform, target_transform, co_transform)
    #test_dataset = ListDataset(root, test_list, transform, target_transform, flow_transforms.CenterCrop((384,1024)))

    return train_dataset#, test_dataset


def mpi_sintel_both(root,
                    transform=None,
                    target_transform=None,
                    co_transform=None,
                    train=True):
    '''load images from both clean and final folders.
    We cannot shuffle input, because it would very likely cause data snooping
    for the clean and final frames are not that different'''
    train_list1 = make_dataset(root / 'MPI_Sintel', 'clean', train=train)
    train_list2 = make_dataset(root / 'MPI_Sintel', 'final', train=train)
    train_dataset = ListDataset(root / 'MPI_Sintel', train_list1 + train_list2, transform, target_transform, co_transform)
    #test_dataset = ListDataset(root, test_list1 + test_list2, transform, target_transform, flow_transforms.CenterCrop((384,1024)))

    return train_dataset#, test_dataset
