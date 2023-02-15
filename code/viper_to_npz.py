""" Converts flo to png to save disk space and speed up loading
"""
__author__ = "Steven Wilson"
__copyright__ = "Copyright 2022"
__credits__ = ["Steven Wilson"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Steven Wilson"
__email__ = "Etienne66"
__status__ = "Development"

import platform
import os
import traceback
import winsound
import imageio
import numpy as np
import zipfile
#from data_flow.KITTI_optical_flow import load_flow_from_png
from data_flow.util import write_flow_png,load_flo
from data_flow.listdataset import get_gt_correspondence_mask
from pathlib import Path

if __name__ == '__main__':
    try:
        dataset = Path('F:/workspaces/VIPER/The rest/train/flow') 
        root_dir = Path('E:/workspaces/CDVD-TSP/datasets/VIPER/train/flow_new')
        for train_path in dataset.iterdir():
            #print('train_path',train_path.name)
            rootdir = root_dir / train_path.name
            rootdir.mkdir(parents=True, exist_ok=True)
            for flow_path in train_path.rglob('*.npz'):
                #Skip files that are empty like 041_01849.npz
                if flow_path.stat().st_size == 0:
                    continue
                filename =  rootdir / (flow_path.stem + '.npz')
                #Skip files that have already been created
                if filename.exists():
                    if filename.stat().st_size > 0:
                        continue
                print(filename)
                with np.load(flow_path) as data: #, allow_pickle=True, encoding='bytes', fix_imports=True
                    flow_u = data['u']#.astype(np.float32)
                    flow_v = data['v']#.astype(np.float32)
                h,w = flow_u.shape
                flow_u_mask = flow_u
                flow_v_mask = flow_v
                #print('h: ',h,', w: ',w)
                invalid = np.logical_or(np.logical_or(np.isnan(flow_u_mask), np.isnan(flow_v_mask)),
                                        np.logical_or(np.isinf(flow_u_mask), np.isinf(flow_v_mask)))
                flow_u_mask[invalid] = 0
                flow_v_mask[invalid] = 0
                invalid = np.logical_or(invalid, np.logical_or(np.abs(flow_u_mask) >= w, np.abs(flow_v_mask) >= h))
                flow_u_mask[invalid] = 0
                flow_v_mask[invalid] = 0
                mask = np.logical_not(invalid)
                zipf = zipfile.ZipFile(filename, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9)
                with zipf.open('u.npy', mode='w', force_zip64=True) as f:
                    np.lib.format.write_array(f, np.asanyarray(flow_u))
                with zipf.open('v.npy', mode='w', force_zip64=True) as f:
                    np.lib.format.write_array(f, np.asanyarray(flow_v))
                with zipf.open('mask.npy', mode='w', force_zip64=True) as f:
                    np.lib.format.write_array(f, np.asanyarray(mask))
                zipf.close()
                #exit()
                #break
            #break
    #break

        if platform.system() == 'Windows':
            winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
        elif platform.system() == 'Darwin':
            os.system('say "Your CDVD-TSP program has finished"')
        elif platform.system() == 'Linux':
            os.system('spd-say "Your CDVD-TSP program has finished"')

    except:
        traceback.print_exc()
        if platform.system() == 'Windows':
            winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
        elif platform.system() == 'Darwin':
            os.system('say "Your CDVD-TSP program has crashed"')
        elif platform.system() == 'Linux':
            os.system('spd-say "Your CDVD-TSP program has crashed"')
        #exit()

    finally:
        if platform.system() == 'Windows':
            env = 'CUDA_LAUNCH_BLOCKING'
            if env in os.environ:
                del os.environ[env]
        print("Closed cleanly")
