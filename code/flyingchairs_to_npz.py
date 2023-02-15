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
        dataset = Path('F:/workspaces/FlyingChairs2') 
        root_dir = Path('F:/workspaces/CDVD-TSP/datasets/FlyingChairs2')
        
        for train_path in dataset.iterdir():
            print(train_path.name)
            rootdir = root_dir / train_path.name
            rootdir.mkdir(parents=True, exist_ok=True)
            for flow_path in train_path.rglob('*.flo'):
                flow = load_flo(flow_path)
                h,w = flow[:,:,0].shape
                filename =  rootdir / (flow_path.stem + '.npz')
                print(filename)
                mask = get_gt_correspondence_mask(flow)
                zipf = zipfile.ZipFile(filename, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9)
                with zipf.open('u.npy', mode='w', force_zip64=True) as f:
                    np.lib.format.write_array(f, np.asanyarray(flow[:,:,0]))
                with zipf.open('v.npy', mode='w', force_zip64=True) as f:
                    np.lib.format.write_array(f, np.asanyarray(flow[:,:,1]))
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
