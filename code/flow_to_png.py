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
from data.KITTI_optical_flow import load_flow_from_png
from data.util import write_flow_png,load_flo
from data.listdataset import get_gt_correspondence_mask
from pathlib import Path
import imageio
import numpy as np

if __name__ == '__main__':
    try:
        #path = Path('F:/workspaces/FlyingThings3D_subset_flow/FlyingThings3D_subset/train/flow') 
        path = Path('F:/workspaces/FlyingThings3D_subset_flow/FlyingThings3D_subset/val/flow') 
        #imgpath = Path('F:/workspaces/FlyingThings3D_subset_image_clean/FlyingThings3D_subset/train/image_clean/left')
        dir_list = ['into_future', 'into_past']
        
        for eye_path in path.iterdir():
            if eye_path.is_dir() and eye_path.name in ['left','right']:
                print(eye_path.name)
                for direction_path in eye_path.iterdir():
                    if direction_path.is_dir() and direction_path.name in ['into_future', 'into_past']:
                        print(direction_path.name)
                        if direction_path.name == 'into_future':
                            flow_direction = '-flow_01'
                            dir_path = 'forward'
                        else:
                            flow_direction = '-flow_10'
                            dir_path = 'backward'
                        rootdir = eye_path / Path(dir_path)
                        rootdir.mkdir(exist_ok=True)
                        for flow_path in direction_path.rglob('*.flo'):
                            #print(flow_path)
                            flow = load_flo(flow_path)
                            filename =  str(rootdir / Path(flow_path.stem + flow_direction + '.png'))
                            print(filename)
                            mask = get_gt_correspondence_mask(flow)
                            write_flow_png(filename, uv=flow, mask=mask)
                            #kwargs = {'prefer_uint8': False}
                            #imageio.imwrite(filename, mask, **kwargs)
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
