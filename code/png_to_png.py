""" Converts png to png to save disk space and speed up loading
This wasn't necessary. The images were already well compressed.
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
import cv2

if __name__ == '__main__':
    try:
        imgpath = Path('F:/workspaces/FlyingThings3D_subset_image_clean/FlyingThings3D_subset/train/image_clean')
        dir_list = ['left', 'right']
        
        for eye_path in imgpath.iterdir():
            if eye_path.is_dir() and eye_path.name in ['left', 'right']:
                print(eye_path.name)
                #rootdir = eye_path / Path(eye_path.name + '_png')
                #rootdir.mkdir(exist_ok=True)
                for img_path in eye_path.rglob('*.png'):
                    if img_path.stem.isnumeric():
                        filename =  str(eye_path / Path(img_path.stem + '-img.png'))
                        print(filename)
                        image1 = imageio.imread(img_path)
                        imageio.imwrite(filename, image1)
                        #cv2.imwrite(filename, image1, [int(cv2.IMWRITE_PNG_COMPRESSION),9])
                        
                        #imageio.imwrite(filename, mask)
                        #exit()
                        break
                break

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
