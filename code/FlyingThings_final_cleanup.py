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
from pathlib import Path
from shutil import copyfile

if __name__ == '__main__':
    try:
        imgpath = Path('F:/workspaces/flyingthings3d__frames_finalpass/frames_finalpass')
                        #F:\workspaces\flyingthings3d__frames_finalpass\frames_finalpass\TRAIN\B\0727\right
        
        unused_files = Path('FlyingThings_all_unused_files.txt')
        with unused_files.open() as fp:
            line = fp.readline().rstrip()
            while line:
                image_file = imgpath / Path(line)
                if image_file.exists():
                    print('Deleting: ', image_file)
                    image_file.unlink()
                else:
                    #print('Already deleted: ', image_file)
                    pass
                line = fp.readline().rstrip()
        
        
        dest_path = Path('../../CDVD-TSP/dataset/FlyingThings3D_subset')
        dest_type = 'image_final'
        for train_path in sorted(imgpath.iterdir()):
            if train_path.is_dir() and train_path.name in ['TEST', 'TRAIN']:
                if train_path.name == 'TEST':
                    train_val = 'val'
                else:
                    train_val = 'train'
                iteration = 0
                dest_left_path = dest_path / train_val / dest_type / 'left'
                dest_left_path.mkdir(parents=True, exist_ok=True)
                dest_right_path = dest_path / train_val / dest_type / 'right'
                dest_right_path.mkdir(parents=True, exist_ok=True)
                for letter_path in sorted(train_path.iterdir()):
                    if letter_path.is_dir() and letter_path.name in ['A', 'B', 'C']:
                        for scene_path in sorted(letter_path.iterdir()):
                            if scene_path.is_dir():
                                left_path = scene_path / 'left'
                                right_path = scene_path / 'right'
                                for left_img_path in sorted(left_path.rglob('*.png')):
                                    print('img_path: ', left_img_path)
                                    dest_left_filename =  dest_left_path / Path(str(iteration).zfill(7) + '.png')
                                    print('dest_left_filename: ', dest_left_filename)
                                    copyfile(left_img_path, dest_left_filename)
                                    
                                    right_img_path = right_path / left_img_path.name
                                    dest_right_filename =  dest_right_path / Path(str(iteration).zfill(7) + '.png')
                                    copyfile(right_img_path, dest_right_filename)
                                    
                                    iteration += 1
#                                    break
#                            break
#                    break
#            break

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
