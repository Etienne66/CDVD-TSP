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
        dataset = Path('F:/workspaces/FlyingThings3D_subset_flow/FlyingThings3D_subset') 
        root_dir = Path('F:/workspaces/CDVD-TSP/datasets/FlyingThings3D_subset')
        
        for train_path in dataset.iterdir():
            flow_path = train_path / 'flow'
            for eye_path in flow_path.iterdir():
                if eye_path.is_dir() and eye_path.name in ['left','right']:
                    print(eye_path.name)
                    for direction_path in eye_path.iterdir():
                        if direction_path.is_dir() and direction_path.name in ['into_future', 'into_past']:
                            print(direction_path.name)
                            #if direction_path.name == 'into_future':
                            #    flow_direction = '-flow_01'
                            #    dir_path = 'forward'
                            #else:
                            #    flow_direction = '-flow_10'
                            #    dir_path = 'backward'
                            rootdir = root_dir / train_path.name / 'flow' / eye_path.name / direction_path.name
                            rootdir.mkdir(parents=True, exist_ok=True)
                            for flow_path in direction_path.rglob('*.flo'):
                                #print(flow_path)
                                flow = load_flo(flow_path)
                                h,w = flow[:,:,0].shape
                                # Make sure neither vector exceeds the maximum for the KITTI format of 512 pixels stored in 16 bits
                                #if np.amin(flow[:,:,0]) < -511 or np.amax(flow[:,:,0]) > 512 :
                                #    print(flow_path, ' has h of between ', np.amin(flow[:,:,0]), ' and ', np.amax(flow[:,:,0]))
                                #    exit()
                                #if np.amin(flow[:,:,1]) < -511 or np.amax(flow[:,:,1]) > 512 :
                                #    print(flow_path, ' has w of between ', np.amin(flow[:,:,1]), ' and ', np.amax(flow[:,:,1]))
                                #    exit()
                                filename =  rootdir / (flow_path.stem + '.npz')
                                print(filename)
                                #f = gzip.GzipFile(filename, "w")
                                mask = get_gt_correspondence_mask(flow)
                                #np.savez(file=filename, u=flow[:,:,0], v=flow[:,:,1], mask=mask)
                                #np.savez_compressed(filename, u=flow[:,:,0], v=flow[:,:,1], mask=mask)
                                zipf = zipfile.ZipFile(filename, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9)
                                with zipf.open('u.npy', mode='w', force_zip64=True) as f:
                                    np.lib.format.write_array(f, np.asanyarray(flow[:,:,0]))
                                with zipf.open('v.npy', mode='w', force_zip64=True) as f:
                                    np.lib.format.write_array(f, np.asanyarray(flow[:,:,1]))
                                #with zipf.open('flow.npy', mode='w', force_zip64=True) as f:
                                #    np.lib.format.write_array(f, np.asanyarray(flow))
                                with zipf.open('mask.npy', mode='w', force_zip64=True) as f:
                                    np.lib.format.write_array(f, np.asanyarray(mask))
                                zipf.close()
                                #with gzip.open(filename, "rb") as f:
                                #    file_content = f.read()
                                #with gzip.open(filename, "wb") as f:
                                #    #f.write(file_content)
                                #    np.savez(file=f, u=flow[:,:,0], v=flow[:,:,1], mask=mask)
                                #np.savez_compressed(filename, u=flow[:,:,0], v=flow[:,:,1], mask=mask)
                                #write_flow_png(filename, uv=flow, mask=mask)
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
