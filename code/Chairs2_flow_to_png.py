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
from data_flow.util import write_flow_png,load_flo,load_flow_from_png
from data_flow.listdataset import get_gt_correspondence_mask
from pathlib import Path
import imageio
import numpy as np

if __name__ == '__main__':
    try:
        path = Path('../../FlyingChairs2/train') 
        destdir = path / Path('flow_png')
        destdir.mkdir(exist_ok=True)
        
        for flow_path in sorted(path.rglob('*.flo')):
            flow = load_flo(flow_path)
            filename =  str(destdir / Path(flow_path.stem + '.png'))
            print(filename)
            mask = get_gt_correspondence_mask(flow)
            write_flow_png(filename, uv=flow, mask=mask)
            #flow2, mask2 = load_flow_from_png(filename)
            #np.testing.assert_array_equal(mask, mask2)
            #np.testing.assert_allclose(flow, flow2, rtol=1e-10, err_msg='flows do not match')

        path = Path('../../FlyingChairs2/val') 
        destdir = path / Path('flow_png')
        destdir.mkdir(exist_ok=True)

        for flow_path in sorted(path.rglob('*.flo')):
            flow = load_flo(flow_path)
            filename =  str(destdir / Path(flow_path.stem + '.png'))
            print(filename)
            mask = get_gt_correspondence_mask(flow)
            write_flow_png(filename, uv=flow, mask=mask)
            #flow2, mask2 = load_flow_from_png(filename)
            #np.testing.assert_array_equal(mask, mask2)
            #np.testing.assert_allclose(flow, flow2, rtol=1e-10, err_msg='flows do not match')

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

    finally:
        if platform.system() == 'Windows':
            env = 'CUDA_LAUNCH_BLOCKING'
            if env in os.environ:
                del os.environ[env]
        print("Closed cleanly")
