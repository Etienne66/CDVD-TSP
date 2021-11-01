""" Trains a deep CNN to deblur video

Based on the paper "Cascaded Deep Video Deblurring Using Temporal Sharpness Prior"
"""
__author__ = "Jinshan Pan, Haoran Bai, and Jinhui Tang"
__copyright__ = "Copyright 2007, The Cogent Project"
__credits__ = ["Jinshan Pan", "Haoran Bai", "Jinhui Tang", "Steven Wilson"]
__license__ = "MIT"
__version__ = "2.0.1"
__maintainer__ = "Steven Wilson"
__email__ = "Etienne66"
__status__ = "Development"

import platform
import os
# This is to resolve an issue with SciPy interfering with <CTRL>+<C> and <CTRL>+<BREAK> in Windows
# https://github.com/ContinuumIO/anaconda-issues/issues/905
if platform.system() == 'Windows':
    env = 'FOR_DISABLE_CONSOLE_CTRL_HANDLER'
    if env not in os.environ:
        os.environ[env] = '1'
    # You can force synchronous computation by setting environment variable CUDA_LAUNCH_BLOCKING=1. This can be handy when an error occurs on the GPU. (With asynchronous execution, such an error isn't reported until after the operation is actually executed, so the stack trace does not show where it was requested.)
    # https://pytorch.org/docs/stable/notes/cuda.html
    # to debug cupy_backends.cuda.api.driver.CUDADriverError: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
    env = 'CUDA_LAUNCH_BLOCKING'
    if env not in os.environ:
        os.environ[env] = '1'

import torch
import data
import model
import loss
import option
from trainer.trainer_cdvd_tsp import Trainer_CDVD_TSP
from logger import logger
import random
import numpy as np
import traceback

if __name__ == '__main__':
    args = option.args
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if float(torch.version.cuda) >= 11.0:
        torch.backends.cudnn.enabled = False
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    chkp = logger.Logger(args)

    try:
        if args.task == 'VideoDeblur':
            print("Selected task: {}".format(args.task))
            model = model.Model(args, chkp)
            loss = loss.Loss(args, chkp) if not args.test_only else None
            loader = data.Data(args)
            t = Trainer_CDVD_TSP(args, loader, model, loss, chkp)
            start_time = args.start_time
            print("Start time: {}".format(args.start_time.strftime("%Y-%m-%d %H:%M:%S")))
            if not args.lr_finder:
                while not t.terminate():
                    t.train()
                    t.test()
        else:
            raise NotImplementedError('Task [{:s}] is not found'.format(args.task))

    #except:
    #    traceback.print_exc()
    #    exit()

    finally:
        if platform.system() == 'Windows':
            env = 'CUDA_LAUNCH_BLOCKING'
            if env in os.environ:
                del os.environ[env]
        print("Closed cleanly")
        if not args.lr_finder:
            chkp.done()
