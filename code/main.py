import platform
import os
# This is to resolve an issue with SciPy interfering with <CTRL>+<C> and <CTRL>+<BREAK> in Windows
# https://github.com/ContinuumIO/anaconda-issues/issues/905
if platform.system() == 'Windows':
    env = 'FOR_DISABLE_CONSOLE_CTRL_HANDLER'
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

if __name__ == '__main__':
    args = option.args
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
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

    finally:
        print("Closed cleanly")
        if not args.lr_finder:
            chkp.done()
