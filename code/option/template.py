from datetime import datetime

def set_template(args):
    if args.template == 'CDVD_TSP':
        args.task = "VideoDeblur"
        args.model = "CDVD_TSP"
        args.n_sequence = 5 # [3,5,7]
        args.n_channel=12 # = n_colors * n_sequence [3,12,27] used for MS_SSIM_LOSS(MSL)
        args.n_frames_per_video = 200
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        #args.loss = '1*L1+2*HEM'
        #args.loss = '0.84*MSL+0.16*L1' # alpha*MSL + (1-alpha)*L1 - Alpha 0.84/0.025 paper/code
        args.loss = '0.16*MSL+0.84*L1' # alpha*MSL + (1-alpha)*L1 - Alpha 0.84/0.025 paper/code
        args.lr = 1e-4
        args.lr_decay = 200
        args.start_time = datetime.now()
        args.total_train_time = datetime.now() - datetime.now()
        args.total_test_time = datetime.now() - datetime.now()
        args.epochs_completed = 0
    else:
        raise NotImplementedError('Template [{:s}] is not found'.format(args.template))
