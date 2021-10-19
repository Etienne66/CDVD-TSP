import argparse
from option import template

parser = argparse.ArgumentParser(description='Video_Deblur')

parser.add_argument('--template', default='CDVD_TSP',
                    help='You can set various templates in options.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=4,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--use_checkpoint', action='store_true',
                    help='Use torch.utils.checkpoint to lower the Memory usage but take longer')
                    

# Data specifications
parser.add_argument('--dir_data', type=str, default='DVD',
                    help='comma separated list of directories in ../dataset to use for train')
parser.add_argument('--dir_data_test', type=str, default='DVD',
                    help='comma separated list of directories in ../dataset to use for test')
parser.add_argument('--data_train', type=str, default='DVD',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DVD',
                    help='test dataset name')
parser.add_argument('--process', action='store_true',
                    help='if True, load all dataset at once at RAM')
parser.add_argument('--patch_size', type=int, default=256,
                    help='output patch size')
parser.add_argument('--size_must_mode', type=int, default=1,
                    help='the size of the network input must mode this number')
parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--no_patch', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--n_sequence', type=int, default=5,
                    help='Set number of frames to evaluate for 1 output frame')

# Model specifications
parser.add_argument('--model', default='.',
                    help='model name')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')

# Training specifications
parser.add_argument('--test_every', type=int, default=1000,
                    help='Minimum number of images for training')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--batch_size_test', type=int, default=1,
                    help='input batch size for testing')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--lr_finder', action='store_true',
                    help='Only run the LRFinder to determine your lr and max_lr for OneCycleLR')
parser.add_argument('--lr_finder_Leslie_Smith', action='store_true',
                    help='Run the LRFinder using Leslie Smith''s approach')
parser.add_argument('--Adam', action='store_true',
                    help='Use the original Adam Optimizer instead of AdamW')
parser.add_argument('--StepLR', action='store_true',
                    help='Use the original StepLR Scheduler instead of OneCycleLR')
parser.add_argument('--OneCycleLR', action='store_true',
                    help='Use the OneCycleLR Scheduler')
parser.add_argument('--LossL1HEM', action='store_true',
                    help='Use the original 1*L1+2*HEM Loss instead of 0.84*MSL+0.16*L1')
parser.add_argument('--LossMslL1', action='store_true',
                    help='Use 0.84*MSL+0.16*L1')
parser.add_argument('--original_loss', action='store_true',
                    help='Combine the losses of all stages the same as the original')
parser.add_argument('--separate_loss', action='store_true',
                    help='Calculate losses during separate backward passes.')

# Optimization specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Sets Learning Rate')
parser.add_argument('--max_lr', type=float, default=1e-4,
                    help='Sets maximum Learning Rate for LRFinder and OneCycleLR')
parser.add_argument('--lr_decay', type=int, default=200,
                    help='learning rate decay per N epochs')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0.,
                    help='weight decay')
parser.add_argument('--AdamW_weight_decay', type=float, default=1e-2,
                    help='weight decay')
parser.add_argument('--mid_loss_weight', type=float, default=1.,
                    help='the weight of mid loss in trainer')

# Log specifications
parser.add_argument('--experiment_dir', type=str, default='../experiment/',
                    help='file name to save')
parser.add_argument('--pretrain_models_dir', type=str, default='../pretrain_models/',
                    help='file name to save')
parser.add_argument('--save', type=str, default='CDVD_TSP_Video_Deblur',
                    help='experiment name to save')
parser.add_argument('--save_middle_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--load', type=str, default='.',
                    help='experiment name to load')
parser.add_argument('--resume', action='store_true',
                    help='resume from the latest complete epoch')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_images', action='store_true',
                    help='save images during test phase of epoch')
parser.add_argument('--plot_running_average', action='store_true',
                    help='Plot the total for every epoch as a running average')
parser.add_argument('--running_average', type=int, default=2000,
                    help='How many iterations to include in the running average. Also how often to save plot during training.')



args = parser.parse_args()
template.set_template(args)

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
