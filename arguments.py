import argparse
import numpy as np
import time
from utils import *

parser = argparse.ArgumentParser(description='Parameter Processing')

# Basic arguments
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
parser.add_argument('--n-train', type=int, default=50000, help='num_samples')
parser.add_argument('--n-class', type=int, default=10, help='num_class')
parser.add_argument('--n-query', type=int, default=1000, help='num_query for oneshot')
parser.add_argument('--core-resolution', type=int, default=224, help='core-resolution')
parser.add_argument('--resolution', type=int, default=32, help='resolution') # 32
parser.add_argument('--core-model', type=str, default='ResNet18', help='core-model')
parser.add_argument('--model', type=str, default='ResNet18', help='model')
parser.add_argument('--penultimate', type=str_to_bool, default=False, help='penultimate')
parser.add_argument('--selection', type=str, default="uniform", help="selection method") #Uniform, Uncertainty
parser.add_argument('--robust-learner', type=str, default="DivideMix", help="robust learning method") #DivideMix, ELR, SOP
parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--data_path', type=str, default='/path/to/data/', help='dataset path')
parser.add_argument('--gpu', default=None, nargs="+", type=int, help='GPU id to use')
parser.add_argument('--print_freq', '-p', default=300, type=int, help='print frequency (default: 20)')
parser.add_argument('--fraction', default=0.2, type=float, help='fraction of data to be selected (default: 0.1)')
parser.add_argument('--seed', default=0, type=int, help="random seed")
parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument("--cross", type=str, nargs="+", default=None, help="models for cross-architecture experiments")
parser.add_argument("--pre-trained", type=str_to_bool, default=False, help='pre-trained model')
parser.add_argument("--multi-gpu", type=str_to_bool, default=False, help='multi_gpu')
parser.add_argument("--pre-warmuped", type=str_to_bool, default=False, help='pre-trained warmup model')
parser.add_argument('--pre-warmed-filename', type=str, default='/', help='pre-selected path')
parser.add_argument("--pre-selected", type=str_to_bool, default=False, help='pre-selected')
parser.add_argument('--pre-selected-filename', type=str, default='/', help='pre-selected path')



# Problem Setup arguments
parser.add_argument('--n-redundant', type=int, default=2, help='num_redundancy')
parser.add_argument('--noise-type', type=str, default='sym', help='label noise type') #sym, asym, real123AW
parser.add_argument('--noise-rate', type=float, default=0, help='label noise rate') # 0~1
parser.add_argument('--ood-rate', type=float, default=0.2, help='ood rate') # 0~1

# Optimizer and scheduler
parser.add_argument('--optimizer', default="SGD", help='optimizer to use, e.g. SGD, Adam')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate for updating network parameters')
parser.add_argument('--min_lr', type=float, default=1e-4, help='minimum learning rate') #1e-4
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)', dest='weight_decay')
parser.add_argument("--nesterov", default=True, type=str_to_bool, help="if set nesterov")
parser.add_argument("--scheduler", default="CosineAnnealingLR", type=str, help="Learning rate scheduler") #CosineAnnealingLR, MultiStepLR, StepLR
parser.add_argument('--milestone', type=list, default=[50], metavar='M', help='milestone for lr decay')
parser.add_argument("--gamma", type=float, default=0.1, help="Gamma value for StepLR")
parser.add_argument("--step_size", type=float, default=100, help="Step size for StepLR")

# Training
parser.add_argument('--batch-size', "-b", default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument("--train_batch", "-tb", default=None, type=int,
                    help="batch size for training, if not specified, it will equal to batch size in argument --batch")
parser.add_argument("--selection_batch", "-sb", default=None, type=int,
                    help="batch size for selection, if not specified, it will equal to batch size in argument --batch")

# Robust Learners
# DivideMix
parser.add_argument('--lambda-u', default=1, type=float, metavar='N', help='lambda-u for DivideMix')
parser.add_argument('--p-threshold', default=0.5, type=float, help='probability threshold for DivideMix')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
# ELR
parser.add_argument('--lambda-elr', default=1, type=int, metavar='N', help='lambda for ELR')
parser.add_argument('--beta', default=0.9, type=float, help='probability threshold for ELR')
# ELR+
parser.add_argument('--mixup-alpha', default=1, type=float, help='mixup alpha for ELR+')
parser.add_argument('--gamma-elr-plus', default=0.997, type=float, help='EMA gamma for ELR+')
# SOP+
parser.add_argument('--lr-u', type=float, default=10, help='learning rate for U')
parser.add_argument('--lr-v', type=float, default=100, help='learning rate for V')
parser.add_argument('-rc', '--ratio-consistency', default=0, type=float, help='consistency regularization lambda for SOP+')
parser.add_argument('-rb', '--ratio-balance', default=0, type=float, help='balance regularization lambda for SOP+')
#parser.add_argument('--cutout', default=0, type=float, help='cutout hyperparameter for SOP+')

# Testing
parser.add_argument("--test-batch-size", default=32, type=int,
                    help="batch size for training, if not specified, it will equal to batch size in argument --batch")
parser.add_argument("--test_interval", '-ti', default=1, type=int, help=
"the number of training epochs to be preformed between two test epochs; a value of 0 means no test will be run (default: 1)")
parser.add_argument("--test_fraction", '-tf', type=float, default=1.,
                    help="proportion of test dataset used for evaluating the model (default: 1.)")

# Selecting
parser.add_argument("--selection_epochs", "-se", default=10, type=int,
                    help="number of epochs whiling performing selection on full dataset")
parser.add_argument("--meta-epochs", "-me", default=20, type=int,
                    help="number of epochs whiling performing selection on full dataset")
parser.add_argument('--selection_momentum', '-sm', default=0.9, type=float, metavar='M',
                    help='momentum whiling performing selection (default: 0.9)')
parser.add_argument('--selection_weight_decay', '-swd', default=5e-4, type=float,
                    metavar='W', help='weight decay whiling performing selection (default: 5e-4)',
                    dest='selection_weight_decay')
parser.add_argument('--selection_optimizer', "-so", default="SGD",
                    help='optimizer to use whiling performing selection, e.g. SGD, Adam')
parser.add_argument("--selection_nesterov", "-sn", default=True, type=str_to_bool,
                    help="if set nesterov whiling performing selection")
parser.add_argument('--selection_lr', '-slr', type=float, default=0.1, help='learning rate for selection')
parser.add_argument('--meta-lr', '-mlr', type=float, default=0.1, help='learning rate for selection')
parser.add_argument("--meta-steps", default=1, type=int, help="number of steps for outer loop optimization")
parser.add_argument("--selection_test_interval", '-sti', default=1, type=int, help=
"the number of training epochs to be preformed between two test epochs during selection (default: 1)")
parser.add_argument("--selection_test_fraction", '-stf', type=float, default=1.,
            help="proportion of test dataset used for evaluating the model while preforming selection (default: 1.)")
parser.add_argument('--balance', default=False, type=str_to_bool, help="whether balance selection is performed per class")
parser.add_argument('--lamda', default=0.1, type=float, help="weights for balancing loss and regularizer")
parser.add_argument('--lamda-scheduling', default=False, type=str_to_bool, help="scheduling")
parser.add_argument('--pairwise-mask', default=False, type=str_to_bool, help="Pairwise Masking")
parser.add_argument("--num-bins", default=100, type=int, help="number of batches (bins) for BatchGradImpact")
parser.add_argument("--label-smoothing", default=1, type=float, help="label-smoothing factor")

# Coreset Selection Methods
parser.add_argument('--submodular', default="GraphCut", help="specifiy submodular function to use") #GraphCut, FacilityLocation, LogDeterminant
parser.add_argument('--submodular_greedy', default="StochasticGreedy", help="specifiy greedy algorithm for submodular optimization") #NaiveGreedy, LazyGreedy, StochasticGreedy
parser.add_argument('--uncertainty', default="Margin", help="specifiy uncertanty score to use") #LeastConfidence, Entropy, Margin
parser.add_argument('--repeat', default=1, type=int, help="# of repeat for GraNd")

# For Prune4Rel
parser.add_argument('--num-easy', default=1000, type=int, help="# of easy samples for EasyHardBalance")
parser.add_argument('--is-core', default=False, type=str_to_bool, help="How to select easy samples for EasyHardBalance")
parser.add_argument('--metric', default="euclidean", type=str, help="Distance measure in Emb space") #euclidean, cossim
parser.add_argument('--tau', default=0.95, type=float, help="to determine Neighborhood")
parser.add_argument('--eta', default=100, type=float, help="for utility function")

# Checkpoint and resumption
parser.add_argument('--save-log', default=False, type=str_to_bool, help="whether to save log or not")
parser.add_argument('--save_path', "-sp", type=str, default='', help='path to save results (default: do not save)')
parser.add_argument('--resume', '-r', type=str, default='', help="path to latest checkpoint (default: do not load)")

args = parser.parse_args()