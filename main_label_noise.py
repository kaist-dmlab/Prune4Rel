import os
import torch.nn as nn
import argparse
import deepcore.nets as nets
import deepcore.datasets as datasets
import deepcore.methods as methods
from torchvision import transforms
from utils import *
from datetime import datetime
from time import sleep
import numpy as np

# custom
from arguments import parser
from load_noise_data import get_dataloader  # get_dataset
from robustlearner.dividemix import train_DivideMix
from robustlearner.elr import train_ELR
from robustlearner.elr_plus import train_ELR_PLUS
from robustlearner.sop import train_SOP
from robustlearner.ce import train_CE

# for NSML
# from nsml import HAS_DATASET, DATASET_PATH

def main():
    # parse arguments
    args = parser.parse_args()
    args, checkpoint, start_exp, start_epoch = get_more_args(args)

    args.nsml = False  # HAS_DATASET
    args.nsml_data_path = False  # DATASET_PATH

    for exp in range(start_exp, args.num_exp):
        print('\n================== Exp %d ==================' % exp)
        print("dataset: ", args.dataset, ", model: ", args.model, ", selection: ", args.selection, ", num_ex: ",
              args.num_exp, ", epochs: ", args.epochs, ", fraction: ", args.fraction, ", seed: ", args.seed,
              ", lr: ", args.lr, ", save_path: ", args.save_path, ", resume: ", args.resume, ", device: ", args.device,
              "\n", sep="")

        loader = get_dataloader(args)
        torch.random.manual_seed(exp + args.seed)  # Should change this for changing seed

        # Get configurations
        networks, optimizers, schedulers, criterion = get_configuration(args, nets, args.model, checkpoint)
        print("| Main Model: {}, with resolution: {}".format(args.model, args.im_size))

        configs = [networks, optimizers, schedulers, criterion]

        # Core-set Selection
        print('| Coreset Selection with {}'.format(args.selection))
        selection_args = dict(epochs=args.selection_epochs,
                              selection_method=args.uncertainty,
                              balance=args.balance,
                              greedy=args.submodular_greedy,
                              function=args.submodular,
                              repeat=args.repeat
                              )
        method = methods.__dict__[args.selection](loader, configs, args, args.robust_learner, args.fraction, args.seed,
                                                  **selection_args)

        # FIXME: remove this, only for ImageNet-1k
        if args.pre_selected == True: 
            folder = '/data/sachoi/RobustPruningLogs/selection_indices/' + str(args.dataset) + '/' + str(
                args.robust_learner) + '/' \
                     + str(args.selection) + '/'
            filename = args.pre_selected_filename
            subset = {"indices": []}
            subset["indices"] = np.load(folder + filename)

        else:
            start_time = time.time()
            subset, configs = method.select()

            assert len(subset) == len(list(set(list(subset))))  # check duplication
            print("Elapsed Time for Core-set Selection: ", time.time() - start_time)

            # TODO: Remove these
            del method
            del configs
            torch.cuda.empty_cache()

        # How many noise selected?
        noise_idx = loader.noise_idx
        noise_in_selected_set = [idx for idx in subset["indices"] if idx in noise_idx]
        noise_ratio_in_coreset = len(noise_in_selected_set) / len(subset["indices"]) * 100
        print("# of noise/coreset: {}/{}".format(len(noise_in_selected_set), len(subset["indices"])))

        # Class-imbalance of selected examples?
        y = loader.train_dataset.targets[subset["indices"]]
        uni, cnt = np.unique(np.array(y), return_counts=True)
        print("uniques: ", uni)
        print("counts: ", cnt)
        print("# of noise/coreset: {}/{}".format(len(noise_in_selected_set), len(subset["indices"])))

        loader.set_coreset_idxs(subset["indices"])
        # print("# of Selected Coreset in train_set: ", len(loader.coreset_idxs))

        ##### Training of Robust Learners ##### 
        print('| Robust Training with {}'.format(args.robust_learner))
        networks, optimizers, schedulers, criterion = get_configuration(args, nets, args.model, checkpoint)
        start_time = time.time()
        
        #TODO: Unifying robust learners with Deepcore template like "methods.__dict__[args.selection]""
        if args.robust_learner == 'DivideMix':
            last_acc, best_acc = train_DivideMix(args, networks, optimizers, schedulers, loader)
        elif args.robust_learner == 'ELR':
            last_acc, best_acc = train_ELR(args, networks, optimizers, schedulers, loader)
        elif args.robust_learner == 'ELR_PLUS':
            last_acc, best_acc = train_ELR_PLUS(args, networks, optimizers, schedulers, criterion, loader)
        elif args.robust_learner == 'SOP':
            last_acc, best_acc = train_SOP(args, networks, optimizers, schedulers, criterion, loader)
        elif args.robust_learner == 'CE':
            last_acc, best_acc = train_CE(args, networks, optimizers, schedulers, criterion, loader)
        print("Elapsed Time for Robust Training on Coreset: ", time.time() - start_time)

        # Class-imbalance of selected examples?
        print("uniques: ", uni)
        print("counts: ", cnt)
        print("# of noise/coreset: {}/{}".format(len(noise_in_selected_set), len(subset["indices"])))

        # save log
        if args.save_log == True:
            log = np.array([last_acc, best_acc, noise_ratio_in_coreset] + list(cnt)).reshape((1, -1))
            folder = '/data/pdm102207/RobustCoreLogs/' + str(args.dataset) + '/' + str(
                args.robust_learner) + '/' + str(args.selection) + '/'
            if args.selection == "Prune4Rel":
                if args.balance == True:
                    folder = folder + 'tau'+ str(args.tau) + '_eta' + str(args.eta) + '_balance/'
                else: 
                    folder = folder + 'tau'+ str(args.tau) + '_eta' + str(args.eta) + '_unbalance/'
            filename = str(args.noise_type) + '_r' + str(args.noise_rate) + '_' + str(args.model) + '_fr' + str(
                args.fraction) + '_trial' + str(exp) + '_date' + str(datetime.now()) + '.txt'
            if not os.path.exists(folder):
                print("===making folder at ", folder)
                os.makedirs(folder)
            np.savetxt(folder + filename, log, fmt="%f")

            print('\n========= Results with Method %s saved! Trial: %d ========\n' % (args.selection, exp))

if __name__ == '__main__':
    main()
