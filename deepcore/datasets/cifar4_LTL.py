import os
import numpy as np
import torch
import math
import random
from torch.utils.data.dataset import Subset, Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import datasets
import torchvision.transforms as T
from PIL import ImageChops

class MyCIFAR4(Dataset):
    def __init__(self, file_path, train, download, transform):
        self.cifar10 = CIFAR10(root=file_path,download=download,train=train,transform=transform)
        self.sub_targets = self.cifar10.targets
        self.targets = np.copy(self.sub_targets) #binary
        self.classes = self.cifar10.classes

        for i, target in enumerate(self.sub_targets):
            if target not in [2,3,4,5]:
                self.targets[i] = -1
                self.sub_targets[i] = -1

        for i, target in enumerate(self.sub_targets):
            if target == 2:
                self.targets[i] = 0
                self.sub_targets[i] = 0
            if target == 3:
                self.targets[i] = 0
                self.sub_targets[i] = 1
            if target == 4:
                self.targets[i] = 1
                self.sub_targets[i] = 2
            if target == 5:
                self.targets[i] = 1
                self.sub_targets[i] = 3
                

    def __getitem__(self, index):
        data, _ = self.cifar10[index]
        target = self.targets[index] #binary
        sub_target = self.sub_targets[index]
        return data, target, sub_target

    def __len__(self):
        return len(self.cifar10)

def get_4class_indices(args, dataset, classes, imbalance_rate):
    num_labeled_per_class = int(imbalance_rate*len(dataset)/args.n_class)
    L_index = []
    for c in range(len(classes)):
        idx = [i for i in range(len(dataset)) if dataset[i][2] == c] 
        if c == 1 or c == 3:
            idx = np.random.choice(idx, num_labeled_per_class, False)
        L_index.extend(idx)
    #L_index = [i for i in range(len(dataset)) if dataset[i][2] in classes]
    return L_index

def CIFAR4_LTL(args):
    channel = 3
    im_size = (32, 32)
    num_classes = 2
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    test_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    dst_train = MyCIFAR4(args.data_path+'/cifar10', train=True, download=False, transform=train_transform)
    dst_test = MyCIFAR4(args.data_path+'/cifar10', train=False, download=False, transform=test_transform)
    
    target_list = [0,1,2,3]
    imbalance_rate = 0.2
    print("Target classes: ", target_list)

    # get IN indices
    L_train_index = get_4class_indices(args, dst_train, target_list, imbalance_rate)
    L_test_index = get_4class_indices(args, dst_test, target_list, imbalance_rate=1)
    print("# Train: {}, # Test: {}".format(len(L_train_index), len(L_test_index)))

    dst_train = torch.utils.data.Subset(dst_train, L_train_index)
    dst_test = torch.utils.data.Subset(dst_test, L_test_index)

    return channel, im_size, num_classes, mean, std, dst_train, dst_test