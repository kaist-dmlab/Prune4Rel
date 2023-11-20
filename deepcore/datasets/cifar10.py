from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms as T
from torch import tensor, long
import numpy as np
import random
import torch
import copy

from .augmentations import Augmentation, CutoutDefault
from .augmentation_archive import autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10#, autoaug_imagenet_policy, svhn_policie

class SubsetSequentialSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class MyCIFAR10(Dataset):
    def __init__(self, file_path, download, noise_type='real', noise_rate = 0.2, mode='train'):
        self.file_path = file_path
        self.download = download

        self.cifar10_train = datasets.CIFAR10(file_path, train=True, download=download)
        self.cifar10_test = datasets.CIFAR10(file_path, train=False, download=download)

        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        self.train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T.Normalize(mean=mean, std=std)])
        self.test_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

        self.train_strong_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T.Normalize(mean=mean, std=std)])
        autoaug = T.Compose([])
        autoaug.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        self.train_strong_transform.transforms.insert(0, autoaug)
        self.train_strong_transform.transforms.append(CutoutDefault(16))

        self.clean_targets = np.array(self.cifar10_train.targets).copy()
        self.targets = np.array(self.cifar10_train.targets)
        self.n_class = len(self.cifar10_train.classes)
        self.n_train = len(self.targets)

        self.mode = mode

        if noise_type == 'clean':
            self.noise_idx = []
        elif self.mode != 'test':
            # Noise Injection
            print("CIFAR10, Noise Injection!")
            train_label = self.targets
            noise_label = []
            idx = list(range(self.n_train))
            random.shuffle(idx)
            num_noise = int(noise_rate * self.n_train)
            noise_idx = idx[:num_noise]
            if noise_type == 'sym':  # Symmetric Noise
                print("Inject Symmetric Noise!")
                without_class = True
                for i in range(self.n_train):
                    if i in noise_idx:
                        if without_class:
                            noiselabel = random.randint(0, self.n_class - 2)
                            if noiselabel >= train_label[i]:
                                noiselabel += 1
                        else:
                            noiselabel = random.randint(0, self.n_class - 1)
                        noise_label.append(noiselabel)
                    else:
                        noise_label.append(train_label[i])

            elif noise_type == 'asym':  # Asymmetric Noise
                print("Inject Asymmetric Noise!")
                transition = {0: 2, 2: 0, 4: 7, 7: 4, 1: 9, 9: 1, 3: 5, 5: 3, 6: 8, 8: 6}  # bi-directional
                for i in range(self.n_train):
                    if i in noise_idx:
                        noiselabel = transition[train_label[i]]
                        noise_label.append(noiselabel)
                    else:
                        noise_label.append(train_label[i])

            elif noise_type in ['real1', 'real2', 'real3', 'realA', 'realW']:  # Real Noise
                noise_labels = torch.load(file_path + 'CIFAR-10_human.pt')
                if noise_type == 'real1':
                    print("Inject Real (Random1) Noise!")
                    noise_label = noise_labels['random_label1']  # ['aggre_label'], ['worse_label']
                elif noise_type == 'real2':
                    print("Inject Real (Random2) Noise!")
                    noise_label = noise_labels['random_label2']
                elif noise_type == 'real3':
                    print("Inject Real (Random3) Noise!")
                    noise_label = noise_labels['random_label3']
                elif noise_type == 'realA':
                    print("Inject Real (Aggregate) Noise!")
                    noise_label = noise_labels['aggre_label']
                elif noise_type == 'realW':
                    print("Inject Real (Worst) Noise!")
                    noise_label = noise_labels['worse_label']
                noise_idx = np.where(noise_label != noise_labels['clean_label'])[0]

            assert noise_label != []

            self.targets = np.array(noise_label)
            self.noise_idx = noise_idx

    def set_mode(self, mode):
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'test':
            img, target = self.cifar10_test[index]
            img = self.test_transform(img)  # test transform
            return img, target, index

        else:
            img, _ = self.cifar10_train[index]
            target = self.targets[index]

            if self.mode == 'train':
                img = self.train_transform(img)
                return img, target, index

            elif self.mode == 'eval_train':
                img = self.test_transform(img)
                return img, target, index

            # TODO:
            elif self.mode == 'eval_train_strong':
                img = self.train_strong_transform(img)
                return img, target, index

            # TODO:
            elif self.mode == 'eval_train_strong_multi':
                n_aug = 10
                imgs = []
                for i in range(n_aug):
                    img_ = self.train_strong_transform(img)
                    imgs.append(img_)
                return imgs, target, index

            elif self.mode == 'eval_train_clean':
                img = self.test_transform(img)
                target = self.clean_targets[index]
                return img, target, index

            elif self.mode == 'L_DivideMix':
                prob = self.probability[index]

                img1 = self.train_transform(img)
                img2 = self.train_transform(img)
                return img1, img2, target, prob

            elif self.mode == 'U_DivideMix':
                img1 = self.train_transform(img)
                img2 = self.train_transform(img)
                return img1, img2

            elif self.mode == 'SOP':
                img1 = self.train_transform(img)
                img2 = self.train_strong_transform(img)
                return img1, img2, target, index

    def __len__(self):
        if self.mode == "test":
            return len(self.cifar10_test)
        else:
            return len(self.cifar10_train)

class cifar10_dataloader():
    def __init__(self, args, file_path, download, noise_type='real', noise_rate = 0.2):
        self.args = args
        self.file_path = file_path
        self.download = download
        self.noise_type = noise_type
        self.noise_rate = noise_rate

        self.train_dataset = MyCIFAR10(self.file_path, self.download, self.noise_type, self.noise_rate, mode='train')
        self.test_dataset = MyCIFAR10(self.file_path, self.download, self.noise_type, self.noise_rate, mode='test')

        self.n_train = len(self.train_dataset)
        self.noise_idx = self.train_dataset.noise_idx

        self.coreset_idxs = list(range(self.n_train))
        self.n_coreset = len(self.coreset_idxs)

    def subsample_train_dataset(self, subset):
        self.train_dataset = torch.utils.data.Subset(self.train_dataset, subset)

    def set_coreset_idxs(self, idxs):
        self.coreset_idxs = idxs
        self.n_coreset = len(self.coreset_idxs)

    def run(self, mode, pred=[], prob=[]):
        if mode == 'train':
            self.train_dataset.set_mode(mode)
            train_sampler = SubsetRandomSampler(self.coreset_idxs)
            trainloader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, sampler=train_sampler, num_workers=self.args.workers)
            return trainloader

        elif mode in ['eval_train', 'eval_train_strong', 'eval_train_strong_multi', 'eval_train_clean']: # TODO: check
            self.train_dataset.set_mode(mode)
            eval_sampler = SubsetSequentialSampler(self.coreset_idxs)
            eval_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, sampler=eval_sampler, num_workers=self.args.workers)
            return eval_loader

        elif mode == 'test':
            test_dataset = MyCIFAR10(self.file_path, self.download, self.noise_type, self.noise_rate, mode)
            test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.workers)
            return test_loader

        elif mode == 'DivideMix':
            self.train_dataset.set_mode('L_DivideMix')
            pred_idx = pred.nonzero()[0]

            self.train_dataset.probability = np.zeros(self.n_train)
            for i in pred_idx:
                self.train_dataset.probability[self.coreset_idxs[i]] = prob[i]

            labeled_sampler = SubsetRandomSampler(self.coreset_idxs[pred_idx])
            labeled_trainloader = copy.deepcopy(DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, sampler=labeled_sampler, num_workers=self.args.workers))

            self.train_dataset.set_mode('U_DivideMix')
            pred_idx = (1 - pred).nonzero()[0]

            unlabeled_sampler = SubsetRandomSampler(self.coreset_idxs[pred_idx])
            unlabeled_trainloader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, sampler=unlabeled_sampler, num_workers=self.args.workers)
            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'SOP':
            self.train_dataset.set_mode('SOP')
            sop_sampler = SubsetRandomSampler(self.coreset_idxs)
            sop_trainloader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, sampler=sop_sampler, num_workers=self.args.workers)

            return sop_trainloader


