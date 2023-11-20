from torchvision import datasets, transforms
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torch import tensor, long
import numpy as np
import random
import torch
import copy
from tqdm import tqdm
import pickle
import os

from .augmentations import Augmentation, CutoutDefault
from .augmentation_archive import autoaug_policy, autoaug_paper_cifar10, \
    fa_reduced_cifar10  # , autoaug_imagenet_policy, svhn_policie


class SubsetSequentialSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class MyImageNet(Dataset):
    def __init__(self, file_path, download, noise_type='sym', noise_rate=0.2, mode='train'):
        self.file_path = file_path
        self.download = download

        self.imagenet_train = ImageFolder(file_path + '/train/')
        self.imagenet_test = ImageFolder(file_path + '/val/')

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        T_normalize = T.Normalize(mean, std)

        # self.train_transform, self.test_transform = get_augmentations_224(T_normalize)
        self.train_transform = T.Compose(
            [T.Resize(256), T.RandomHorizontalFlip(), T.RandomCrop(size=224, padding=4), T.ToTensor(),
             T.Normalize(mean=mean, std=std)])
        self.test_transform = T.Compose(
            [T.Resize(256), T.CenterCrop(size=224), T.ToTensor(), T.Normalize(mean=mean, std=std)])

        self.train_strong_transform = T.Compose(
            [T.Resize(256), T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(),
             T.Normalize(mean=mean, std=std)])
        autoaug = T.Compose([])
        autoaug.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        self.train_strong_transform.transforms.insert(0, autoaug)
        self.train_strong_transform.transforms.append(CutoutDefault(16))

        self.targets = np.array(self.imagenet_train.targets)
        self.n_class = len(self.imagenet_train.classes)
        self.n_train = len(self.targets)

        self.mode = mode

        if noise_type == 'clean':
            self.noise_idx = []
        elif self.mode != 'test':
            # Noise Injection
            print("Imagenet, Noise Injection!")
            train_label = self.targets
            noise_label = []

            folder = '/data/sachoi/RobustPruningLogs/noisy_label/'
            filename = 'imagenet_noisy_idx_' + str(noise_type) + '_r' + str(noise_rate)

            if os.path.isfile(folder + filename + '.npy'):
                print("noisy idx already exist, take them")
                noise_idx = np.load(folder + filename + '.npy')
            else:
                idx = list(range(self.n_train))
                random.shuffle(idx)
                num_noise = int(noise_rate * self.n_train)
                noise_idx = idx[:num_noise]

                if not os.path.exists(folder):
                    os.makedirs(folder)
                np.save(folder + filename + '.npy', noise_idx)
                np.savetxt(folder + filename + '.txt', noise_idx, fmt="%d")

            if noise_type == 'sym':  # Symmetric Noise
                print("Inject Symmetric Noise! Noise rate:", noise_rate)
                filename = 'imagenet_noisy_label_' + str(noise_type) + '_r' + str(noise_rate)

                if os.path.isfile(folder + filename + '.npy'):
                    print("noisy label already exist, take them")
                    noise_label = np.load(folder + filename + '.npy')

                else:
                    without_class = True
                    for i in tqdm(range(self.n_train)):
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

                    noise = np.array(noise_label)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    np.save(folder + filename + '.npy', noise)
                    np.savetxt(folder + filename + '.txt', noise, fmt="%d")

            elif noise_type == 'asym':  # Asymmetric Noise
                print("Inject Asymmetric Noise! Noise rate:", noise_rate)
                filename = 'imagenet_noisy_label_' + str(noise_type) + '_r' + str(noise_rate)

                if os.path.isfile(folder + filename + '.npy'):
                    print("noisy label already exist, take them")
                    noise_label = np.load(folder + filename + '.npy')
                else:
                    for i in tqdm(range(self.n_train)):
                        if i in noise_idx:
                            noiselabel = (train_label[i] + 1) % 1000
                            noise_label.append(noiselabel)
                        else:
                            noise_label.append(train_label[i])

                    noise = np.array(noise_label)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    np.save(folder + filename + '.npy', noise)
                    np.savetxt(folder + filename + '.txt', noise, fmt="%d")

            assert noise_label != []

            self.targets = np.array(noise_label)
            self.noise_idx = noise_idx

    def set_mode(self, mode):
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'test':
            img, target = self.imagenet_test[index]
            img = self.test_transform(img)  # test transform
            return img, target, index

        else:
            img, _ = self.imagenet_train[index]
            target = self.targets[index]

            if self.mode == 'train':
                img = self.train_transform(img)
                return img, target, index

            elif self.mode == 'eval_train':
                img = self.test_transform(img)
                return img, target, index

            elif self.mode == 'eval_train_strong':
                img = self.train_strong_transform(img)
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
            return len(self.imagenet_test)
        else:
            return len(self.imagenet_train)


def get_augmentations_32(T_normalize):
    train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T_normalize])
    test_transform = T.Compose([T.ToTensor(), T_normalize])

    return train_transform, test_transform


def get_augmentations_224(T_normalize):
    train_transform = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), T_normalize])
    test_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T_normalize])

    return train_transform, test_transform


class imagenet_dataloader():
    def __init__(self, args, file_path, download, noise_type='sym', noise_rate=0.2):
        self.args = args
        self.file_path = file_path
        self.download = download
        self.noise_type = noise_type
        self.noise_rate = noise_rate

        self.train_dataset = MyImageNet(self.file_path, self.download, self.noise_type, self.noise_rate, mode='train')
        self.test_dataset = MyImageNet(self.file_path, self.download, self.noise_type, self.noise_rate, mode='test')

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
            trainloader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, sampler=train_sampler,
                                     num_workers=self.args.workers)
            return trainloader

        elif mode == 'eval_train':
            self.train_dataset.set_mode(mode)
            eval_sampler = SubsetSequentialSampler(self.coreset_idxs)
            eval_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, sampler=eval_sampler,
                                     num_workers=self.args.workers)
            return eval_loader

        elif mode == 'eval_train_strong':
            self.train_dataset.set_mode(mode)
            eval_sampler = SubsetSequentialSampler(self.coreset_idxs)
            eval_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, sampler=eval_sampler,
                                     num_workers=self.args.workers)
            return eval_loader

        elif mode == 'test':
            test_dataset = MyImageNet(self.file_path, self.download, mode='test')
            test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.batch_size, shuffle=False,
                                     num_workers=self.args.workers)
            return test_loader

        elif mode == 'DivideMix':
            self.train_dataset.set_mode('L_DivideMix')
            pred_idx = pred.nonzero()[0]

            self.train_dataset.probability = np.zeros(self.n_train)
            for i in pred_idx:
                self.train_dataset.probability[self.coreset_idxs[i]] = prob[i]

            labeled_sampler = SubsetRandomSampler(self.coreset_idxs[pred_idx])
            labeled_trainloader = copy.deepcopy(
                DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, sampler=labeled_sampler,
                           num_workers=self.args.workers))

            self.train_dataset.set_mode('U_DivideMix')
            pred_idx = (1 - pred).nonzero()[0]

            unlabeled_sampler = SubsetRandomSampler(self.coreset_idxs[pred_idx])
            unlabeled_trainloader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size,
                                               sampler=unlabeled_sampler, num_workers=self.args.workers)
            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'SOP':
            self.train_dataset.set_mode('SOP')
            sop_sampler = SubsetRandomSampler(self.coreset_idxs)
            sop_trainloader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size,
                                         sampler=sop_sampler, num_workers=self.args.workers)

            return sop_trainloader

# class MyImageNet(Dataset):
#     def __init__(self, file_path, transform=None, resolution=224):
#         self.transform = transform
#         self.resolution = resolution
#         if self.resolution == 224: #32, 64, 128
#             self.data = ImageFolder(file_path)
#         else:
#             print("Resizing Initial Data into {}x{}".format(self.resolution, self.resolution))
#             transform_resize = T.Resize(size=(self.resolution,self.resolution)) #reduce the resolution once at an initial point
#             self.data = ImageFolder(file_path, transform_resize)

#         self.classes = self.data.classes
#         self.targets = self.data.targets

#     def __getitem__(self, index):
#         # id = self.id_sample[index]
#         img, label = self.data[index]
#         if self.transform is not None:
#             img = self.transform(img)

#         return img, label#, index

#     def __len__(self):
#         return len(self.data)

# def get_augmentations_32(T_normalize):
#     train_transform = T.Compose([T.RandomHorizontalFlip(),T.RandomCrop(size=32, padding=4),T.ToTensor(),T_normalize])
#     test_transform = T.Compose([T.ToTensor(),T_normalize])

#     return train_transform, test_transform

# def get_augmentations_224(T_normalize):
#     train_transform = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),T_normalize])
#     test_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T_normalize])

#     return train_transform, test_transform

# def ImageNet(args):
#     channel = 3
#     im_size = (224, 224)
#     num_classes = 1000
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     T_normalize = T.Normalize(mean, std)

#     if args.resolution == 32:
#         train_transform, test_transform = get_augmentations_32(T_normalize)
#     if args.resolution == 224:
#         train_transform, test_transform = get_augmentations_224(T_normalize)

#     dst_train = MyImageNet(args.data_path+'/imagenet/train/', transform=train_transform, resolution=args.resolution)
#     dst_test = MyImageNet(args.data_path+'/imagenet/val/', transform=test_transform, resolution=args.resolution)


#     class_names = dst_train.classes
#     dst_train.targets = tensor(dst_train.targets, dtype=long)
#     dst_test.targets = tensor(dst_test.targets, dtype=long)
#     return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test

