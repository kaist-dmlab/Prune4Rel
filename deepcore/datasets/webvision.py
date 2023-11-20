from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms as T
from torch import tensor, long
import numpy as np
import random
import torch
import copy
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

from .augmentations import Augmentation, CutoutDefault
from .augmentation_archive import autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10#, autoaug_imagenet_policy, svhn_policie
from torchvision.transforms.functional import to_pil_image

# def pil_to_tensor(pil_image):
#     # PIL: [width, height]
#     # -> NumPy: [width, height, channel]
#     # -> Tensor: [channel, width, height]
#     return torch.as_tensor(np.asarray(pil_image)).permute(2,0,1)

# def tensor_to_pil(tensor_image):
#     return to_pil_image(tensor_image)

# def tensor_to_pltimg(tensor_image):
#     return tensor_image.permute(1,2,0).numpy()

class SubsetSequentialSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def sample_traning_set(train_imgs, labels, num_class, num_samples):
    random.shuffle(train_imgs)
    class_num = torch.zeros(num_class)
    sampled_train_imgs = []
    for impath in train_imgs:
        label = labels[impath]
        if class_num[label] < (num_samples / num_class):
            sampled_train_imgs.append(impath)
            class_num[label] += 1
        if len(sampled_train_imgs) >= num_samples:
            break
    return sampled_train_imgs


class MyWebVision(Dataset):
    def __init__(self, file_path, download, mode='train'):
        self.file_path = file_path
        self.download = download
        self.mode = mode
        self.noise_idx =[]

        self.webvision_train ={}
        self.webvision_test={}

        self.train_targets=[]


        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=224, padding=4), T.ToTensor(), T.Normalize(mean=mean, std=std)])
        self.test_transform = T.Compose([T.CenterCrop(size=224), T.ToTensor(), T.Normalize(mean=mean, std=std)])

        # data size 224
        self.train_strong_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=224, padding=4), T.ToTensor(), T.Normalize(mean=mean, std=std)])
        autoaug = T.Compose([])
        autoaug.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        self.train_strong_transform.transforms.insert(0, autoaug)
        self.train_strong_transform.transforms.append(CutoutDefault(16))


        if self.mode == 'test':
            self.val_imgs = []
            self.val_labels = {}
            with open(os.path.join(self.file_path, 'info/val_filelist.txt')) as f:
                lines = f.readlines()
            for idx, line in enumerate(lines):
                img, target = line.split()
                target = int(target)
                if target < 50: #crop 50 class
                    self.webvision_test[idx]= (img, target) #debug
                    self.val_imgs.append(img)
                    self.val_labels[img]=target


        else: #train
            self.train_imgs = []
            self.train_labels = {}
            with open(os.path.join(self.file_path, 'info/train_filelist_google.txt')) as f:
                lines = f.readlines()
                for idx, line in tqdm(enumerate(lines)): #change idx part
                    img, target = line.split()
                    target = int(target)
                    if target < 50:
                        self.train_targets.append(target)
                        self.webvision_train[idx]= (img, target)
                        self.train_imgs.append(img)
                        self.train_labels[img]=target


        self.targets = np.array(self.train_targets)
        self.n_class = 50 #len(set(self.train_targets))
        self.n_train = len(self.webvision_train)





    def set_mode(self, mode):
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            img_path = 'val_images/'+img_path
            img= Image.open(os.path.join(self.file_path, img_path)).convert('RGB').resize((256, 256))
            img = self.test_transform(img)  # test transform
            return img, target, index

        else: #train
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            img= Image.open(os.path.join(self.file_path, img_path)).convert('RGB').resize((256,256))


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
            return len(self.webvision_test)
        else:
            return len(self.webvision_train)



class webvision_dataloader():
    def __init__(self, args, file_path, download):
        self.args = args
        self.file_path = file_path
        self.download = download


        self.train_dataset = MyWebVision(self.file_path, self.download, mode='train')
        self.test_dataset = MyWebVision(self.file_path, self.download, mode='test')

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

        elif mode == 'eval_train':
            self.train_dataset.set_mode(mode)
            eval_sampler = SubsetSequentialSampler(self.coreset_idxs)
            eval_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, sampler=eval_sampler, num_workers=self.args.workers)
            return eval_loader

        elif mode == 'eval_train_strong':
            self.train_dataset.set_mode(mode)
            eval_sampler = SubsetSequentialSampler(self.coreset_idxs)
            eval_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, sampler=eval_sampler, num_workers=self.args.workers)
            return eval_loader

        elif mode == 'test':
            test_dataset = MyWebVision(self.file_path, self.download, mode='test')
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
