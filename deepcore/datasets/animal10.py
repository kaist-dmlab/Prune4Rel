from torchvision import datasets
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from torch import tensor, long
import numpy as np
import deeplake

class MyAnimal10(Dataset):
    def __init__(self, file_path, train, download, transform):
        self.file_path = file_path
        self.train = train
        if self.train == True:
            self.file_path = "hub://activeloop/animal10n-train"
        else:
            self.file_path = "hub://activeloop/animal10n-test"
        #self.download = download

        self.animal10 = deeplake.load(self.file_path)

        #self.targets = np.array(self.animal10.targets)

    def __getitem__(self, index):
        data = self.animal10[index]
        #target = self.targets[index]

        return data, index

    def __len__(self):
        return len(self.animal10)

def Animal10(args):
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    test_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    if args.nsml == False: #local
        dst_train = MyAnimal10(args.data_path+'/cifar10', train=True, download=False, transform=train_transform)
        dst_train_noTrans = MyAnimal10(args.data_path + '/cifar10', train=True, download=False, transform=test_transform)
        dst_test = MyAnimal10(args.data_path+'/cifar10', train=False, download=False, transform=test_transform)
    else: #NSML
        file_path = args.nsml_data_path +'/train'
        print(args.nsml_data_path, file_path)
        dst_train = MyAnimal10(file_path, train=True, download=False, transform=train_transform)
        dst_train_noTrans = MyAnimal10(file_path, train=True, download=False, transform=test_transform)
        dst_test = MyAnimal10(file_path, train=False, download=False, transform=test_transform)

    print(dst_train)
    print(dst_train['labels'])
    print(len(dst_train['labels']))

    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_train_noTrans, dst_test