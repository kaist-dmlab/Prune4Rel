from torchvision import datasets
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from torch import tensor, long
import numpy as np

class MyMNIST(Dataset):
    def __init__(self, file_path, train, download, transform):
        self.mnist = datasets.MNIST(file_path, train=train, download=download, transform=transform)
        self.targets = self.mnist.targets
        self.classes = self.mnist.classes      

    def __getitem__(self, index):
        data, target = self.mnist[index]
        return data, target, index

    def __len__(self):
        return len(self.mnist)

def MNIST(args, permuted=False, permutation_seed=None):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]
    
    '''
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    if permuted:
        np.random.seed(permutation_seed)
        pixel_permutation = np.random.permutation(28 * 28)
        transform = transforms.Compose(
            [transform, transforms.Lambda(lambda x: x.view(-1, 1)[pixel_permutation].view(1, 28, 28))])

    dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    '''

    train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=28, padding=4), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    test_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    #dst_train = datasets.CIFAR10(args.data_path+'/cifar10', train=True, download=False, transform=train_transform)
    dst_train = MyMNIST(args.data_path+'/mnist', train=True, download=True, transform=train_transform)
    dst_test = datasets.MNIST(args.data_path+'/mnist', train=False, download=True, transform=test_transform)

    class_names = [str(c) for c in range(num_classes)]
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
    

def permutedMNIST(data_path, permutation_seed=None):
    return MNIST(data_path, True, permutation_seed)
