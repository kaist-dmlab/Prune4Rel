import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch import tensor, long
from randaugment import *

class MyImageNet30(Dataset):
    def __init__(self, file_path, transform=None, resolution=224):
        self.transform = transform
        self.resolution = resolution
        if self.resolution == 224: #32, 64, 128
            self.data = ImageFolder(file_path)
        else:
            print("Resizing Initial Data into {}x{}".format(self.resolution, self.resolution))
            transform_resize = T.Resize(size=(self.resolution,self.resolution)) #reduce the resolution once at an initial point
            self.data = ImageFolder(file_path, transform_resize)

        self.classes = self.data.classes
        self.targets = self.data.targets

        self.classes = ['acorn', 'airliner', 'ambulance', 'american_alligator', 'banjo', 'barn', 'bikini', 'digital_clock', 
            'dragonfly', 'dumbbell', 'forklift', 'goblet', 'grand_piano', 'hotdog', 'hourglass', 'manhole_cover', 
            'mosque', 'nail', 'parking_meter', 'pillow', 'revolver', 'rotary_dial_telephone', 'schooner', 'snowmobile',
            'soccer_ball', 'stingray', 'strawberry', 'tank', 'toaster', 'volcano']
        

    def __getitem__(self, index):
        # id = self.id_sample[index]
        img, label = self.data[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, label, index

    def __len__(self):
        return len(self.data)

def get_augmentations_4(T_normalize):
    train_transform = T.Compose([T.RandomHorizontalFlip(),T.RandomCrop(size=4, padding=1),T.ToTensor(),T_normalize])
    test_transform = T.Compose([T.ToTensor(),T_normalize])

    strong_transforms = T.Compose([
        T.RandomCrop(size=4, padding=1),
        T.RandomHorizontalFlip(),
        RandAugmentPC(n=3, m=5),
        T.ToTensor(),
        T_normalize])
    
    return train_transform, strong_transforms, test_transform

def get_augmentations_8(T_normalize):
    train_transform = T.Compose([T.RandomHorizontalFlip(),T.RandomCrop(size=8, padding=1),T.ToTensor(),T_normalize])
    test_transform = T.Compose([T.ToTensor(),T_normalize])

    strong_transforms = T.Compose([
        T.RandomCrop(size=8, padding=1),
        T.RandomHorizontalFlip(),
        RandAugmentPC(n=3, m=5),
        T.ToTensor(),
        T_normalize])
    
    return train_transform, strong_transforms, test_transform

def get_augmentations_16(T_normalize):
    train_transform = T.Compose([T.RandomHorizontalFlip(),T.RandomCrop(size=16, padding=2),T.ToTensor(),T_normalize])
    test_transform = T.Compose([T.ToTensor(),T_normalize])

    strong_transforms = T.Compose([
        T.RandomCrop(size=16, padding=2),
        T.RandomHorizontalFlip(),
        RandAugmentPC(n=3, m=5),
        T.ToTensor(),
        T_normalize])
    
    return train_transform, strong_transforms, test_transform

def get_augmentations_32(T_normalize):
    train_transform = T.Compose([T.RandomHorizontalFlip(),T.RandomCrop(size=32, padding=4),T.ToTensor(),T_normalize])
    test_transform = T.Compose([T.ToTensor(),T_normalize])

    strong_transforms = T.Compose([
        T.RandomCrop(size=32, padding=4),
        T.RandomHorizontalFlip(),
        RandAugmentPC(n=3, m=5),
        T.ToTensor(),
        T_normalize])
    
    return train_transform, strong_transforms, test_transform

def get_augmentations_224(T_normalize):
    train_transform = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),T_normalize])
    test_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T_normalize])

    strong_transforms = T.Compose([
        T.RandomResizedCrop(size=224),
        T.RandomHorizontalFlip(),
        RandAugmentPC(n=3, m=5),
        T.ToTensor(),
        T_normalize])
    
    return train_transform, strong_transforms, test_transform

def ImageNet30(args):
    channel = 3
    im_size = (args.resolution, args.resolution)
    num_classes = 30
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    T_normalize = T.Normalize(mean, std)

    if args.resolution == 4:
        train_transform, strong_transforms, test_transform = get_augmentations_4(T_normalize)
    if args.resolution == 8:
        train_transform, strong_transforms, test_transform = get_augmentations_8(T_normalize)
    if args.resolution == 16:
        train_transform, strong_transforms, test_transform = get_augmentations_16(T_normalize)
    if args.resolution == 32:
        train_transform, strong_transforms, test_transform = get_augmentations_32(T_normalize)
    if args.resolution == 224:
        train_transform, strong_transforms, test_transform = get_augmentations_224(T_normalize)

    dst_train = MyImageNet30(args.data_path+'/imgnet30/one_class_train/', transform=train_transform, resolution=args.resolution)
    dst_test = MyImageNet30(args.data_path+'/imgnet30/one_class_test/', transform=test_transform, resolution=args.resolution)

    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
