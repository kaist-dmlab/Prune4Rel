from .earlytrain import EarlyTrain
import torch
import torchvision.transforms as transforms
import numpy as np
import argparse
import pickle
from tqdm import tqdm
import os
import time
from deepcore.nets.preactresent import PreActResNet18Extractor # preactresnet typo
from deepcore.nets.resnet import ResNet50Extractor
from deepcore.nets.inceptionresnetv2 import Inception_ResNetv2_Extractor
from collections import OrderedDict


class Moderate(EarlyTrain):
    def __init__(self, loader, configs, args, robust_learner, fraction=0.5, random_seed=None, epochs=200,
                 balance=False, **kwargs):
        super().__init__(loader, configs, args, robust_learner, fraction, random_seed, epochs, **kwargs)

        self.epochs = epochs
        self.balance = balance
        self.args = args

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
            epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def finish_run(self):
        features, targets = self.get_features()
        distance = self.get_distance(features, targets)
        ids = self.get_prune_idx(distance) # drop id -> selected id

        if self.balance:
            print("No balance version yet")
        else:
            st = time.time()
            selection_result = ids
            et = time.time()
            print("Elapsed Time for LookAheadLoss: ", et - st)

        return {"indices": selection_result}


    def get_median(self, features, targets):
        # get the median feature vector of each class
        num_classes = len(np.unique(targets, axis=0))
        prot = np.zeros((num_classes, features.shape[-1]), dtype=features.dtype)

        for i in range(num_classes):
            prot[i] = np.median(features[(targets == i).nonzero(), :].squeeze(), axis=0, keepdims=False)
        return prot


    def get_distance(self, features, labels):

        prots = self.get_median(features, labels)
        prots_for_each_example = np.zeros(shape=(features.shape[0], prots.shape[-1]))

        num_classes = len(np.unique(labels))
        for i in range(num_classes):
            prots_for_each_example[(labels==i).nonzero()[0], :] = prots[i]
        distance = np.linalg.norm(features - prots_for_each_example, axis=1)

        return distance


    def get_features(self):
        # obtain features of each sample
        model = self.get_extractor()
        # model.load_state_dict(self.model.module.state_dict())

        loaded_state_dict = self.model.state_dict()
        new_state_dict = OrderedDict()
        for n, v in loaded_state_dict.items():
            name = n.replace("module.","") 
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

        model = model.to(self.args.device)

        train_loader = self.loader.run('eval_train')
        targets, features = [], []

        # for _, img, target in tqdm(train_loader):
        #     targets.extend(target.numpy().tolist())

        #     img = img.to(args.device)
        #     feature = model(img).detach().cpu().numpy()
        #     features.extend([feature[i] for i in range(feature.shape[0])])

        for i, data in tqdm(enumerate(train_loader)):
            inputs, target, index = data[0], data[1], data[2]
            targets.extend(target.numpy().tolist())
            inputs = inputs.to(self.args.device)
            feature = model(inputs).detach().cpu().numpy()
            features.extend([feature[i] for i in range(feature.shape[0])])

        features = np.array(features)
        targets = np.array(targets)

        return features, targets

    def get_extractor(self):
        model_name = self.args.model
        num_classes = self.args.n_class

        if model_name == 'PreActResNet18':
            print("model is PreActResNet18")
            print("num_classes: ", num_classes)
            model = PreActResNet18Extractor(num_classes=num_classes)

        elif model_name == 'ResNet50':
            print("model is ResNet50")
            print("num_classes: ", num_classes)
            model = ResNet50Extractor(num_classes=num_classes)
            
        elif model_name == 'Inception_ResNetv2':
            print("model is Inception_ResNetv2")
            print("num_classes: ", num_classes)
            model = Inception_ResNetv2_Extractor(num_classes=num_classes)
            
        else:
            raise NotImplementedError("Only extractors resnet18 are implemented")

        return model

    def get_prune_idx(self, distance):

        low = 0.5 - self.args.fraction / 2
        high = 0.5 + self.args.fraction / 2
        print(self.args.fraction)
        sorted_idx = distance.argsort()
        low_idx = round(distance.shape[0] * low)
        high_idx = round(distance.shape[0] * high)

        # ids = np.concatenate((sorted_idx[:low_idx], sorted_idx[high_idx:]))
        ids = sorted_idx[low_idx:high_idx]

        return ids


    def select(self, **kwargs):
        selection_result, configs = self.run()
        return selection_result, configs