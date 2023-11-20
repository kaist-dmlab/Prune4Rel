from .earlytrain import EarlyTrain
import torch
import numpy as np
import time
import copy
from utils import *
import deepcore.nets as nets


class SmallLoss(EarlyTrain):
    def __init__(self, loader, configs, args, robust_learner, fraction=0.5, random_seed=None, epochs=200,
                 balance=False, **kwargs):
        super().__init__(loader, configs, args, robust_learner, fraction, random_seed, epochs, **kwargs)

        self.epochs = epochs
        self.balance = balance

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
        if self.balance:
            print("No balance version yet")
            scores = self.rank_loss_easy()
            selection_result = np.argsort(scores)[:self.coreset_size]
        else:
            st = time.time()
            scores = self.rank_loss_easy()
            et = time.time()
            print("Elapsed Time for LookAheadLoss: ", et - st)

            selection_result = np.argsort(scores)[:self.coreset_size]
        return {"indices": selection_result, "scores": scores}

    def rank_loss_easy(self, index=None):
        self.model.eval()
        with torch.no_grad():
            train_loader = self.loader.run('eval_train')
            criterion = nn.CrossEntropyLoss(reduction='none').to(self.args.device)

            scores = np.array([])
            batch_num = len(train_loader)

            for i, data in enumerate(train_loader):
                # input processing
                inputs, targets, index = data[0].to(self.args.device), data[1].to(self.args.device), data[2]

                outputs = self.model(inputs)

                loss = criterion(outputs, targets)

                scores = np.append(scores, loss.cpu().numpy())
        return scores

    def select(self, **kwargs):
        selection_result, configs = self.run()
        return selection_result, configs
