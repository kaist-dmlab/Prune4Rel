from .earlytrain import EarlyTrain
import torch
import numpy as np
from torchvision import transforms as T

class Uncertainty(EarlyTrain):
    def __init__(self, loader, configs, args, robust_learner, fraction=0.5, random_seed=None, epochs=200, selection_method="LeastConfidence",balance=False, **kwargs):
        super().__init__(loader, configs, args, robust_learner, fraction, random_seed, epochs, **kwargs)

        selection_choices = ["LeastConfidence",
                             "Entropy",
                             "Margin"]
        if selection_method not in selection_choices:
            raise NotImplementedError("Selection algorithm unavailable.")
        self.selection_method = selection_method

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
            selection_result = np.array([], dtype=np.int64)
            scores = []
            for c in range(self.args.n_class):
                class_index = np.arange(self.n_train)[self.loader.train_dataset.targets == c]
                scores.append(self.rank_uncertainty(class_index))
                selection_result = np.append(selection_result, class_index[np.argsort(scores[-1])[
                                                               :round(len(class_index) * self.fraction)]])
        else:
            scores = self.rank_uncertainty()
            selection_result = np.argsort(scores)[:self.coreset_size]
        return {"indices": selection_result, "scores": scores}

    def rank_uncertainty(self, index=None):
        self.model.eval()
        with torch.no_grad():
            train_loader = self.loader.run('eval_train')

            scores = np.array([])
            batch_num = len(train_loader)

            for i, data in enumerate(train_loader):
                input = data[0]
                if i % self.args.print_freq == 0:
                    print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))
                if self.selection_method == "LeastConfidence":
                    scores = np.append(scores, self.model(input.to(self.args.device)).max(axis=1).values.cpu().numpy())
                elif self.selection_method == "Entropy":
                    preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1).cpu().numpy()
                    scores = np.append(scores, (np.log(preds + 1e-6) * preds).sum(axis=1))
                elif self.selection_method == 'Margin':
                    preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1)
                    preds_argmax = torch.argmax(preds, dim=1)
                    max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                    preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                    preds_sub_argmax = torch.argmax(preds, dim=1)
                    scores = np.append(scores, (max_preds - preds[
                        torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy())
        return scores

    def select(self, **kwargs):
        selection_result, configs = self.run()
        return selection_result, configs
