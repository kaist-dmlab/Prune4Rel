from .earlytrain import EarlyTrain
import torch
import numpy as np
from .methods_utils import euclidean_dist
from ..nets.nets_utils import MyDataParallel


def k_center_greedy(matrix, budget: int, metric, device, random_seed=None, index=None, already_selected=None,
                    print_freq: int = 20):
    if type(matrix) == torch.Tensor:
        assert matrix.dim() == 2
    elif type(matrix) == np.ndarray:
        assert matrix.ndim == 2
        matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)
    #print("matrix.shape: ", matrix.shape) # [1300, 512]

    sample_num = matrix.shape[0]
    assert sample_num >= 1

    if budget < 0:
        raise ValueError("Illegal budget size.")
    elif budget > sample_num:
        budget = sample_num

    if index is not None:
        assert matrix.shape[0] == len(index)
    else:
        index = np.arange(sample_num)

    assert callable(metric)

    already_selected = np.array(already_selected)

    with torch.no_grad():
        np.random.seed(random_seed)
        if already_selected.__len__() == 0:
            select_result = np.zeros(sample_num, dtype=bool)
            # Randomly select one initial point.
            already_selected = [np.random.randint(0, sample_num)]
            budget -= 1
            select_result[already_selected] = True
        else:
            select_result = np.in1d(index, already_selected)

        num_of_already_selected = np.sum(select_result)

        # Initialize a (num_of_already_selected+budget-1)*sample_num matrix storing distances of pool points from
        # each clustering center.
        dis_matrix = -1 * torch.ones([num_of_already_selected + budget - 1, sample_num], requires_grad=False).to(device)

        dis_matrix[:num_of_already_selected, ~select_result] = metric(matrix[select_result], matrix[~select_result])

        mins = torch.min(dis_matrix[:num_of_already_selected, :], dim=0).values

        for i in range(budget):
            if i % print_freq == 0:
                print("| Selecting [%3d/%3d]" % (i + 1, budget))
            p = torch.argmax(mins).item()
            select_result[p] = True

            if i == budget - 1:
                break
            mins[p] = -1
            dis_matrix[num_of_already_selected + i, ~select_result] = metric(matrix[[p]], matrix[~select_result])
            mins = torch.min(mins, dis_matrix[num_of_already_selected + i])
    return index[select_result]


class kCenterGreedy(EarlyTrain):
    def __init__(self, loader, configs, args, robust_learner, fraction=0.5, random_seed=None, epochs=0,
                 balance: bool = False, already_selected=[], metric="euclidean",
                 torchvision_pretrain: bool = False, **kwargs):
        super().__init__(loader, configs, args, robust_learner, fraction, random_seed, epochs=epochs,
                         torchvision_pretrain=torchvision_pretrain, **kwargs)

        if already_selected.__len__() != 0:
            if min(already_selected) < 0 or max(already_selected) >= self.n_train:
                raise ValueError("List of already selected points out of the boundary.")
        self.already_selected = np.array(already_selected)

        self.min_distances = None

        if metric == "euclidean":
            self.metric = euclidean_dist
        elif callable(metric):
            self.metric = metric
        else:
            self.metric = euclidean_dist
            self.run = lambda : self.finish_run()
            def _construct_matrix(index=None):
                data_loader = torch.utils.data.DataLoader(
                    self.dst_train_noTrans if index is None else torch.utils.data.Subset(self.dst_train_noTrans, index),
                    batch_size=self.n_train if index is None else len(index),
                    num_workers=self.args.workers)
                inputs, _, _ = next(iter(data_loader))
                return inputs.flatten(1).requires_grad_(False).to(self.args.device)
            self.construct_matrix = _construct_matrix

        self.balance = balance

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
            epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def construct_matrix(self, index=None):
        self.model.eval()
        self.model.no_grad = True
        with torch.no_grad():
            with self.model.embedding_recorder:
                matrix = []

                # original
                data_loader = self.loader.run('eval_train')

                # strong aug
                #data_loader = self.loader.run('eval_train_strong')

                for i, data in enumerate(data_loader):
                    inputs = data[0]
                    self.model(inputs.to(self.args.device))
                    matrix.append(self.model.embedding_recorder.embedding)

        self.model.no_grad = False
        return torch.cat(matrix, dim=0)

    def before_run(self):
        self.emb_dim = self.model.get_last_layer().in_features

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

    def select(self, **kwargs):
        selection_result, configs =self.run()
        matrix = self.construct_matrix() # [50000, 512]
        selection_result = k_center_greedy(matrix, budget=self.coreset_size,
                                           metric=self.metric, device=self.args.device,
                                           random_seed=self.random_seed,
                                           already_selected=self.already_selected, print_freq=self.args.print_freq)
        return {"indices": selection_result}, configs