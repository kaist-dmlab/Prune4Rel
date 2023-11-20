import numpy as np
from .coresetmethod import CoresetMethod


class Uniform(CoresetMethod):
    def __init__(self, loader, configs, args, robust_learner, fraction=0.5, random_seed=None, balance=False, replace=False, **kwargs):
        super().__init__(loader, configs, args, robust_learner, fraction, random_seed)
        self.balance = balance
        self.replace = replace

    def select_balance(self):
        """The same sampling proportions were used in each class separately."""
        np.random.seed(self.random_seed)
        self.index = np.array([], dtype=np.int64)
        all_index = np.arange(self.n_train)
        for c in range(self.num_classes):
            c_index = (self.dst_train.targets == c)
            self.index = np.append(self.index,
                                   np.random.choice(all_index[c_index], round(self.fraction * c_index.sum().item()),
                                                    replace=self.replace))
        return self.index

    def select_no_balance(self):
        np.random.seed(self.random_seed)
        self.index = np.random.choice(np.arange(self.n_train), round(self.n_train * self.fraction),
                                      replace=self.replace)

        return self.index

    def select(self, **kwargs):
        return {"indices": self.select_balance() if self.balance else self.select_no_balance()}, self.configs
