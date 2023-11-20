from .earlytrain import EarlyTrain
import torch
import numpy as np
from .methods_utils import euclidean_dist, cossim
from ..nets.nets_utils import MyDataParallel
import os

class Prune4Rel(EarlyTrain):
    def __init__(self, loader, configs, args, robust_learner, fraction=0.5, random_seed=None, epochs=0,
                 selection_method="LeastConfidence", balance: bool = False, metric="cossim",
                 torchvision_pretrain: bool = False, **kwargs):
        super().__init__(loader, configs, args, robust_learner, fraction, random_seed, epochs=epochs,
                         torchvision_pretrain=torchvision_pretrain, **kwargs)

        self.min_distances = None

        self.metric_name = metric
        if metric == "euclidean":
            self.metric = euclidean_dist
        elif metric == "cossim":
            self.metric = cossim

        self.balance = balance

        selection_choices = ["LeastConfidence",
                             "Entropy",
                             "Margin"]
        if selection_method not in selection_choices:
            raise NotImplementedError("Selection algorithm unavailable.")
        self.selection_method = selection_method

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
            epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def get_prob_embedding(self):
        print("Getting probs & embs!")
        self.model.eval()
        self.model.no_grad = True
        with torch.no_grad():
            with self.model.embedding_recorder:
                probs, embs = [], []
                # eval_train, eval_train_strong
                data_loader = self.loader.run('eval_train_strong')
                for i, data in enumerate(data_loader):
                    inputs = data[0]
                    output = self.model(inputs.to(self.args.device))
                    prob = torch.nn.functional.softmax(output.data, dim=1)

                    probs.append(prob.half())
                    embs.append(self.model.embedding_recorder.embedding.half())

                    if i%1000==0:
                        print("emb_batch: ", i)

        self.model.no_grad = False
        return torch.cat(probs, dim=0), torch.cat(embs, dim=0)

    def before_run(self):
        self.emb_dim = self.model.get_last_layer().in_features

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

    def k_neighbor_confidence_greedy(self, probs, embs, budget: int, metric, device, random_seed=None, index=None,
                                      print_freq: int = 20):
        if type(embs) == torch.Tensor:
            assert embs.dim() == 2
        elif type(embs) == np.ndarray:
            assert embs.ndim == 2
            matrix = torch.from_numpy(embs).requires_grad_(False).to(device)

        sample_num = embs.shape[0]
        assert sample_num >= 1

        if budget < 0:
            raise ValueError("Illegal budget size.")

        assert callable(metric)

        # Calculate Per-sample Confidence (from Uncertainty scores)
        if self.selection_method == "Entropy":
            uniform = (1/self.args.n_class)*torch.ones(self.args.n_class)
            highest = -(torch.log(uniform + 1e-6) * uniform).sum()
            print("highest entropy: ", highest)
            confs = -(torch.log(probs + 1e-6) * probs).sum(axis=1)  # [n_train]
        elif self.selection_method == 'Margin':
            topk_idxs = torch.topk(probs, k=2).indices  # [n_train, k]
            max_probs = probs.gather(1, topk_idxs[:, 0].reshape((-1, 1)))  # different column index in each row
            max_2nd_probs = probs.gather(1, topk_idxs[:, 1].reshape((-1, 1)))
            confs = (max_probs - max_2nd_probs).reshape(-1)
        elif self.selection_method == "LeastConfidence":
            confs = probs.max(axis=1).values

        if self.args.balance == True:
            print("balanced sampling!")
            available_classes = np.arange(self.args.n_class)

            #pseudo_labels = probs.argmax(axis=1)
            noisy_labels = torch.tensor(self.loader.train_dataset.targets)

            num_sample_per_class, target_class_idxs_list = [], []
            for c in available_classes:
                num_sample_per_class.append((noisy_labels == c).sum().item())
                target_class_idxs_list.append( (noisy_labels == c).nonzero(as_tuple=True)[0] )
            print("num_sample_per_class: ", num_sample_per_class)
            init_num_sample_per_class = num_sample_per_class.copy()

        # Prune4Rel
        with torch.no_grad():
            np.random.seed(random_seed)
            select_result = np.zeros(sample_num, dtype=bool)

            # Initialize NNconfs as 0
            NNconfs = torch.zeros([sample_num], requires_grad=False).to(device)

            # Greedy Selection Algorithm
            balance_break = False
            for i in range(budget):

                # Objective (Monotonic & Submodular)
                # FIXME: tanh quickly converge to 1, tanh(6)=1.0
                # FIXME: best eta? better than 10?
                selection_scores = torch.tanh((NNconfs+confs)/self.args.eta)-torch.tanh(NNconfs/self.args.eta)
                
                # Once selected, no more select
                selection_scores[select_result] = -1

                if self.args.balance == False or balance_break == True:
                    p = torch.argmax(selection_scores).item()
                    select_result[p] = True
                else:
                    # Round-robin target_label selection
                    target_label = available_classes[i % len(available_classes)]
                    target_class_idxs = target_class_idxs_list[target_label]

                    # if there's no more target_label examples to be selected, 
                    # remove it from available_class and move to next target_label
                    while num_sample_per_class[target_label] == 0:
                        if len(available_classes) == self.args.n_class:
                            min_balance = init_num_sample_per_class[target_label]

                        #print(num_sample_per_class)
                        #print("old: ", target_label)
                        available_classes = [c for c in available_classes if c != target_label]
                        target_label = available_classes[i % len(available_classes)]

                        #print("new: ", target_label)
                        target_class_idxs = target_class_idxs_list[target_label]

                        if init_num_sample_per_class[target_label]-num_sample_per_class[target_label]>1.5*min_balance:
                            balance_break = True

                    # Select!
                    p = target_class_idxs[torch.argmax(selection_scores[target_class_idxs])].item()
                    select_result[p] = True
                    num_sample_per_class[target_label] -= 1
                    
                if i%1000 == 0:
                    print("| Selecting [{:3d}/{:3d}], idx: {:d}, NNconf: {:.2f}, conf: {:.2f}, score: {:.8f}".format(
                        i + 1, budget, p, NNconfs[p].item(), confs[p].item(), selection_scores[p].item()))

                if i == (budget-1):
                    break

                # Calculate similarity_scores to the selected point p
                if self.metric_name == "euclidean":
                    dist = metric(embs, embs[[p]])
                    sim_to_selected = 1/(dist+0.1)-0.1   # [n_train, 1]
                    sim_to_selected = torch.clamp(sim_to_selected, min=0, max=10)
                elif self.metric_name == "cossim":
                    sim = metric(embs, embs[[p]])
                    sim_to_selected = sim*(sim > self.args.tau)
                    #print("How many # of NN in avg? ", (sim > const).sum())

                # Update Reduced Neighbor Confidences
                NNconfs += sim_to_selected.reshape(-1) * confs[p]
                #NNconfs[select_result] = 999999  # once selected, very confident, no more to be selected

        return np.arange(self.args.n_train)[select_result]

    def select(self, **kwargs):
        _, configs =self.run()

        # save
        if self.args.dataset in ['WebVision', 'Clothing1M', 'ImageNet']:
            probs_path = '/data/pdm102207/RobustCoreLogs/' + str(self.args.dataset) + '/' + str(self.args.robust_learner) \
                         + '/probs_nr'+ str(self.args.noise_rate)+'.t'
            embs_path = '/data/pdm102207/RobustCoreLogs/' + str(self.args.dataset) + '/' + str(self.args.robust_learner) \
                        + '/embs'+ str(self.args.noise_rate)+'.t'
            if os.path.isfile(probs_path) and os.path.isfile(embs_path):
                probs = torch.load(probs_path)
                embs = torch.load(embs_path)
                print("probs & embs loaded!")
            else:
                probs, embs = self.get_prob_embedding()

                torch.save(probs, probs_path)
                torch.save(embs, embs_path)
                print("probs & embs saved!")
        else:
            probs, embs = self.get_prob_embedding()

        selection_result = self.k_neighbor_confidence_greedy(probs, embs, budget=self.coreset_size,
                                           metric=self.metric, device=self.args.device,
                                           random_seed=self.random_seed, print_freq=self.args.print_freq)
        probs, embs = None, None
        del self.model
        torch.cuda.empty_cache()

        return {"indices": selection_result}, self.configs