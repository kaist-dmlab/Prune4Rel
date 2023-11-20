from .coresetmethod import CoresetMethod
import torch, time
from torch import nn
import numpy as np
from ema_pytorch import EMA
from datetime import datetime
import os
import sys
import pickle

# To import robustlearner
import os, sys
sys.path.append('./../robustlearner/')
from .methods_utils.robust_loss import ELRLoss, ELRPLUSLoss, OverparametrizationLoss, mixup_data

class EarlyTrain(CoresetMethod):
    '''
    Core code for training related to coreset selection methods when pre-training is required.
    '''
    def __init__(self, loader, configs, args, robust_learner, fraction=0.5, random_seed=None, epochs=200,
                 torchvision_pretrain: bool = False, dst_pretrain_dict: dict = {}, fraction_pretrain=1., **kwargs):
        super().__init__(loader, configs, args, robust_learner, fraction, random_seed)
        self.epochs = epochs
        self.loader = loader
        self.robust_learner = robust_learner

        self.networks = configs[0]
        self.optimizers = configs[1]
        self.schedulers = configs[2]
        self.criterion = configs[3]

        self.model, self.optimizer, self.scheduler = self.networks['net1'], self.optimizers['optimizer1'], self.schedulers['scheduler1']
        if self.args.robust_learner in ['DivideMix', 'ELR_PLUS']:
            self.model2, self.optimizer2, self.scheduler2 = self.networks['net2'], self.optimizers['optimizer2'], \
                                                            self.schedulers['scheduler2']
        if self.args.robust_learner == 'SOP':
            self.optimizer_u, self.optimizer_v = self.optimizers['optimizer_u'], self.optimizers['optimizer_v']

        self.n_train = loader.n_train
        self.coreset_size = round(self.n_train * fraction)

    def load_configs(self):
        file_name = "/data/pdm102207/RobustCoreLogs/"+self.args.dataset+"/"+self.robust_learner+\
                    "/configs_"+str(self.args.noise_type)+"_warmup_"+str(self.args.selection_epochs)+"epochs.pickle"
        with open(file_name, 'rb') as f:
            configs = pickle.load(f)

        self.networks = configs[0]
        self.optimizers = configs[1]
        self.schedulers = configs[2]
        self.criterion = configs[3]

        self.model, self.optimizer, self.scheduler = self.networks['net1'], self.optimizers['optimizer1'], self.schedulers['scheduler1']
        if self.args.robust_learner in ['DivideMix', 'ELR_PLUS']:
            self.model2, self.optimizer2, self.scheduler2 = self.networks['net2'], self.optimizers['optimizer2'], \
                                                            self.schedulers['scheduler2']
        if self.args.robust_learner == 'SOP':
            self.optimizer_u, self.optimizer_v = self.optimizers['optimizer_u'], self.optimizers['optimizer_v']
        
        return configs

    def save_configs(self, configs):
         # save the warm-up configs

        folder = "/data/pdm102207/RobustCoreLogs/"+self.args.dataset+"/"+self.robust_learner
        if not os.path.exists(folder):
            print("===making folder at ", folder)
            os.makedirs(folder)

        file_name = folder+"/configs_"+str(self.args.noise_type)+"_warmup_"+str(self.args.selection_epochs)+"epochs"+\
                    "_gpu"+str(self.args.gpu)+".pickle"

        outfile = open(file_name, 'wb')
        pickle.dump(configs, outfile)
        outfile.close()

    def run(self):
        if self.args.pre_warmuped == True and self.args.selection != "Forgetting":
            print("Load the pre-warmuped configs")
            configs = self.load_configs()
        else:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            # self.train_indx = np.arange(self.n_train) # Forgetting

            start_train_time = time.time()
            if self.robust_learner == 'DivideMix':
                print("Warm-up training of Network 1 with DivideMix")
                self.train_DivideMix_warmup(self.model, self.optimizer,
                                            self.scheduler)  # self.model, self.optimizer, self.scheduler =
                print("Warm-up training of Network 2 with DivideMix")
                self.train_DivideMix_warmup(self.model2, self.optimizer2,
                                            self.scheduler2)  # self.model2, self.optimizer2, self.scheduler2 =

            elif self.robust_learner == 'ELR':
                print("Warm-up training of Network 1 with ELR")
                self.train_ELR_warmup(self.args, self.model, self.optimizer,
                                    self.scheduler)  # self.model, self.optimizer, self.scheduler =

            elif self.robust_learner == 'ELR_PLUS':
                print("Warm-up training of Network 1 & 2 with ELR_PLUS")
                self.train_ELR_PLUS_warmup(self.args)

            elif self.robust_learner == 'SOP':
                print("Warm-up training of Network 1 with SOP")
                self.train_SOP_warmup(self.args, self.model, self.optimizer, self.scheduler)

            elif self.robust_learner == 'CE':
                print('Warm-up training of Network 1 with CE')
                self.train_CE_warmup(self.args, self.model, self.optimizer, self.scheduler)

            print("Warmup Training Time: ", time.time() - start_train_time)

            # Update configuration
            self.networks['net1'], self.optimizers['optimizer1'], self.schedulers[
                'scheduler1'] = self.model, self.optimizer, self.scheduler
            if self.args.robust_learner in ['DivideMix', 'ELR_PLUS']:
                self.networks['net2'], self.optimizers['optimizer2'], self.schedulers[
                    'scheduler2'] = self.model2, self.optimizer2, self.scheduler2
            if self.args.robust_learner == 'SOP':
                self.optimizers['optimizer_u'], self.optimizers['optimizer_v'] = self.optimizer_u, self.optimizer_v

            configs = [self.networks, self.optimizers, self.schedulers, self.criterion]

            # save the warm-up configs
            self.save_configs(configs)
        
        # warm-up accuracy testing
        warmup_test_acc = self.test(self.epochs)

        return self.finish_run(), configs

    def train_SOP_warmup(self, args, net, optimizer, scheduler):
        run_time = str(datetime.now())
        acc_per_epoch = []
        pre_epoch = 0
        self.before_run()

        if args.pre_warmuped:
            checkpoint = torch.load(args.pre_warmed_filename)
            # print(checkpoint)
            pre_epoch = checkpoint['epoch'] + 1
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer1_state_dict'])
            self.optimizer_u.load_state_dict(checkpoint['optimizerU_state_dict'])
            self.optimizer_v.load_state_dict(checkpoint['optimizerV_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if pre_epoch == 5: scheduler.load_state_dict(checkpoint['scheduler'].state_dict())

        for epoch in range(pre_epoch, self.epochs):
            train_loader = self.loader.run('SOP')
            self.before_epoch()
            self.before_train()
            for i, data in enumerate(train_loader):
                inputs_w, inputs_s, targets, indexs = data[0], data[1], data[2], data[3]
                inputs_w, inputs_s, targets, indexs = inputs_w.to(self.args.device), inputs_s.to(self.args.device), \
                                                      targets.to(self.args.device), indexs.to(self.args.device)

                targets_ = torch.zeros(len(targets), self.args.n_class).to(self.args.device).scatter_(1,
                                                                                                      targets.view(-1,
                                                                                                                   1),
                                                                                                      1)

                if self.args.ratio_consistency > 0:
                    inputs_all = torch.cat([inputs_w, inputs_s]).cuda()
                else:
                    inputs_all = inputs_w

                outputs = net(inputs_all)

                # Forward propagation, compute loss, get predictions
                optimizer.zero_grad()
                self.optimizer_u.zero_grad()
                self.optimizer_v.zero_grad()

                loss = self.criterion(indexs, outputs, targets_)

                # for DeepCore
                if self.args.ratio_consistency > 0:
                    outputs, _ = torch.chunk(outputs, 2)

                self.after_loss(outputs, loss, targets, indexs,
                                epoch)  # TODO: Check if indexs is suitable for Forgetting
                self.while_update(outputs, loss, targets, epoch, i, self.args.selection_batch)

                loss.backward()

                optimizer.step()
                self.optimizer_u.step()
                self.optimizer_v.step()

            scheduler.step()
            self.after_epoch()
            warmup_test_acc = self.test(epoch)

            if args.save_log == True and False:
                acc_per_epoch.append((epoch, warmup_test_acc))
                log = np.array(acc_per_epoch).reshape((-1, 2))
                folder = '/home/sachoi/2023_1/RobustDataPruning/logs/' + str(args.dataset) + '/' + str(
                    args.robust_learner) + '/'
                filename = 'AccLog_warmup_' + str(args.noise_type) + '_r' + str(args.noise_rate) + '_' + str(
                    args.model) + '_' + str(run_time) + '.txt'

                if not os.path.exists(folder):
                    os.makedirs(folder)
                np.savetxt(folder + filename, log, fmt=["%d", "%f"])

                folder = '/data/pdm102207/RobustPruningLogs/checkpoint/' + str(args.dataset) + '/' + str(
                    args.robust_learner) + '/'
                filename = 'ckpt_warmup_' + str(epoch) + '_' + str(args.noise_type) + '_r' + str(
                    args.noise_rate) + '_' + str(args.model) + '_' + str(run_time) + '.tar'
                if not os.path.exists(folder):
                    os.makedirs(folder)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer1_state_dict': optimizer.state_dict(),
                    'optimizerU_state_dict': self.optimizer_u.state_dict(),
                    'optimizerV_state_dict': self.optimizer_v.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, folder + filename)

    def train_CE_warmup(self, args, net, optimizer, scheduler):
        self.before_run()
        for epoch in range(self.epochs):
            train_loader = self.loader.run('train')
            self.before_epoch()
            self.before_train()
            for i, data in enumerate(train_loader):
                inputs, targets, indexs = data[0], data[1], data[2]
                inputs, targets, indexs = inputs.to(self.args.device), targets.to(self.args.device), indexs.to(
                    self.args.device)

                # Forward propagation, compute loss, get predictions
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = self.criterion(outputs, targets)

                self.after_loss(outputs, loss, targets, indexs,
                                epoch)  # TODO: Check if indexs is suitable for Forgetting

                # Update loss, backward propagate, update optimizer
                self.while_update(outputs, loss, targets, epoch, i, self.args.selection_batch)

                loss.backward()
                optimizer.step()
            scheduler.step()
            self.after_epoch()

            warmup_test_acc = self.test(epoch)

    def train_ELR_PLUS_epoch(self, epoch, net, net_ema, net2_ema, train_loader, opt, sched):

        for i, data in enumerate(train_loader):
            inputs, targets, indexs = data[0], data[1], data[2]

            inputs_original = inputs

            targets = torch.zeros(len(targets), self.args.n_class).scatter_(1, targets.view(-1, 1), 1)
            inputs, targets, indexs = inputs.to(self.args.device), targets.float().to(self.args.device), indexs.to(
                self.args.device)

            inputs, targets, mixup_l, mix_index = mixup_data(inputs, targets, alpha=self.args.mixup_alpha,
                                                             device=self.args.device)
            outputs = net(inputs)

            inputs_original = inputs_original.to(self.args.device)
            output_original = net2_ema(inputs_original)
            output_original = output_original.data.detach()
            self.criterion.update_hist(epoch, output_original, indexs.cpu().numpy().tolist(), mix_index=mix_index,
                                       mixup_l=mixup_l)

            steps = epoch * len(train_loader) + i + 1
            loss, probs = self.criterion(steps, outputs, targets)

            self.after_loss(outputs, loss, targets, indexs, epoch)
            self.while_update(outputs, loss, targets, epoch, i, self.args.selection_batch)

            opt.zero_grad()
            loss.backward()

            opt.step()

            # TODO: EMA rampup!!!
            net_ema.update()
        sched.step()

    def train_ELR_PLUS_warmup(self, args):
        # TODO: PLUS VERSION~
        self.model.train()
        self.model2.train()

        self.model_ema = EMA(self.model, beta=args.gamma_elr_plus, update_after_step=0, update_every=1)
        self.model2_ema = EMA(self.model2, beta=args.gamma_elr_plus, update_after_step=0, update_every=1)

        # criterion_elr_plus = ELRPLUSLoss(args)  # .to(self.args.device)

        self.before_run()
        for epoch in range(self.epochs):
            train_loader = self.loader.run('train')
            self.before_epoch()
            self.before_train()

            self.train_ELR_PLUS_epoch(epoch, self.model, self.model_ema, self.model2_ema, train_loader, self.optimizer,
                                      self.scheduler)
            self.train_ELR_PLUS_epoch(epoch, self.model2, self.model2_ema, self.model_ema, train_loader,
                                      self.optimizer2, self.scheduler2)
            self.after_epoch()

        self.networks['net1_ema'], self.networks['net2_ema'] = self.model_ema, self.model2_ema

    def train_ELR_warmup(self, args, net, optimizer, scheduler):
        # criterion_elr = ELRLoss(args) #.to(self.args.device)

        self.before_run()
        for epoch in range(self.epochs):
            train_loader = self.loader.run('train')
            self.before_epoch()
            self.before_train()
            for i, data in enumerate(train_loader):
                inputs, targets, indexs = data[0], data[1], data[2]
                inputs, targets, indexs = inputs.to(self.args.device), targets.to(self.args.device), indexs.to(
                    self.args.device)

                # Forward propagation, compute loss, get predictions
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = self.criterion(indexs, outputs, targets)

                self.after_loss(outputs, loss, targets, indexs, epoch)

                # Update loss, backward propagate, update optimizer
                loss = loss.mean()

                self.while_update(outputs, loss, targets, epoch, i, self.args.selection_batch)

                loss.backward()
                optimizer.step()
            scheduler.step()
            self.after_epoch()

    def train_DivideMix_warmup(self, net, optimizer, scheduler):
        # criterion = nn.CrossEntropyLoss().to(self.args.device)

        self.before_run()
        for epoch in range(self.epochs):
            train_loader = self.loader.run('train')
            self.before_epoch()
            self.before_train()
            for i, data in enumerate(train_loader):
                inputs, targets, indexs = data[0], data[1], data[2]
                inputs, targets, indexs = inputs.to(self.args.device), targets.to(self.args.device), indexs.to(
                    self.args.device)

                # Forward propagation, compute loss, get predictions
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = self.criterion(outputs, targets)

                self.after_loss(outputs, loss, targets, indexs, epoch)

                # Update loss, backward propagate, update optimizer
                loss = loss.mean()

                self.while_update(outputs, loss, targets, epoch, i, self.args.selection_batch)

                loss.backward()
                optimizer.step()
            scheduler.step()
            self.after_epoch()

        # return net, optimizer, scheduler

    def test(self, epoch):
        self.model.eval()

        test_loader = self.loader.run('test')
        correct = 0.
        total = 0.
        print('=> Warm-up Testing Epoch #%d' % epoch)
        for batch_idx, data in enumerate(test_loader):
            input, target = data[0], data[1]
            output = self.model(input.to(self.args.device))
            predicted = torch.max(output.data, 1).indices.cpu()

            correct += predicted.eq(target).sum().item()
            total += target.size(0)
        warmup_test_acc = 100. * correct / total
        print(' Test Acc: %.3f%%' % warmup_test_acc)
        self.model.train()
        return warmup_test_acc

    def num_classes_mismatch(self):
        pass

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        pass

    def finish_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def finish_run(self):
        pass

    def select(self, **kwargs):
        selection_result, configs = self.run()
        return selection_result, configs

