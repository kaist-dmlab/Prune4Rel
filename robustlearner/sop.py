from utils import *
import numpy as np
import sys

class OverparametrizationLoss(torch.nn.Module):
    def __init__(self, args): #, num_examp, num_classes=10, ratio_consistency=0, ratio_balance=0):
        super(OverparametrizationLoss, self).__init__()
        self.args = args
        self.num_classes = args.n_class
        self.num_examp = args.n_train

        self.ratio_consistency = args.ratio_consistency
        self.ratio_balance = args.ratio_balance

        self.u = torch.nn.Parameter(torch.empty(self.num_examp, 1).to(args.device))
        self.v = torch.nn.Parameter(torch.empty(self.num_examp, self.num_classes).to(args.device))
        self.init_param(mean=0, std=1e-8)

    def init_param(self, mean=0., std=1e-8):
        torch.nn.init.normal_(self.u, mean=mean, std=std)
        torch.nn.init.normal_(self.v, mean=mean, std=std)

    def forward(self, index, outputs, label):
        if len(outputs) > len(index):
            output, output2 = torch.chunk(outputs, 2)
        else:
            output = outputs

        eps = 1e-4

        U_square = self.u[index] ** 2 * label
        V_square = self.v[index] ** 2 * (1 - label)

        U_square = torch.clamp(U_square, 0, 1)
        V_square = torch.clamp(V_square, 0, 1)

        E = U_square - V_square

        original_prediction = F.softmax(output, dim=1)
        prediction = torch.clamp(original_prediction + U_square - V_square.detach(), min=eps)
        prediction = F.normalize(prediction, p=1, eps=eps)
        prediction = torch.clamp(prediction, min=eps, max=1.0)

        loss_CE = torch.mean(-torch.sum((label) * torch.log(prediction), dim=-1))

        label_one_hot = self.soft_to_hard(output.detach())
        loss_MSE = F.mse_loss((label_one_hot + U_square - V_square), label, reduction='sum') / len(label)

        loss = loss_CE + loss_MSE

        if self.ratio_balance > 0:
            avg_prediction = torch.mean(prediction, dim=0)
            prior_distr = 1.0 / self.num_classes * torch.ones_like(avg_prediction)

            avg_prediction = torch.clamp(avg_prediction, min=eps, max=1.0)

            balance_kl = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))

            loss += self.ratio_balance * balance_kl

        if (len(outputs) > len(index)) and (self.ratio_consistency > 0):
            consistency_loss = self.consistency_loss(index, output, output2)

            loss += self.ratio_consistency * torch.mean(consistency_loss)

        return loss

    def consistency_loss(self, index, output1, output2):
        preds1 = F.softmax(output1, dim=1).detach()
        preds2 = F.log_softmax(output2, dim=1)
        loss_kldiv = F.kl_div(preds2, preds1, reduction='none')
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        return loss_kldiv

    def soft_to_hard(self, x):
        with torch.no_grad():
            return (torch.zeros(len(x), self.num_classes)).to(self.args.device).scatter_(1, (x.argmax(dim=1)).view(-1, 1), 1)
        

def train_epoch(args, epoch, networks, optimizers, schedulers, criterion, trainloader):
    net = networks['net1']
    net.train()

    num_iter = len(trainloader) + 1
    for batch_idx, data in enumerate(trainloader):
        inputs_w, inputs_s, targets, indexs = data[0], data[1], data[2], data[3]
        inputs_w, inputs_s, targets, indexs = inputs_w.to(args.device), inputs_s.to(args.device), \
                                              targets.to(args.device), indexs.to(args.device)

        targets_ = torch.zeros(len(targets), args.n_class).to(args.device).scatter_(1, targets.view(-1, 1), 1)

        if args.ratio_consistency > 0:
            inputs_all = torch.cat([inputs_w, inputs_s]).to(args.device)
        else:
            inputs_all = inputs_w

        outputs = net(inputs_all)

        # Forward propagation, compute loss, get predictions
        optimizers['optimizer1'].zero_grad()
        optimizers['optimizer_u'].zero_grad()
        optimizers['optimizer_v'].zero_grad()

        loss = criterion(indexs, outputs, targets_)

        # for DeepCore
        if args.ratio_consistency > 0:
            outputs, _ = torch.chunk(outputs, 2)

        loss.backward()

        optimizers['optimizer1'].step()
        optimizers['optimizer_u'].step()
        optimizers['optimizer_v'].step()

        sys.stdout.write('\r')
        sys.stdout.write('%s-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t SOP loss: %.2f'
                         % (args.dataset, args.noise_type, epoch, args.epochs, batch_idx + 1, num_iter,
                            loss.item()))
        sys.stdout.flush()

    if schedulers['scheduler1'] is not None:
        schedulers['scheduler1'].step()

def train_SOP(args, networks, optimizers, schedulers, criterion, loader):
    best_acc = 0
    for epoch in range(1, args.epochs+1):
        print("Epoch: {}".format(epoch))
        trainloader = loader.run('SOP')
        train_epoch(args, epoch, networks, optimizers, schedulers, criterion, trainloader)

        acc = test(args, epoch, networks['net1'], loader)
        best_acc = max(acc, best_acc)
    print("Best Accuracy: %.2f%%" % (best_acc))

    return acc, best_acc

def test(args, epoch, net1, loader):
    net1.eval()
    correct = 0
    total = 0

    test_loader = loader.run('test')
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net1(inputs)
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%" % (epoch, acc))
    net1.train()
    return acc