from utils import *
import numpy as np
from sklearn.mixture import GaussianMixture
import sys

class ELRLoss(nn.Module):
    def __init__(self, args): #, num_examp, num_classes=10, beta=0.3
        super(ELRLoss, self).__init__()
        self.num_classes = args.n_class
        self.target = torch.zeros(args.n_train, self.num_classes).cuda(args.device)
        self.beta = args.beta
        self.lamda = args.lambda_elr

    def forward(self, index, output, label):
        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1 - self.beta) * \
                             ((y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1 - (self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + self.lamda * elr_reg
        return final_loss

def train_epoch(args, epoch, net, optimizer, scheduler, trainloader):
    net.train()
    criterion_elr = ELRLoss(args)

    num_iter = len(trainloader) + 1
    for batch_idx, (inputs, labels, indexs) in enumerate(trainloader):
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = net(inputs)

        loss = criterion_elr(indexs, outputs, labels) #indexs.cpu().detach().numpy().tolist()
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t ELR loss: %.2f'
                         % (args.dataset, args.noise_type, epoch, args.epochs, batch_idx + 1, num_iter,
                            loss.item()))
        sys.stdout.flush()
    if scheduler is not None:
        scheduler.step()


def train_ELR(args, networks, optimizers, schedulers, loader):
    net1 = networks['net1']
    optimizer1 = optimizers['optimizer1']
    scheduler1 = schedulers['scheduler1']

    best_acc = 0
    for epoch in range(args.selection_epochs, args.epochs):
        print("Epoch: {}".format(epoch))
        trainloader = loader.run('train')  # co-divide
        train_epoch(args, epoch, net1, optimizer1, scheduler1, trainloader)

        # TODO: best-acc & last-acc
        acc = test(epoch, net1, loader)
        best_acc = max(acc, best_acc)
    print("Best Accuracy: %.2f%%" % (best_acc))

    return acc, best_acc

def test(epoch, net1, loader):
    net1.eval()
    correct = 0
    total = 0

    test_loader = loader.run('test')
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net1(inputs)
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%" % (epoch, acc))
    net1.train()
    return acc