from utils import *
import numpy as np
from sklearn.mixture import GaussianMixture
import sys
from ema_pytorch import EMA

# TODO: CHECK THIS!!!!!!!!!!!
class ELRPLUSLoss(torch.nn.Module):
    def __init__(self, args): #num_examp, config, device, num_classes=10, beta=0.3
        super(ELRPLUSLoss, self).__init__()
        self.num_classes = args.n_class
        self.pred_hist = torch.zeros(args.n_train, self.num_classes).to(args.device)
        self.beta = args.beta
        self.lamda = args.lambda_elr
        self.q = 0

    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from  2"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def forward(self, iteration, output, y_labeled):
        y_pred = torch.nn.functional.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)

        if self.num_classes == 100:
            y_labeled = y_labeled * self.q
            y_labeled = y_labeled / (y_labeled).sum(dim=1, keepdim=True)

        ce_loss = torch.mean(-torch.sum(y_labeled * torch.nn.functional.log_softmax(output, dim=1), dim=-1))
        reg = ((1 - (self.q * y_pred).sum(dim=1)).log()).mean()

        #print(ce_loss.item(), reg.item())

        final_loss = ce_loss + self.sigmoid_rampup(iteration, 0)*(self.lamda * reg)

        return final_loss, y_pred.cpu().detach()

    def update_hist(self, epoch, out, index=None, mix_index=..., mixup_l=1):
        y_pred_ = torch.nn.functional.softmax(out, dim=1)
        self.pred_hist[index] = self.beta*self.pred_hist[index] + (1-self.beta) * y_pred_/(y_pred_).sum(dim=1,keepdim=True)
        self.q = mixup_l * self.pred_hist[index] + (1 - mixup_l) * self.pred_hist[index][mix_index]


def mixup_data(x, y, alpha=1.0, device='cpu'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)
        batch_size = x.size()[0]
        mix_index = torch.randperm(batch_size).to(device)

        mixed_x = lam * x + (1 - lam) * x[mix_index, :]  #
        mixed_target = lam * y + (1 - lam) * y[mix_index, :]

        return mixed_x, mixed_target, lam, mix_index
    else:
        lam = 1
        return x, y, lam, ...


def train_epoch(args, epoch, net, net_ema, net2_ema, train_loader, criterion, opt, sched):
    num_iter = len(train_loader) + 1
    for i, data in enumerate(train_loader):
        inputs, targets, indexs = data[0], data[1], data[2]

        inputs_original = inputs

        targets = torch.zeros(len(targets), args.n_class).scatter_(1, targets.view(-1, 1), 1)
        inputs, targets, indexs = inputs.to(args.device), targets.float().to(args.device), indexs.to(args.device)

        inputs, targets, mixup_l, mix_index = mixup_data(inputs, targets, alpha=args.mixup_alpha, device=args.device)
        outputs = net(inputs)

        inputs_original = inputs_original.to(args.device)
        output_original = net2_ema(inputs_original)
        output_original = output_original.data.detach()
        criterion.update_hist(epoch, output_original, indexs.cpu().numpy().tolist(), mix_index=mix_index, mixup_l=mixup_l)

        steps = epoch * len(train_loader) + i + 1
        loss, probs = criterion(steps, outputs, targets)

        opt.zero_grad()
        loss.backward()

        opt.step()

        # TODO: EMA rampup!!!
        net_ema.update()

        sys.stdout.write('\r')
        sys.stdout.write('%s-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t ELR_PLUS loss: %.2f'
                         % (args.dataset, args.noise_type, epoch, args.epochs, i + 1, num_iter,
                            loss.item()))
        sys.stdout.flush()

    if sched is not None:
        sched.step()

def train_ELR_PLUS(args, networks, optimizers, schedulers, criterion, loader):
    net1, net2 = networks['net1'], networks['net2']
    opt1, opt2 = optimizers['optimizer1'], optimizers['optimizer2']
    sched1, sched2 = schedulers['scheduler1'], schedulers['scheduler2']

    net1_ema, net2_ema = networks['net1_ema'], networks['net2_ema']

    #net1_ema = EMA(net1, beta=args.gamma_elr_plus, update_after_step=0, update_every=1)
    #net2_ema = EMA(net2, beta=args.gamma_elr_plus, update_after_step=0, update_every=1)
    #criterion_elr_plus = ELRPLUSLoss(args)  # .to(self.args.device)

    net1.train()
    net2.train()
    best_acc = 0
    for epoch in range(args.selection_epochs, args.epochs):
        print("Epoch: {}".format(epoch))
        train_loader = loader.run('train')  # co-divide
        train_epoch(args, epoch, net1, net1_ema, net2_ema, train_loader, criterion, opt1, sched1)
        train_epoch(args, epoch, net2, net2_ema, net1_ema, train_loader, criterion, opt2, sched2)

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