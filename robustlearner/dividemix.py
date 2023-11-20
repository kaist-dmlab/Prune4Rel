from utils import *
import numpy as np
from sklearn.mixture import GaussianMixture
import sys


class SemiLoss(object):
    def __init__(self, lambda_u):
        self.lambda_u = lambda_u

    def linear_rampup(self, current, warm_up, rampup_length=16):
        current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0) # this is ok
        return self.lambda_u # * float(current)

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        #print(probs_u, targets_u)

        return Lx, Lu, self.linear_rampup(epoch, warm_up)

def eval_train(model, all_loss, eval_loader, loader, criterion):
    model.eval()

    losses = torch.zeros(loader.n_train)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b] #.detach() #??? print(losses)

    losses = losses[loader.coreset_idxs]

    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    input_loss = losses.reshape(-1, 1)

    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob, all_loss


def train_MixMatch(args, epoch, net, net2, optimizer, scheduler, labeled_trainloader, unlabeled_trainloader):
    net.train()
    net2.eval()  # fix one network and train the other

    criterion = SemiLoss(args.lambda_u)

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = len(labeled_trainloader) + 1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.n_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)

            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(
                outputs_u22, dim=1)) / 4
            ptu = pu ** (1 / args.T)  # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)  # temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

            # mixmatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits = net(mixed_input)
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        # CIFAR10
        #Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size * 2], logits_u, mixed_target[batch_size * 2:],
        #                         epoch + batch_idx / num_iter, 0) #warm_up - already done in RobustCore

        # CIFAR100
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size * 2], logits_u, mixed_target[batch_size * 2:],
                                 epoch, 0)  # warm_up - already done in RobustCore

        # regularization
        prior = torch.ones(args.n_class) / args.n_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + lamb * Lu + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.6f Lambda_U: %.2f'
                         % (args.dataset, args.noise_type, epoch, args.epochs, batch_idx + 1, num_iter,
                            Lx.item(), Lu.item(), lamb))
        sys.stdout.flush()
    if scheduler is not None:
        scheduler.step()

def train_DivideMix(args, networks, optimizers, schedulers, loader):
    net1, net2 = networks['net1'], networks['net2']
    optimizer1, optimizer2 = optimizers['optimizer1'], optimizers['optimizer2']
    scheduler1, scheduler2 = schedulers['scheduler1'], schedulers['scheduler2']

    all_loss = [[], []]  # save the history of losses from two networks
    best_acc = 0
    for epoch in range(args.epochs):
        print("Epoch: {}".format(epoch))

        eval_loader = loader.run('eval_train')
        criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
        prob1, all_loss[0] = eval_train(net1, all_loss[0], eval_loader, loader, criterion)
        prob2, all_loss[1] = eval_train(net2, all_loss[1], eval_loader, loader, criterion)

        pred1 = (prob1 > args.p_threshold) # len: # of coreset, type: [True, False, ...]
        pred2 = (prob2 > args.p_threshold)

        #print('\nTrain Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('DivideMix', pred2, prob2)  # co-divide
        train_MixMatch(args, epoch, net1, net2, optimizer1, scheduler1, labeled_trainloader, unlabeled_trainloader)  # train net1

        #print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('DivideMix', pred1, prob1)  # co-divide
        train_MixMatch(args, epoch, net2, net1, optimizer2, scheduler2, labeled_trainloader, unlabeled_trainloader)  # train net2

        # TODO: best-acc & last-acc
        acc = test(epoch, net1, net2, loader)
        best_acc = max(acc, best_acc)
    print("Best Accuracy: %.2f%%" % (best_acc))

    return acc, best_acc

def test(epoch, net1, net2, loader):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0

    test_loader = loader.run('test')
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%" % (epoch, acc))
    net1.train()
    return acc