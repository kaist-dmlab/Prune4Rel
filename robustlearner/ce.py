from utils import *
import numpy as np
import sys
from datetime import datetime
import os

def train_epoch(args, epoch, networks, optimizers, schedulers, criterion, trainloader):
    net = networks['net1']
    net.train()

    num_iter = len(trainloader) + 1
    for batch_idx, data in enumerate(trainloader):
        inputs, targets, indexs = data[0], data[1], data[2]
        inputs, targets, indexs = inputs.to(args.device), targets.to(args.device), indexs.to(args.device)

        outputs = net(inputs)
        # print(inputs.shape, outputs.shape, targets.shape)
        # Forward propagation, compute loss, get predictions
        optimizers['optimizer1'].zero_grad()

        loss = criterion(outputs, targets)

        # for DeepCore
        if args.ratio_consistency > 0:
            outputs, _ = torch.chunk(outputs, 2)

        loss.backward()

        optimizers['optimizer1'].step()

        sys.stdout.write('\r')
        sys.stdout.write('%s-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t SOP loss: %.2f'
                         % (args.dataset, args.noise_type, epoch, args.epochs, batch_idx + 1, num_iter,
                            loss.item()))
        sys.stdout.flush()

    if schedulers['scheduler1'] is not None:
        schedulers['scheduler1'].step()

def train_CE(args, networks, optimizers, schedulers, criterion, loader):
    best_acc = 0
    run_time = str(datetime.now())
    acc_per_epoch = []

    for epoch in range(1, args.epochs + 1):
        print("Epoch: {}".format(epoch))
        trainloader = loader.run('train')
        train_epoch(args, epoch, networks, optimizers, schedulers, criterion, trainloader)

        acc = test(args, epoch, networks['net1'], loader)
        best_acc = max(acc, best_acc)
        if args.save_log == True:
            acc_per_epoch.append((epoch, acc))
            log = np.array(acc_per_epoch).reshape((-1, 2))
            folder = '/home/sachoi/RobustDataPruning/logs/' + str(args.dataset) + '/' + str(args.robust_learner) + '/' \
                     + str(args.selection) + '/'
            filename = 'ACC_' + str(args.noise_type) + '_r' + str(args.noise_rate) + '_' + str(
                args.model) + '_fr' + str(args.fraction) \
                       + '_' + str(run_time) + '.txt'
            if not os.path.exists(folder):
                os.makedirs(folder)
            np.savetxt(folder + filename, log, fmt=["%d", "%f"])

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

