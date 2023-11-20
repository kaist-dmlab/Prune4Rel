import time
from argparse import ArgumentTypeError
from prefetch_generator import BackgroundGenerator
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from robustlearner.dividemix import SemiLoss
from robustlearner.elr import ELRLoss
from robustlearner.elr_plus import ELRPLUSLoss
from robustlearner.sop import OverparametrizationLoss


# Remove if useless
class WeightedSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices, weights) -> None:
        self.dataset = dataset
        assert len(indices) == len(weights)
        self.indices = indices
        self.weights = weights

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]], self.weights[[i for i in idx]]
        return self.dataset[self.indices[idx]], self.weights[idx]


def train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted: bool = False):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to train mode
    network.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        if if_weighted:
            target = data[0][1].to(args.device)
            input = data[0][0].to(args.device)

            # Compute output
            output = network(input)
            weights = data[1].to(args.device).requires_grad_(False)
            loss = torch.sum(criterion(output, target) * weights) / torch.sum(weights)
        else:
            target = data[1].to(args.device)
            input = data[0].to(args.device)

            # Compute output
            output = network(input)
            loss = criterion(output, target).mean()
            # print(loss)

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i == len(train_loader) - 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))

    record_train_stats(rec, epoch, losses.avg, top1.avg, optimizer.state_dict()['param_groups'][0]['lr'])


def test(test_loader, network, criterion, epoch, args, rec):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # Switch to evaluate mode
    network.eval()
    network.no_grad = True

    end = time.time()
    for i, data in enumerate(test_loader):
        input, target = data[0], data[1]

        target = target.to(args.device)
        input = input.to(args.device)

        # Compute output
        with torch.no_grad():
            output = network(input)

            loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        '''
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(test_loader), batch_time=batch_time, loss=losses,
                top1=top1))
        '''
    print('Test acc: * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    network.no_grad = False

    record_test_stats(rec, epoch, losses.avg, top1.avg)
    return top1.avg


def get_train_loss_acc(train_loader, network, criterion, epoch, args, rec):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # Switch to evaluate mode
    network.eval()
    network.no_grad = True

    end = time.time()
    for i, data in enumerate(train_loader):
        input, target = data[0], data[1]

        target = target.to(args.device)
        input = input.to(args.device)

        # Compute output
        with torch.no_grad():
            output = network(input)

            loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

    network.no_grad = False
    return top1.avg, losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def str_to_bool(v):
    # Handle boolean type in arguments.
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def save_checkpoint(state, path, epoch, prec):
    print("=> Saving checkpoint for epoch %d, with Prec@1 %f." % (epoch, prec))
    torch.save(state, path)


def init_recorder():
    from types import SimpleNamespace
    rec = SimpleNamespace()
    rec.train_step = []
    rec.train_loss = []
    rec.train_acc = []
    rec.lr = []
    rec.test_step = []
    rec.test_loss = []
    rec.test_acc = []
    rec.ckpts = []
    return rec


def record_train_stats(rec, step, loss, acc, lr):
    rec.train_step.append(step)
    rec.train_loss.append(loss)
    rec.train_acc.append(acc)
    rec.lr.append(lr)
    return rec


def record_test_stats(rec, step, loss, acc):
    rec.test_step.append(step)
    rec.test_loss.append(loss)
    rec.test_acc.append(acc)
    return rec


def record_ckpt(rec, step):
    rec.ckpts.append(step)
    return rec


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_more_args(args):
    gpus = ""
    if args.gpu is not None:
        for i, g in enumerate(args.gpu):
            gpus = gpus + str(g)
            if i != len(args.gpu) - 1:
                gpus = gpus + ","

        state = {k: v for k, v in args._get_kwargs()}
        # if args.dataset == 'ImageNet':
        #     args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if args.multi_gpu == True:
            print(gpus)
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            args.device = 'cuda:' + str(gpus) if torch.cuda.is_available() else 'cpu'
    else:
        args.device = 'cpu'

    if args.train_batch is None:
        args.train_batch = args.batch_size
    if args.selection_batch is None:
        args.selection_batch = args.batch_size
    if args.save_path != "" and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if args.resume != "":
        # Load checkpoint
        try:
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=args.device)
            assert {"exp", "epoch", "state_dict", "opt_dict", "best_acc1", "rec", "subset", "sel_args"} <= set(
                checkpoint.keys())
            assert 'indices' in checkpoint["subset"].keys()
            start_exp = checkpoint['exp']
            start_epoch = checkpoint["epoch"]
        except AssertionError:
            try:
                assert {"exp", "subset", "sel_args"} <= set(checkpoint.keys())
                assert 'indices' in checkpoint["subset"].keys()
                print("=> The checkpoint only contains the subset, training will start from the begining")
                start_exp = checkpoint['exp']
                start_epoch = 0
            except AssertionError:
                print("=> Failed to load the checkpoint, an empty one will be created")
                checkpoint = {}
                start_exp = 0
                start_epoch = 0
    else:
        checkpoint = {}
        start_exp = 0
        start_epoch = 0

    return args, checkpoint, start_exp, start_epoch


def init_param(u, v, mean=0., std=1e-8):
    torch.nn.init.normal_(u, mean=mean, std=std)
    torch.nn.init.normal_(v, mean=mean, std=std)


def get_configuration(args, nets, model, checkpoint):

    if args.robust_learner == 'DivideMix':
        criterion = nn.CrossEntropyLoss()
        # criterion = SemiLoss(args.lambda_u) # Will call later after warm-up training
    elif args.robust_learner == 'ELR':
        criterion = ELRLoss(args)
    elif args.robust_learner == 'ELR_PLUS':
        criterion = ELRPLUSLoss(args)
    elif args.robust_learner == 'SOP':
        criterion = OverparametrizationLoss(args)
    else:
        criterion = nn.CrossEntropyLoss().to(args.device)

    # For Standard, ELR, and SOP, where only a single network is trained
    net1 = nets.__dict__[model](args.channel, args.n_class, args.im_size, pretrained=args.pre_trained).to(args.device)

    if args.device == "cpu":
        print("Using CPU.")
    elif args.multi_gpu == True:
        net1 = nets.nets_utils.MyDataParallel(net1)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu[0])
        net1 = nets.nets_utils.MyDataParallel(net1, device_ids=args.gpu)
    elif torch.cuda.device_count() > 1:
        net1 = nets.nets_utils.MyDataParallel(net1).cuda()

    # TODO: if using a pre-trained model
    if "state_dict" in checkpoint.keys():
        # Loading model state_dict
        net1.load_state_dict(checkpoint["state_dict"])

    # Optimizer
    if args.optimizer == "SGD":
        optimizer1 = torch.optim.SGD(net1.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                     nesterov=args.nesterov)
    elif args.optimizer == "Adam":
        optimizer1 = torch.optim.Adam(net1.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        optimizer1 = torch.optim.__dict__[args.optimizer](net1.parameters(), args.lr, momentum=args.momentum,
                                                          weight_decay=args.weight_decay, nesterov=args.nesterov)
        
    # LR scheduler
    if args.scheduler == "CosineAnnealingLR":
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == "StepLR":
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "MultiStepLR":
        scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1,
                                                          milestones=[args.epochs / 2])  # making half epoch milestone
    else:
        scheduler1 = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer1)

    networks = {'net1': net1}
    optimizers = {'optimizer1': optimizer1}
    schedulers = {'scheduler1': scheduler1}

    # For DivideMix, ELR+, and SOP+, where two networks are trained
    if args.robust_learner in ['DivideMix', 'ELR_PLUS']:
        net2 = nets.__dict__[model](args.channel, args.n_class, args.im_size, pretrained=args.pre_trained).to(args.device)

        if args.device == "cpu":
            print("Using CPU.")
        elif args.multi_gpu == True:
            # torch.cuda.set_device(args.gpu[0])
            net2 = nets.nets_utils.MyDataParallel(net2)
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu[0])
            net2 = nets.nets_utils.MyDataParallel(net2, device_ids=args.gpu)
        elif torch.cuda.device_count() > 1:
            net2 = nets.nets_utils.MyDataParallel(net2).cuda()

        # TODO: if using a pre-trained model
        if "state_dict" in checkpoint.keys():
            # Loading model state_dict
            net2.load_state_dict(checkpoint["state_dict"])

        # Optimizer
        if args.optimizer == "SGD":
            optimizer2 = torch.optim.SGD(net2.parameters(), args.lr, momentum=args.momentum,
                                         weight_decay=args.weight_decay, nesterov=args.nesterov)
        elif args.optimizer == "Adam":
            optimizer2 = torch.optim.Adam(net2.parameters(), args.lr, weight_decay=args.weight_decay)
        else:
            optimizer2 = torch.optim.__dict__[args.optimizer](net2.parameters(), args.lr, momentum=args.momentum,
                                                              weight_decay=args.weight_decay, nesterov=args.nesterov)

        # LR scheduler
        if args.scheduler == "CosineAnnealingLR":
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.epochs, eta_min=args.min_lr)
        elif args.scheduler == "StepLR":
            scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=args.step_size, gamma=args.gamma)
        elif args.scheduler == "MultiStepLR":
            scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[150])
        else:
            scheduler2 = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer2)

        networks['net2'] = net2
        optimizers['optimizer2'] = optimizer2
        schedulers['scheduler2'] = scheduler2

    # For SOP, instance-wise parameter
    if args.robust_learner == 'SOP':
        optimizer_u = torch.optim.SGD([criterion.u], lr=args.lr_u, weight_decay=0)
        optimizer_v = torch.optim.SGD([criterion.v], lr=args.lr_v, weight_decay=0)

        optimizers['optimizer_u'], optimizers['optimizer_v'] = optimizer_u, optimizer_v

    return networks, optimizers, schedulers, criterion


def get_fresh_configuration(args, nets, model, train_loader, start_epoch):
    network = nets.__dict__[model](args.channel, args.num_classes, args.im_size).to(args.device)

    if args.device == "cpu":
        print("Using CPU.")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu[0])
        network = nets.nets_utils.MyDataParallel(network, device_ids=args.gpu)
    elif torch.cuda.device_count() > 1:
        network = nets.nets_utils.MyDataParallel(network).cuda()

    criterion = nn.CrossEntropyLoss().to(args.device)

    # Optimizer
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(network.parameters(), args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(network.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.__dict__[args.optimizer](network.parameters(), args.lr, momentum=args.momentum,
                                                         weight_decay=args.weight_decay, nesterov=args.nesterov)

    # LR scheduler
    if args.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs,
                                                               eta_min=args.min_lr)
    elif args.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) * args.step_size,
                                                    gamma=args.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer)
    # scheduler.last_epoch = (start_epoch - 1) * len(train_loader)

    rec = init_recorder()

    return network, criterion, optimizer, scheduler, rec


def get_model(args, nets, model):
    network = nets.__dict__[model](args.channel, args.num_classes, args.im_size).to(args.device)

    if args.device == "cpu":
        print("Using CPU.")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu[0])
        network = nets.nets_utils.MyDataParallel(network, device_ids=args.gpu)
    elif torch.cuda.device_count() > 1:
        network = nets.nets_utils.MyDataParallel(network).cuda()

    return network


def get_optim_configurations(args, network, train_loader, start_epoch=0):
    print("lr: {}, momentum: {}, decay: {}".format(args.lr, args.momentum, args.weight_decay))

    criterion = nn.CrossEntropyLoss().to(args.device)

    # Optimizer
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(network.parameters(), args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(network.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.__dict__[args.optimizer](network.parameters(), args.lr, momentum=args.momentum,
                                                         weight_decay=args.weight_decay, nesterov=args.nesterov)

    # LR scheduler
    if args.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs,
                                                               eta_min=args.min_lr)
    elif args.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) * args.step_size,
                                                    gamma=args.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer)
    # scheduler.last_epoch = (start_epoch - 1) * len(train_loader)

    rec = init_recorder()

    return criterion, optimizer, scheduler, rec


def get_optim_configurations_epochs(args, network, start_epoch=0):
    criterion = nn.CrossEntropyLoss().to(args.device)

    # Optimizer
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(network.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(network.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.__dict__[args.optimizer](network.parameters(), args.lr, momentum=args.momentum,
                                                         weight_decay=args.weight_decay, nesterov=args.nesterov)

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    rec = init_recorder()

    return criterion, optimizer, scheduler, rec


def calc_jacobian(args, prob, target):
    target_oh = torch.zeros([prob.shape[0], prob.shape[1]], requires_grad=False).to(args.device)
    for i in range(target.shape[0]):
        target_oh[i, target[i]] = 1

    temp = target_oh - prob

    left = torch.unsqueeze(temp, 2)
    right = torch.unsqueeze(prob, 1)

    jacobian = torch.matmul(left, right)
    # print(left.shape, right.shape, jacobian.shape)

    return jacobian, -temp.unsqueeze(1)


def save_2nd_derivative(args, epoch, dst_train, model, criterion, optimizer, noise_idx):
    losses = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    probs = torch.zeros([args.n_train, args.num_classes], requires_grad=False).to(args.device)
    frob_norms = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    is_noise = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)

    for i in range(args.n_train):
        if i in noise_idx:
            is_noise[i] = 1

    data_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.selection_batch, num_workers=args.workers)
    embedding_dim = model.get_last_layer().in_features
    model.embedding_recorder.record_embedding = True
    for i, data in enumerate(data_loader):
        input, target, index = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)

        optimizer.zero_grad()
        output = model(input)
        prob = torch.nn.functional.softmax(output.requires_grad_(True), dim=1)

        loss_batch = criterion(output, target)

        batch_num = target.shape[0]
        with torch.no_grad():
            # jacobian_matrices = torch.autograd.functional.jacobian(prob, output)[0]  # [batch_size (128), n_class (10), n_class (10)]
            jacobian_matrices, first_derivs = calc_jacobian(args, prob, target)

            # jacobian_matrices, first_derivs = jacobian_matrices.reshape((jacobian_matrices.shape[0], -1))
            # print(jacobian_matrices.shape)
            grad_norm_delta = torch.matmul(first_derivs, jacobian_matrices).reshape((-1, prob.shape[1]))

            # print(jacobian_matrices.shape, first_derivs.shape, grad_norm_delta.shape)

            second_deriv_norm = torch.norm(grad_norm_delta, dim=1, p=2)
            # bias_parameters_grads.view(batch_num,args.num_classes,1).repeat(1, 1, embedding_dim)

            # TODO: why p_norm*z_norm != g_norm, how to calculate g_norm?
            losses[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = loss_batch.detach()
            probs[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), :] = prob
            frob_norms[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train),
            0] = second_deriv_norm

    print(losses.shape, prob.shape, second_deriv_norm.shape)

    logs = torch.cat((losses, probs, frob_norms, is_noise), 1).cpu().numpy()
    print(logs[:30])

    # TODO: save
    file_name = 'logs/' + args.dataset + '/labelnoise20_epoch' + str(epoch) + '_loss_prob_second_norms.txt'
    np.savetxt(file_name, logs, fmt="%.4f", delimiter=',')

    return 0


def save_loss_g_logs(args, epoch, dst_train, model, criterion, optimizer, noise_idx):
    losses = torch.zeros([args.n_train, args.repeat], requires_grad=False).to(args.device)
    p_norms = torch.zeros([args.n_train, args.repeat], requires_grad=False).to(args.device)
    z_norms = torch.zeros([args.n_train, args.repeat], requires_grad=False).to(args.device)
    g_norms = torch.zeros([args.n_train, args.repeat], requires_grad=False).to(args.device)
    g_norms2 = torch.zeros([args.n_train, args.repeat], requires_grad=False).to(args.device)
    w_norms = torch.zeros([args.n_train, args.repeat], requires_grad=False).to(args.device)
    is_noise = torch.zeros([args.n_train, args.repeat], requires_grad=False).to(args.device)
    for i in range(args.n_train):
        if i in noise_idx:
            is_noise[i] = 1

    data_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.selection_batch, num_workers=args.workers)
    embedding_dim = model.get_last_layer().in_features
    model.embedding_recorder.record_embedding = True
    for i, data in enumerate(data_loader):
        input, target, index = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)

        optimizer.zero_grad()
        output = model(input)
        # softmax = torch.nn.functional.softmax(output.requires_grad_(True), dim=1)

        loss_batch = criterion(output, target)
        loss = loss_batch.sum()
        # if i == 0:
        #    print(loss_batch)
        batch_num = target.shape[0]
        with torch.no_grad():
            bias_parameters_grads = torch.autograd.grad(loss, output)[0]  # [batch_size (128), n_class (10)]

            # temp = torch.cat([bias_parameters_grads, target.reshape((-1, 1)), is_noise[index].reshape((-1, 1))], dim=1)
            # print(temp[:10])

            # TODO: this might be wrong
            # maxInds = torch.argmax(prob, 1)
            # for j, max_idx in enumerate(maxInds):
            #    prob[j][max_idx] = prob[j][max_idx] - 1

            # p_norm = torch.norm(prob, dim=1, p=2)
            z_norm = torch.norm(model.embedding_recorder.embedding.view(batch_num, embedding_dim), dim=1, p=2)
            g_norm = torch.norm(bias_parameters_grads, dim=1, p=2)

            w_norm = torch.norm((model.embedding_recorder.embedding.view(batch_num, 1, embedding_dim).repeat(1,
                                                                                                             args.num_classes,
                                                                                                             1) * bias_parameters_grads.view(
                batch_num, args.num_classes,
                1).repeat(1, 1, embedding_dim)).view(batch_num, -1), dim=1, p=2)

            # g_norm = torch.norm(torch.cat([bias_parameters_grads, (model.embedding_recorder.embedding.view(batch_num,1,
            #            embedding_dim).repeat(1, args.num_classes, 1) * bias_parameters_grads.view(batch_num,
            #            args.num_classes,1).repeat(1, 1, embedding_dim)).view(batch_num,-1)], dim=1), dim=1, p=2)
            # bias_parameters_grads.view(batch_num,args.num_classes,1).repeat(1, 1, embedding_dim)

            # TODO: why p_norm*z_norm != g_norm, how to calculate g_norm?
            losses[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = loss_batch.detach()
            # p_norms[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = p_norm
            z_norms[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = z_norm
            # g_norms1[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = g_norm1
            g_norms[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = g_norm
            w_norms[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = w_norm

    # print(losses.shape, g_norms.shape, is_noise.shape)

    logs = torch.cat((losses, g_norms, z_norms, w_norms, is_noise), 1).cpu().numpy()
    print(logs[:30])

    # TODO: save
    file_name = 'logs/' + args.dataset + '/labelnoise20_epoch' + str(epoch) + '_loss_z_g_norms.txt'
    np.savetxt(file_name, logs, fmt="%.4f", delimiter=',')

    return 0


def save_p_z_g_cnn_logs(args, epoch, dst_train, model, criterion, optimizer, noise_idx):
    losses = torch.zeros([args.n_train, args.repeat], requires_grad=False).to(args.device)
    p_norms = torch.zeros([args.n_train, args.repeat], requires_grad=False).to(args.device)
    z_norms = torch.zeros([args.n_train, args.repeat], requires_grad=False).to(args.device)
    g_norms1 = torch.zeros([args.n_train, args.repeat], requires_grad=False).to(args.device)
    g_norms2 = torch.zeros([args.n_train, args.repeat], requires_grad=False).to(args.device)
    g_norms = torch.zeros([args.n_train, args.repeat], requires_grad=False).to(args.device)
    is_noise = torch.zeros([args.n_train, args.repeat], requires_grad=False).to(args.device)
    for i in range(args.n_train):
        if i in noise_idx:
            is_noise[i] = 1

    data_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.selection_batch, num_workers=args.workers)
    model.embedding_recorder.record_embedding = True
    for i, data in enumerate(data_loader):
        input, target, index = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)

        optimizer.zero_grad()
        output = model(input)
        prob = torch.nn.functional.softmax(output.requires_grad_(True), dim=1)
        loss_batch = criterion(prob, target)
        loss = loss_batch.sum()

        # temp = torch.cat([prob, target.reshape((-1,1)), is_noise[index].reshape((-1,1))], dim=1)
        # print(temp[:10])

        # Measure <p,z,g> and record logs
        batch_num = target.shape[0]
        with torch.no_grad():
            outcnn = model.embedding_recorder.embedding
            bias_parameters_grads = torch.autograd.grad(loss, outcnn)[0]  # [batch_size (128), emb_dim (512)]

            embedding_dim = model.get_last_layer().in_features

            # TODO: this might be wrong
            maxInds = torch.argmax(prob, 1)
            for j, max_idx in enumerate(maxInds):
                prob[j][max_idx] = prob[j][max_idx] - 1

            p_norm = torch.norm(prob, dim=1, p=2)
            z_norm = torch.norm(model.embedding_recorder.embedding.view(batch_num, embedding_dim), dim=1, p=2)
            g_norm = torch.norm(bias_parameters_grads, dim=1, p=2)

            # bias_parameters_grads.view(batch_num,args.num_classes,1).repeat(1, 1, embedding_dim)

            # TODO: why p_norm*z_norm != g_norm, how to calculate g_norm?
            losses[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = loss_batch.detach()
            p_norms[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = p_norm
            z_norms[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = z_norm
            g_norms[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = g_norm

    logs = torch.cat((losses, p_norms, z_norms, g_norms1, g_norms2, g_norms, is_noise), 1).cpu().numpy()

    # TODO: save
    file_name = 'logs/' + args.dataset + '/ood20_epoch' + str(epoch) + '_outcnn_p_z_g_norms.txt'
    np.savetxt(file_name, logs, fmt="%.4f", delimiter=',')

    return 0


def save_loss_gWc_logs(args, epoch, dst_train, model, criterion, optimizer, selected_idx, noise_idx):
    losses = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    dim_emb = model.get_last_layer().in_features  # 512
    gW_target = torch.zeros([args.n_train, dim_emb], requires_grad=False).to(args.device)

    preds = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    noisy_targets = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    is_selected = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    is_noise = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    for i in range(args.n_train):
        if i in selected_idx:
            is_selected[i] = 1
        if i in noise_idx:
            is_noise[i] = 1

    data_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.selection_batch, num_workers=args.workers)
    model.embedding_recorder.record_embedding = True
    for i, data in enumerate(data_loader):
        input, target, index = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)
        optimizer.zero_grad()

        output = model(input)
        pred = output.argmax(axis=1)  # .reshape((1,-1))

        loss_batch = criterion(output, target)  # output_dropped

        # Get gradient
        with torch.no_grad():
            prob = torch.nn.functional.softmax(output, dim=1)
            outcnn = model.embedding_recorder.embedding
            for j in range(len(target)):
                for c in range(args.num_classes):  # gWc
                    if c == target[j]:
                        gW_target[index[j]] = outcnn[j] * (1 - prob[j][c])

        losses[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = loss_batch.detach()
        preds[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = pred.reshape((1, -1))
        noisy_targets[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = target

    logs_t = torch.cat((losses, gW_target, preds, noisy_targets, is_selected, is_noise), 1).cpu().numpy()

    # TODO: save
    file_name = '/data/pdm102207/DataPruningLogs/' + str(args.dataset) + '/noise' + str(
        int(args.noise_rate * 100)) + '_' + str(args.noise_mode) \
                + '_epoch' + str(epoch) + '_gW_target_subset_weak_aug.txt'
    np.savetxt(file_name, logs_t, fmt="%.4f", delimiter=',')

    return 0


def save_loss_gW_logs(args, epoch, dst_train, model, criterion, optimizer, subset, noise_idx):
    losses = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    dim_emb = model.get_last_layer().in_features  # 512
    gW_target = torch.zeros([args.n_train, dim_emb * args.num_classes], requires_grad=False).to(args.device)
    gW_pred = torch.zeros([args.n_train, dim_emb * args.num_classes], requires_grad=False).to(args.device)

    preds = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    noisy_targets = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    is_selected = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    is_noise = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    for i in range(args.n_train):
        if i in subset:
            is_selected[i] = 1
        if i in noise_idx:
            is_noise[i] = 1

    data_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.selection_batch, num_workers=args.workers)
    model.embedding_recorder.record_embedding = True
    for i, data in enumerate(data_loader):
        input, target, index = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)
        optimizer.zero_grad()

        output = model(input)
        pred = output.argmax(axis=1)  # .reshape((1,-1))

        loss_batch = criterion(output, target)  # output_dropped

        # Get gradient
        with torch.no_grad():
            prob = torch.nn.functional.softmax(output, dim=1)
            outcnn = model.embedding_recorder.embedding
            for j in range(len(target)):
                for c in range(args.num_classes):  # gW
                    if c == target[j]:
                        gW_target[index[j]][dim_emb * c: dim_emb * (c + 1)] = outcnn[j] * (1 - prob[j][c])
                    else:
                        gW_target[index[j]][dim_emb * c: dim_emb * (c + 1)] = outcnn[j] * (-1 * prob[j][c])

                # for c in range(args.num_classes): # gW
                #    if c == pred[j]:
                #        gW_pred[index[j]][dim_emb * c: dim_emb * (c + 1)] = outcnn[j] * (1 - prob[j][c])
                #    else:
                #        gW_pred[index[j]][dim_emb * c: dim_emb * (c + 1)] = outcnn[j] * (-1 * prob[j][c])

        losses[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = loss_batch.detach()
        preds[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = pred.reshape((1, -1))
        noisy_targets[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = target

    logs_t = torch.cat((losses, gW_target, preds, noisy_targets, is_selected, is_noise), 1).cpu().numpy()
    logs_p = torch.cat((losses, gW_pred, preds, noisy_targets, is_selected, is_noise), 1).cpu().numpy()

    # TODO: save
    file_name = '/data/pdm102207/DataPruningLogs/' + str(args.dataset) + '/noise' + str(
        int(args.noise_rate * 100)) + '_' + str(args.noise_mode) \
                + '_epoch' + str(epoch) + '_gW_target.txt'
    np.savetxt(file_name, logs_t, fmt="%.4f", delimiter=',')
    # file_name = '/data/pdm102207/DataPruningLogs/' + str(args.dataset) + '/noise' + str(
    #    int(args.noise_rate * 100)) + '_' + str(args.noise_mode) + '_epoch' + str(epoch) + '_gW_pred.txt'
    # np.savetxt(file_name, logs_p, fmt="%.4f", delimiter=',')

    return 0


def save_loss_gW_logs_ood(args, epoch, dst_train, model, criterion, optimizer, noise_idx):
    losses = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    dim_emb = model.get_last_layer().in_features  # 512
    gW_target = torch.zeros([args.n_train, dim_emb * args.num_classes], requires_grad=False).to(args.device)
    gW_pred = torch.zeros([args.n_train, dim_emb * args.num_classes], requires_grad=False).to(args.device)

    preds = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    noisy_targets = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    is_noise = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    for i in range(args.n_train):
        if i in noise_idx:
            is_noise[i] = 1

    data_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.selection_batch, num_workers=args.workers)
    model.embedding_recorder.record_embedding = True
    for i, data in enumerate(data_loader):
        input, target, index = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)
        optimizer.zero_grad()

        output = model(input)
        pred = output.argmax(axis=1)  # .reshape((1, -1))

        loss_batch = criterion(output, target)  # output_dropped

        # Get gradient
        with torch.no_grad():
            prob = torch.nn.functional.softmax(output, dim=1)
            outcnn = model.embedding_recorder.embedding
            for j in range(len(target)):
                # gW to Y_target
                for c in range(args.num_classes):  # gW
                    if c == target[j]:
                        gW_target[index[j]][dim_emb * c: dim_emb * (c + 1)] = outcnn[j] * (1 - prob[j][c])
                    else:
                        gW_target[index[j]][dim_emb * c: dim_emb * (c + 1)] = outcnn[j] * (-1 * prob[j][c])
                # Y_pred
                for c in range(args.num_classes):  # gW
                    if c == pred[j]:
                        gW_pred[index[j]][dim_emb * c: dim_emb * (c + 1)] = outcnn[j] * (1 - prob[j][c])
                    else:
                        gW_pred[index[j]][dim_emb * c: dim_emb * (c + 1)] = outcnn[j] * (-1 * prob[j][c])

        losses[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = loss_batch.detach()
        preds[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = pred.reshape((1, -1))
        noisy_targets[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = target

    logs_t = torch.cat((losses, gW_target, preds, noisy_targets, is_noise), 1).cpu().numpy()
    logs_p = torch.cat((losses, gW_pred, preds, noisy_targets, is_noise), 1).cpu().numpy()

    # TODO: save
    file_name = '/data/pdm102207/DataPruningLogs/' + str(args.dataset) + '/ood' + str(
        int(args.noise_rate * 100)) + '_epoch' + str(epoch) + '_gW_target.txt'
    np.savetxt(file_name, logs_t, fmt="%.4f", delimiter=',')
    file_name = '/data/pdm102207/DataPruningLogs/' + str(args.dataset) + '/ood' + str(
        int(args.noise_rate * 100)) + '_epoch' + str(epoch) + '_gW_pred.txt'
    np.savetxt(file_name, logs_p, fmt="%.4f", delimiter=',')

    return 0


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, reduction='none', smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.reduction = reduction
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        if self.reduction == 'none':
            return torch.sum(-true_dist * pred, dim=self.dim)
        elif self.reduction == 'mean':
            return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def save_loss_smooth_gW_logs(args, epoch, dst_train, model, criterion, optimizer, noise_idx):
    losses = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    dim_emb = model.get_last_layer().in_features  # 512
    gW = torch.zeros([args.n_train, dim_emb], requires_grad=False).to(args.device)

    noisy_targets = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    is_noise = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    for i in range(args.n_train):
        if i in noise_idx:
            is_noise[i] = 1

    data_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.selection_batch, num_workers=args.workers)
    model.embedding_recorder.record_embedding = True
    criterion = LabelSmoothingLoss(classes=args.num_classes, reduction='none', smoothing=args.label_smoothing).to(
        args.device)
    for i, data in enumerate(data_loader):
        input, target, index = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)

        optimizer.zero_grad()
        output = model(input)
        loss_batch = criterion(output, target)
        loss = loss_batch.sum()

        # Measure <p,z,g> and record logs
        batch_num = target.shape[0]
        with torch.no_grad():
            prob = torch.nn.functional.softmax(output, dim=1)
            outcnn = model.embedding_recorder.embedding

            logit_grads = torch.autograd.grad(loss, outcnn)[0]  # [batch_size (128), emb_dim (512)]
            if i == 0:
                print("logit_grads.shape: ", logit_grads.shape)

            gW[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train)] = logit_grads

            losses[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = loss_batch.detach()
            noisy_targets[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = target

    logs = torch.cat((losses, gW, noisy_targets, is_noise), 1).cpu().numpy()
    print(logs.shape)
    # print(logs[0])

    # TODO: save
    file_name = '/data/pdm102207/DataPruningLogs/' + args.dataset + '/noisylabel40_epoch' + str(epoch) + '_loss_gW.txt'
    np.savetxt(file_name, logs, fmt="%.4f", delimiter=',')

    return 0


def save_loss_gW_norm_impact_logs(args, epoch, dst_train, model, criterion, optimizer, noise_idx):
    losses = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    dim_emb = model.get_last_layer().in_features  # 512
    gW = torch.zeros([args.n_train, dim_emb * args.num_classes], requires_grad=False).to(args.device)
    gW_norm = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    gW_impact = torch.zeros([args.n_train, 1], requires_grad=False).to(args.device)
    is_noise = torch.zeros([args.n_train, args.repeat], requires_grad=False).to(args.device)
    for i in range(args.n_train):
        if i in noise_idx:
            is_noise[i] = 1

    data_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.selection_batch, num_workers=args.workers)
    model.embedding_recorder.record_embedding = True
    for i, data in enumerate(data_loader):
        input, target, index = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)

        optimizer.zero_grad()
        output = model(input)
        loss_batch = criterion(output, target)

        # Measure <p,z,g> and record logs
        batch_num = target.shape[0]
        with torch.no_grad():
            prob = torch.nn.functional.softmax(output, dim=1)
            outcnn = model.embedding_recorder.embedding

            for j in range(len(target)):
                for c in range(args.num_classes):
                    if c == target[j]:
                        gW[index[j]][dim_emb * c: dim_emb * (c + 1)] = outcnn[j] * (1 - prob[j][c])
                    else:
                        gW[index[j]][dim_emb * c: dim_emb * (c + 1)] = outcnn[j] * (-1 * prob[j][c])

            losses[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train), 0] = loss_batch.detach()

    gW_norm = torch.norm(gW, axis=1)
    # gW_impact =

    logs = torch.cat((losses, gW, is_noise), 1).cpu().numpy()
    print(logs.shape)
    print(logs[0])

    # TODO: save
    file_name = 'logs/' + args.dataset + '/ood20_epoch' + str(epoch) + '_loss_gW.txt'
    np.savetxt(file_name, logs, fmt="%.4f", delimiter=',')

    return 0


def update_LogitGradNorm_history(args, epoch, dst_train, model, criterion, optimizer, noise_idx, norm_history):
    is_noise = torch.zeros([args.n_train, args.repeat], requires_grad=False).to(args.device)
    for i in range(args.n_train):
        if i in noise_idx:
            is_noise[i] = 1

    data_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.selection_batch, num_workers=args.workers)
    embedding_dim = model.get_last_layer().in_features
    model.embedding_recorder.record_embedding = True
    for i, data in enumerate(data_loader):
        input, target, index = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)

        optimizer.zero_grad()
        output = model(input)
        # prob = torch.nn.functional.softmax(output.requires_grad_(True), dim=1)
        loss_batch = criterion(output, target)
        loss = loss_batch.sum()

        print(loss_batch)

        # Measure <p,z,g> and record logs
        batch_num = target.shape[0]
        with torch.no_grad():
            logit_grads = torch.autograd.grad(loss, output)[0]  # [batch_size (128), n_class (10)]

            g_norm = torch.norm(logit_grads, dim=1, p=2)

            import numpy as np
            idx = torch.argmax(g_norm)
            # print(logit_grads[idx], logit_grads[idx].sum(), g_norm[idx])
            break

            # TODO: why p_norm*z_norm != g_norm, how to calculate g_norm?
            norm_history[i * args.selection_batch:min((i + 1) * args.selection_batch, args.n_train),
            epoch] = g_norm.detach()
    return norm_history

