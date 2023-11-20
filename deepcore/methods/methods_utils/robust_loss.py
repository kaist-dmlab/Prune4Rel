import torch
import torch.nn.functional as F
import numpy as np

class ELRLoss(torch.nn.Module):
    def __init__(self, args): #, num_examp, num_classes=10, beta=0.3
        super(ELRLoss, self).__init__()
        self.num_classes = args.n_class
        self.target = torch.zeros(args.n_train, self.num_classes).cuda(args.device)
        self.beta = args.beta
        self.lamda = args.lambda_elr

    def forward(self, index, output, label):
        y_pred = torch.nn.functional.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1 - self.beta) * \
                             ((y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        ce_loss = torch.nn.functional.cross_entropy(output, label)
        elr_reg = ((1 - (self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + self.lamda * elr_reg
        return final_loss

class ELRPLUSLoss(torch.nn.Module):
    def __init__(self, args): #num_examp, config, device, num_classes=10, beta=0.3
        super(ELRPLUSLoss, self).__init__()
        self.num_classes = args.n_class
        self.pred_hist = torch.zeros(args.n_train, self.num_classes).cuda(args.device)
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
        final_loss = ce_loss + self.sigmoid_rampup(iteration, 0)*(self.lamda * reg)

        return final_loss, y_pred.cpu().detach()

    def update_hist(self, epoch, out, index=None, mix_index=..., mixup_l=1):
        y_pred_ = torch.nn.functional.softmax(out, dim=1)
        self.pred_hist[index] = self.beta * self.pred_hist[index] + (1 - self.beta) * y_pred_ / (y_pred_).sum(dim=1,
                                                                                                              keepdim=True)
        self.q = mixup_l * self.pred_hist[index] + (1 - mixup_l) * self.pred_hist[index][mix_index]


class OverparametrizationLoss(torch.nn.Module):
    def __init__(self, args): #, num_examp, num_classes=10, ratio_consistency=0, ratio_balance=0):
        super(OverparametrizationLoss, self).__init__()
        self.num_classes = args.n_class
        self.num_examp = args.n_train

        self.ratio_consistency = args.ratio_consistency
        self.ratio_balance = args.ratio_balance
    '''
        self.u = torch.nn.Parameter(torch.empty(self.num_examp, 1, dtype=torch.float32)).to(args.device)
        self.v = torch.nn.Parameter(torch.empty(self.num_examp, self.num_classes, dtype=torch.float32)).to(args.device)

        self.init_param(mean=0, std=1e-8)

    def init_param(self, mean=0., std=1e-8):
        torch.nn.init.normal_(self.u, mean=mean, std=std)
        torch.nn.init.normal_(self.v, mean=mean, std=std)
    '''

    def forward(self, u, v, index, outputs, label):
        if len(outputs) > len(index):
            output, output2 = torch.chunk(outputs, 2)
        else:
            output = outputs

        eps = 1e-4

        U_square = u[index] ** 2 * label
        V_square = v[index] ** 2 * (1 - label)

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
            return (torch.zeros(len(x), self.num_classes)).cuda().scatter_(1, (x.argmax(dim=1)).view(-1, 1), 1)


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