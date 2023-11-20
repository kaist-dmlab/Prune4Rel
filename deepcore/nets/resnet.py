import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled, flatten, Tensor
from .nets_utils import EmbeddingRecorder
from torchvision.models import resnet


# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_32x32(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, record_embedding: bool = False,
                 no_grad: bool = False, penultimate: bool = False):
        super().__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(channel, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes, bias=False)

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad
        self.penultimate = penultimate
        self.attn_drop = False

    # by DM
    ###
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                # NOTE:
                # print(type(param_t)) #This is Parameter
                # print(type(grad)) # But, this is Tensor!
                tmp = param_t - lr_inner * grad
                # print(tmp)
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    # if first_order:
                    #    grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):  # name = curr_mod_layer
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            # NOTE:
            # setattr(curr_mod, name, param) # Need to convert all the Parameter into Tensor
            curr_mod._parameters[name] = param  # Parameter -> Tensor

    def get_last_layer(self):
        return self.linear

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def activate_attn_drop(self):
        self.attn_drop = True

    def deactivate_attn_drop(self):
        self.attn_drop = False

    def get_drop_pixels(self, x):
        print("ATTN_DROP!")
        with set_grad_enabled(not self.no_grad):
            # Virtual Forward-pass, To drop 5% highest attn_scores
            temp = F.relu(self.bn1(self.conv1(x)))
            temp1 = self.layer1(temp)  # torch.Size([128, 64, 32, 32])
            temp = self.layer2(temp1)
            temp = self.layer3(temp)
            temp4 = self.layer4(temp)  # [128, 512, 4, 4])

            # TODO: SHOULD BE CHANGED TO GradCAM,    attention score
            attn_scores1 = temp1.sum(dim=1)
            Up = nn.Upsample(scale_factor=8)
            attn_scores4 = Up(temp4).sum(dim=1)

            attn_scores = (attn_scores1 * attn_scores4).reshape((x.shape[0], -1))
            topk_scores, topk_idxs = torch.topk(attn_scores, k=int(0.1 * (temp1.shape[-1] ** 2)), dim=1)  # descending

            drop_idxs = []
            # DropMasking
            for b, idxs in enumerate(topk_idxs):
                drop_idxs.append([idxs // temp1.shape[2], idxs % temp1.shape[3]])

                # print(idxs, idxs//out1.shape[2], idxs%out1.shape[3])
                temp1[b, :, idxs // temp1.shape[2], idxs % temp1.shape[3]] = 0

            return drop_idxs

    # TODO: Get drop pixels as above, and check the CAM on Jupyter
    def get_drop_pixels_cam(self, x, y):
        # print("Get drop pixels by CAM")
        with set_grad_enabled(not self.no_grad):
            # Virtual Forward-pass, To drop 5% highest attn_scores
            out = F.relu(self.bn1(self.conv1(x)))
            out1 = self.layer1(out)  # torch.Size([128, 64, 32, 32])
            out = self.layer2(out1)
            out = self.layer3(out)
            out4 = self.layer4(out)  # [128, 512, 4, 4])

            # TODO: channel-wise summation
            W = self.linear.weight  # [10, 512]
            W_batch = W[y].unsqueeze(2).unsqueeze(3).repeat(1, 1, out4.shape[2], out4.shape[3])

            # for input drop
            Up = nn.Upsample(scale_factor=8)
            # attn_scores = Up(out4*W_batch).sum(axis=1)
            attn_scores = Up(out4).sum(axis=1)

            attn_scores = attn_scores.reshape((out1.shape[0], -1))
            topk_scores, topk_idxs = torch.topk(attn_scores, k=int(0.05 * (out1.shape[-1] ** 2)), dim=1)  # descending

            drop_idxs = []
            # DropMasking
            for b, idxs in enumerate(topk_idxs):
                drop_idxs.append([idxs // out1.shape[2], idxs % out1.shape[3]])
                # print(idxs, idxs//out1.shape[2], idxs%out1.shape[3])
                # out1[b, :, idxs // out1.shape[2], idxs % out1.shape[3]] = 0

            return drop_idxs

    def get_drop_maps_weighted(self, x, y):
        # print("Get drop pixels by CAM")
        with set_grad_enabled(not self.no_grad):
            # Virtual Forward-pass, To drop 5% highest attn_scores
            out = F.relu(self.bn1(self.conv1(x)))
            out1 = self.layer1(out)  # torch.Size([128, 64, 32, 32])
            out = self.layer2(out1)
            out = self.layer3(out)
            out4 = self.layer4(out)  # [128, 512, 4, 4])

            # TODO: channel-wise summation
            W = self.linear.weight  # [10, 512]
            W_batch = W[y].unsqueeze(2).unsqueeze(3).repeat(1, 1, out4.shape[2], out4.shape[3])

            # for out_map drop
            attn_scores = (out4 * W_batch).sum(axis=1)

            attn_scores = attn_scores.reshape((out4.shape[0], -1))
            topk_scores, topk_idxs = torch.topk(attn_scores, k=int(0.1 * (out4.shape[-1] ** 2)), dim=1)  # descending

            drop_idxs = []
            # DropMasking
            for b, idxs in enumerate(topk_idxs):
                drop_idxs.append([idxs // out4.shape[2], idxs % out4.shape[3]])
                # print(idxs, idxs//out1.shape[2], idxs%out1.shape[3])
                # out4[b, :, idxs // out4.shape[2], idxs % out4.shape[3]] = 0

            return drop_idxs

    def get_drop_output(self, x, y):
        drop_idxs = self.get_drop_maps_weighted(x, y)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out4 = self.layer4(out)

        # DropMasking
        for b, idxs in enumerate(drop_idxs):
            out4[b, :, idxs[0], idxs[1]] = 0  # -1

        out = F.avg_pool2d(out4, 4)
        out_cnn = out.view(out.size(0), -1)
        out = self.embedding_recorder(out_cnn)
        out = self.linear(out_cnn)

        return out

    def forward(self, x):
        if self.attn_drop == False:
            with set_grad_enabled(not self.no_grad):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = F.avg_pool2d(out, 4)
                out_cnn = out.view(out.size(0), -1)
                out = self.embedding_recorder(out_cnn)
                out = self.linear(out_cnn)
            if self.penultimate == False:
                return out
            else:
                return out, out_cnn
        else:
            # print("ATTN_DROP!")
            with set_grad_enabled(not self.no_grad):
                # Virtual Forward-pass, To drop 5% highest attn_scores
                temp = F.relu(self.bn1(self.conv1(x)))
                temp1 = self.layer1(temp)  # torch.Size([128, 64, 32, 32])
                temp = self.layer2(temp1)
                temp = self.layer3(temp)
                temp4 = self.layer4(temp)  # [128, 512, 4, 4])

                # TODO: SHOULD BE CHANGED TO GradCAM or Class-weighted,    attention score
                attn_scores1 = temp1  # .sum(dim=1)
                Up = nn.Upsample(scale_factor=8)
                attn_scores4 = Up(temp4)  # .sum(dim=1)

                attn_scores = (attn_scores1 * attn_scores4).reshape((x.shape[0], -1))
                topk_scores, topk_idxs = torch.topk(attn_scores, k=int(1 * (temp1.shape[-1] ** 2)), dim=1)  # descending

                # DropMasking
                for b, idxs in enumerate(topk_idxs):
                    # print(idxs, idxs//out1.shape[2], idxs%out1.shape[3])
                    temp1[b, :, idxs // temp1.shape[2], idxs % temp1.shape[3]] = 0  # -1
                # print(temp1[0].le(-1).sum()) #64(c)*51(5% drop)

                # Real Forward-pass
                out1 = temp1.detach()
                # print(out1[0][0])

                out = self.layer2(out1)
                out = self.layer3(out)
                out4 = self.layer4(out)  # [128, 512, 4, 4])

                out = F.avg_pool2d(out4, 4)
                out_cnn = out.view(out.size(0), -1)
                out = self.embedding_recorder(out_cnn)
                out = self.linear(out_cnn)
            return out


class ResNet_224x224(resnet.ResNet):
    def __init__(self, block, layers, channel: int, num_classes: int, record_embedding: bool = False,
                 no_grad: bool = False, **kwargs):
        super().__init__(block, layers, **kwargs)
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        if channel != 3:
            self.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if num_classes != 1000:
            self.fc = nn.Linear(self.fc.in_features, num_classes)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.fc

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        with set_grad_enabled(not self.no_grad):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = flatten(x, 1)
            x = self.embedding_recorder(x)
            x = self.fc(x)

        return x


def ResNet(arch: str, channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
           pretrained: bool = False, penultimate: bool = False):
    arch = arch.lower()
    if pretrained:
        if arch == "resnet18":
            net = ResNet_224x224(resnet.BasicBlock, [2, 2, 2, 2], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet34":
            net = ResNet_224x224(resnet.BasicBlock, [3, 4, 6, 3], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet50":
            net = ResNet_224x224(resnet.Bottleneck, [3, 4, 6, 3], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet101":
            net = ResNet_224x224(resnet.Bottleneck, [3, 4, 23, 3], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet152":
            net = ResNet_224x224(resnet.Bottleneck, [3, 8, 36, 3], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        else:
            raise ValueError("Model architecture not found.")
        from torch.hub import load_state_dict_from_url
        # state_dict = load_state_dict_from_url(resnet.model_urls[arch], progress=True)
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
                                              progress=True)
        net.load_state_dict(state_dict)
        print("pre-trained!")

        if channel != 3:
            net.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if num_classes != 1000:
            net.fc = nn.Linear(net.fc.in_features, num_classes)

    elif im_size[0] == 224 and im_size[1] == 224:
        if arch == "resnet18":
            net = ResNet_224x224(resnet.BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet34":
            net = ResNet_224x224(resnet.BasicBlock, [3, 4, 6, 3], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet50":
            net = ResNet_224x224(resnet.Bottleneck, [3, 4, 6, 3], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet101":
            net = ResNet_224x224(resnet.Bottleneck, [3, 4, 23, 3], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet152":
            net = ResNet_224x224(resnet.Bottleneck, [3, 8, 36, 3], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        else:
            raise ValueError("Model architecture not found.")
    elif (channel == 1 and im_size[0] == 28 and im_size[1] == 28) or (
            channel == 3 and im_size[0] == 32 and im_size[1] == 32):
        if arch == "resnet18":
            net = ResNet_32x32(BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad, penultimate=penultimate)
        elif arch == "resnet34":
            net = ResNet_32x32(BasicBlock, [3, 4, 6, 3], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad, penultimate=penultimate)
        elif arch == "resnet50":
            net = ResNet_32x32(Bottleneck, [3, 4, 6, 3], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad, penultimate=penultimate)
        elif arch == "resnet101":
            net = ResNet_32x32(Bottleneck, [3, 4, 23, 3], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad, penultimate=penultimate)
        elif arch == "resnet152":
            net = ResNet_32x32(Bottleneck, [3, 8, 36, 3], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad, penultimate=penultimate)
        else:
            raise ValueError("Model architecture not found.")
    else:
        raise NotImplementedError("Network Architecture for current dataset has not been implemented.")
    return net


def ResNet18(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
             pretrained: bool = False, penultimate: bool = False):
    return ResNet("resnet18", channel, num_classes, im_size, record_embedding, no_grad, pretrained, penultimate)


def ResNet34(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
             pretrained: bool = False):
    return ResNet("resnet34", channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def ResNet50(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
             pretrained: bool = False):
    return ResNet("resnet50", channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def ResNet101(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
              pretrained: bool = False):
    return ResNet("resnet101", channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def ResNet152(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
              pretrained: bool = False):
    return ResNet("resnet152", channel, num_classes, im_size, record_embedding, no_grad, pretrained)



'''
class ResNet18Extractor(ResNet_32x32):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, record_embedding: bool = False,
                 no_grad: bool = False, penultimate: bool = False):
        super().__init__(block, num_blocks, channel=3, num_classes=10, record_embedding = False,
                 no_grad = False, penultimate = False)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

def ResNet18Extractor(num_classes:int):
    return ResNet18Extractor(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
'''


class ResNetExtractor(ResNet_224x224):
    def __init__(self, block, num_blocks, num_classes=10, record_embedding: bool = False,
                 no_grad: bool = False, penultimate: bool = False):
        super().__init__(block, num_blocks, num_classes=num_classes, record_embedding = False,
                 no_grad = False, penultimate = False)

    def forward(self, x, lin=0, lout=5):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.avg_pool2d(out, 4)
            out_final = out.view(out.size(0), -1)

        return out_final

def ResNet50Extractor(num_classes:int):
    return ResNetExtractor(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)