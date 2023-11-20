import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .nets_utils import EmbeddingRecorder

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        images = images.cuda()
        #labels = labels.cuda()
        out,_ = self(images)                  # Generate predictions        
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def get_embedding(self, batch):
        images, labels = batch 
        images = images.cuda()
        #labels = labels.cuda()
        _,embbedding = self(images)                    # Generate predictions                
        return embbedding
    
    def validation_step(self, batch):
        images, labels = batch 
        images = images.cuda()
        #labels = labels.cuda()
        out,_ = self(images)                    # Generate predictions        
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


def conv_block(in_channels, out_channels, pool=False, pool_no=2):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              Mish()
              ]
    if pool: layers.append(nn.MaxPool2d(pool_no))
    return nn.Sequential(*layers)

class ResNet5(ImageClassificationBase):
    def __init__(self, in_channels, num_classes, im_size, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = False):
        super().__init__()
        
        if im_size[0] == 4 and im_size[1] == 4:
            self.conv1 = conv_block(in_channels, 64)
            self.conv2 = conv_block(64, 128, pool=False) # 128, 4, 4
            self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
            
            self.fc = nn.Linear(128,num_classes)

            self.embedding_recorder = EmbeddingRecorder(record_embedding)
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        out_emb = self.embedding_recorder(out)

        out = self.fc(out_emb)
        return out #, out_emb

    def get_last_layer(self):
        return self.fc