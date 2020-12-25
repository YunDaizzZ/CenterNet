# coding:utf-8
from __future__ import division
import torch.nn as nn
from nets.ResNet50 import resnet50, resnet50_Decoder, resnet50_Head

class CenterNet_Resnet50(nn.Module):
    def __init__(self, num_classes=20, pretrain=False):
        super(CenterNet_Resnet50, self).__init__()
        self.backbone = resnet50(pretrain=pretrain)
        self.decoder = resnet50_Decoder(2048)
        self.head = resnet50_Head(channel=64, num_classes=num_classes)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        feat = self.backbone(x)

        return self.head(self.decoder(feat))