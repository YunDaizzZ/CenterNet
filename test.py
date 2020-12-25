# coding:utf-8
from __future__ import division
from torchsummary import summary
from nets.centernet import CenterNet_Resnet50

if __name__ == "__main__":
    model = CenterNet_Resnet50().train().cuda()
    summary(model, (3, 512, 512))