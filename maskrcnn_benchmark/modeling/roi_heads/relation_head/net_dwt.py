import os
import time
import sys
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import functools
import importlib
from maskrcnn_benchmark.modeling.roi_heads.relation_head.dwt.attentions.wad_module import wad_module

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)



class BottleNect(nn.Module):
    EXPANSION = 4

    def __init__(self, in_channels, out_channels, stride=1, attention_module=None):
        super(BottleNect, self).__init__()

        self.flag = False


        if stride != 1 :
            self.flag = True
            self.wa = attention_module()
            self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        else:
            self.conv2 = conv3x3(out_channels, out_channels, stride=stride)


        self.conv1 = conv1x1(in_channels, out_channels, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = conv1x1(out_channels, out_channels * self.EXPANSION, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.EXPANSION)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * self.EXPANSION:
            self.shortcut = nn.Sequential(conv1x1(in_channels, out_channels * self.EXPANSION, stride=1),
                                                  nn.BatchNorm2d(out_channels * self.EXPANSION))


    def forward(self, x):
        identity = x
        if self.flag:
            out = self.wa(x)
            identity=out
            out = self.conv1(out)
        else:
            out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(identity)

        return self.relu(out)




class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_class=10):
        super(ResNet, self).__init__()

        self.num_class = num_class
        self.in_channels = num_filters = 8

        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, self.in_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(num_filters*2), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(num_filters*4), num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(
            int(num_filters*4*block(16, 16, 1).EXPANSION), num_class)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))    #16,16--5*64�?6、、、�?4�?2--5*128�?2、、、、�?28�?4--5*256�?4
            self.in_channels = int(out_channels * block(16, 16, 1).EXPANSION)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out

def ResNetWrapper(num_blocks, num_class=10, block=None, attention_module=None):

    def b(in_planes, planes, stride):
        return block(in_planes, planes, stride, attention_module=attention_module)

    return ResNet(b, num_blocks, num_class=num_class)

def ResNet50(num_class=51, block=BottleNect, attention_module=wad_module):

    n_blocks = [3, 3, 3]
    return ResNetWrapper(
        num_blocks=n_blocks,
        num_class=num_class,
        block=block,
        attention_module=attention_module)


if __name__=="__main__":
    net = ResNet50(
        num_class=51,
        block=BottleNect,
        attention_module=wad_module
    )
    # print(net)
    data=torch.rand(5,3,10,10)
    print(net(data))