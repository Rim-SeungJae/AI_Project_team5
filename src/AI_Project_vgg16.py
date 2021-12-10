# -*- coding: utf-8 -*-
"""AI_Project_VGG16.py
# Copyright 2021. Dongwon Kim All rights reserved.
# File name : AI_Project_VGG16.py
# Written by Dongwon Kim
#
# VGG16 for MarinDebris dataset
#    13 classes
#    images will be resized to 64 x 64
#
# Modificatoin history
#    written by Dongwon Kim on Dec 11, 2021
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_plane, out_plane, kernel_size = 3, stride = 1, padding = 1, bias=False)
        self.bn = nn.BatchNorm2d(out_plane)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out

class VGGNet(nn.Module):
    def __init__(self, num_classes):
        super(VGGNet, self).__init__()
        self.num_classes = num_classes

        # 64 x 64 -> 32 x 32
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 64)
        self.max_pool = nn.MaxPool2d(2, stride = 2)

        # 32 x 32 -> 16 x 16
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)

        # 16 x 16 -> 8 x 8
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)

        # 8 x 8 -> 4 x 4, 4 x 4 -> 2 x 2
        self.conv7 = ConvBlock(256, 512)
        self.conv8 = ConvBlock(512, 512)

        self.fc1 = nn.Linear(512 * 2 * 2, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(4096, self.num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.max_pool(out)

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.max_pool(out)

        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv6(out)
        out = self.max_pool(out)

        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv8(out)
        out = self.max_pool(out)

        out = self.conv8(out)
        out = self.conv8(out)
        out = self.conv8(out)
        out = self.max_pool(out)

        out = out.view(-1, 512 * 2 * 2)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out
