#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torchsummary import summary


# ## ConvBlock
# 
# > 3 x3 Conv → BatchNorm(BN) → LeakyReLU
# 
# - a convolution block has above structure refering the [paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
# 

# In[ ]:


class ConvBlock(nn.Module):
    def __init__(self, in_plane, out_plane, droprate):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_plane)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_plane, out_plane, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = droprate
       

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=self.droprate, training=self.training, inplace=False)
        
        return out


# ## ResidualBlock
# > 1 x 1 ConvBlock → 3 x 3 ConvBlock → concatenation earlier feature map & upsampled feature
# - get more meaningful semantic information from upsampled feature
# - finer-grained information from earlier feature map

# In[ ]:


class ResidualBlock(nn.Module):
    def __init__(self, in_plane, droprate):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_plane, in_plane/2, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_plane/2)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_plane/2, in_plane, kernel_size = 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_plane)
        self.droprate = droprate
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out += residual
        out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)        

        return out


# ## DarkNet
# - make DarkNet structure using ConvBlock and Resudial Block
# - refering the paper
#     - no. of residual block: 1 2 8 8 4
# 
# > conv(32, 64) → conv(64, 128) → residual(128, 128) → conv(128, 256) → residual(256, 256) → conv(256, 512) → residual(512, 512) → conv(512, 1024) → residual(1024,1024) → avg_pooling → linear(1024, num_class)

# In[ ]:


class Darknet(nn.Module):
    def __init__(self, block = ResidualBlock, num_classes, droprate = 0):
        super(Darknet, self).__init__()
        self.num_classes = num_classes
        self.droprate = droprate

        self.conv1 = ConvBlock(3, 32, self.droprate)
        self.conv2 = ConvBlock(32, 64, self.droprate)
        self.resi1 = self.make_residual_block(block, 64, 1, self.droprate)

        self.conv3 = ConvBlock(64, 128, self.droprate)
        self.resi2 = self.make_residual_block(block, 128, 2, self.droprate)

        self.conv4 = ConvBlock(128, 256, self.droprate)
        self.resi3 = self.make_residual_block(block, 256, 8, self.droprate)

        self.conv5 = ConvBlock(256, 512, self.droprate)
        self.resi4 = self.make_residual_block(block, 512, 8, self.droprate)

        self.conv6 = ConvBlock(512, 1024, self.droprate)
        self.resi5 = self.make_residual_block(block, 1024, 4, self.droprate)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)

    def make_residual_block(self, block, in_plane, nblock, droprate):
        layers = []
        for i in range(nblock):
            layers.append(block(in_plane, droprate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.resi1(out)

        out = self.conv3(out)
        out = self.resi2(out)

        out = self.conv4(out)
        out = self.resi3(out)

        out = self.conv5(out)
        out = self.resi4(out)

        out = self.conv6(out)
        out = self.resi5(out)

        out = self.avg_pool(out)
        out = self.fc(out)

        return out


# In[ ]:




