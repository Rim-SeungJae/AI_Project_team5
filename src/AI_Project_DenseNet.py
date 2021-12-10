#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch
# from google.colab import files
from torchsummary import summary
import math


# ## Bottleneck
# - use bottleneck block instead of normal dense block
# - to reduce feature map insert 1x1 conv layer before 3x3 conv layer
# - also increase computational efficiency
# - reference
#     - [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993v5.pdf) paper
# 
# > BatchNorm(BN) → relu → 1x1 Conv → BN → relu → 3x3 Conv
# 
# - for 1x1 Conv refering the paper, produce 4 * growth rate 
# - add Dropout layer to prevent overfitting
# 

# In[ ]:


class BottleneckBlock(nn.Module):
    def __init__(self, in_plane, growth_rate, droprate):
        super(BottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_plane)
        self.conv1 = nn.Conv2d(in_plane, 4*growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size = 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.droprate = droprate
    
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)        

        return torch.cat([out, x], 1)


# ## TransitionBlock
# - in DenseNet, use transition block for **compression**
# - therefore, use 2 x 2 average pooling
# 
# > BN → relu → 1 x 1 Conv → 2 x 2 Avg Pooling
# 
# - add Dropout layer 

# In[ ]:


class TransitionBlock(nn.Module):
    def __init__(self, in_plane, out_plane, droprate):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_plane)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_plane, out_plane, kernel_size=1, stride=1, bias=False)
        self.droprate = droprate
       

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.dropout(out, p=self.droprate, training=self.training, inplace=False)
        
        return F.avg_pool2d(out, 2)


# ## DenseNet
# - make DenseNet using Bottleneck and transition block
# - refering the paper, 
#     - set θ as 0.5, growth rate as 12
#     - set 1st 3 x 3 Conv layer's output channels as twice the growth rate
#     - for Dense121, 
#         - no. of Dense block: 6 12 24 16 
# 
# > conv → dense1 → transition1 → dense2 → transition2 → dense3 → transition3 → dense4 → classification layer

# In[ ]:


class DenseNet(nn.Module):
    def __init__(self, droprate, block=BottleneckBlock, growth_rate=12, num_classes=10, reduction=0.5):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        

        in_plane = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, in_plane, kernel_size = 3, padding=1, bias=False)

        # 1st Dense & Transition
        self.dense1 = self.make_dense_block(block, in_plane, 6, droprate)
        in_plane += 6 * growth_rate
        out_plane = int(math.floor(in_plane * reduction))
        self.trans1 = TransitionBlock(in_plane, out_plane, droprate)
        in_plane = out_plane

        # 2nd Dense & Transition
        self.dense2 = self.make_dense_block(block, in_plane, 12,droprate)
        in_plane += 12 * growth_rate
        out_plane = int(math.floor(in_plane * reduction))
        self.trans2 = TransitionBlock(in_plane, out_plane, droprate)
        in_plane = out_plane

        # 3rd Dense & Transition
        self.dense3 = self.make_dense_block(block, in_plane, 24, droprate)
        in_plane += 24 * growth_rate
        out_plane = int(math.floor(in_plane * reduction))
        self.trans3 = TransitionBlock(in_plane, out_plane, droprate)
        in_plane = out_plane

        # 4th Dense
        self.dense4 = self.make_dense_block(block, in_plane, 16, droprate)
        in_plane += 16 * growth_rate
        
        self.bn = nn.BatchNorm2d(in_plane)
        self.fc = nn.Linear(in_plane, num_classes)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. /n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
       

    def make_dense_block(self, block, in_plane, nblock, droprate):
        layers=[]
        for i in range(nblock):
            layers.append(block(in_plane, self.growth_rate, droprate))
            in_plane += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        # 32 x 32
        out = self.conv1(x)

        # 32 x 32
        out = self.dense1(out)
        out = self.trans1(out) # 32 -> 16
        
        # 16 x 16
        out = self.dense2(out)
        out = self.trans2(out) # 16 -> 8
        
        # 8 x 8
        out = self.dense3(out)
        out = self.trans3(out) # 8 -> 4

        out = self.dense4(out)

        out = self.bn(out)
        out = self.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


