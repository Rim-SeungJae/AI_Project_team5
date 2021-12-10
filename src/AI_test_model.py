# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 04:31:31 2021

@author: dipreez
"""

import os
import numpy as np
import json
import torch
# from google.colab import drive
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from collections import Counter
import math
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
import AI_Project_ResNet as resnet
import AI_Project_DarkNet as darknet
import AI_Project_vgg16 as vgg
import AI_Project_DenseNet as densenet
from utils.AI_preprocessing import MarinTrashDS


model = "DarkNet"
assert model in ["ResNet", "DarkNet", "VGG", "DenseNet"]

with open('../data/imgclass_map.json') as f:
    explain = json.load(f)
with open('../data/imgname_map.json') as f:
    full_names = json.load(f)
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = 0.5, std = 0.5)                                
])
    
test_dataset = MarinTrashDS(explain = explain, full_names = full_names, debris_file = '../data/debris.json', transform = transform)

test_loader = DataLoader(test_dataset, batch_size = 8, num_workers=0)

num_classes = test_dataset.num_classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if model == "ResNet":
    loaded = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes = num_classes).to(device)
    threshold = 0.4
elif model == "DarkNet":
    loaded = darknet.Darknet(num_classes = num_classes).to(device)
elif model == "VGG":
    loaded = vgg.VGGNet(num_classes = num_classes).to(device)
    threshold = 0.3
else:
    loaded = densenet.DenseNet(0.2, num_classes=num_classes).to(device)

# Load best model
loaded.load_state_dict(torch.load('../model/' + model + '.pt'))

# Test
loaded.eval()

test_loss = 0
test_correct = 0

with torch.no_grad():
    for i, (x, y) in tqdm(enumerate(test_loader)):
        x = x.to(device)
        y = y.to(device)
        outputs = loaded(x)
        
        predicted = torch.sigmoid(outputs)
        predicted = predicted > 0.4
        predicted = torch.tensor(predicted,dtype = float, device = device)
        test_correct += predicted.eq(y).all(dim=1).sum().item()
        
        if i == 0:
            test_preds = predicted
        else:
            test_preds = torch.cat((test_preds, predicted), dim=0)
            
test_preds = test_preds.cpu()
testacc = test_correct/len(test_dataset)

print('TestAcc: %.2f' % (testacc))