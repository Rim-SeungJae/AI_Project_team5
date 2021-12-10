#!/usr/bin/env python
# coding: utf-8

# # Preprocessing

# ## resize

# In[1]:


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
from utils.AI_preprocessing import Debris, MarinTrashDS


# In[2]:


# drive.mount('/content/drive')
FILE_PATH =  "C:\\Users\\dipreez\\Downloads\\해양 침적 쓰레기 이미지\\Training"


# In[3]:


# os.chdir(FILE_PATH)


# In[7]:

with open('../data/imgclass_map.json') as f:
    explain = json.load(f)
with open('../data/imgname_map.json') as f:
    full_names = json.load(f)


# In[11]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = 0.5, std = 0.5)                                
])


# In[12]:


train_dataset = MarinTrashDS(explain = explain, full_names = full_names, debris_file = '../data/debris.json', transform = transform)


# In[13]:


num_classes = train_dataset.num_classes


# In[14]:


tr_index, val_index = train_test_split(list(range(len(train_dataset))), test_size = 0.2, shuffle = True)
tr_sampler = SubsetRandomSampler(tr_index)
val_sampler = SubsetRandomSampler(val_index)


# In[15]:


train_loader = DataLoader(train_dataset, batch_size = 16, num_workers=0, sampler = tr_sampler)
val_loader = DataLoader(train_dataset, batch_size = 16, num_workers = 0, sampler = val_sampler)


# In[16]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[18]:


# net = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes = num_classes).to(device)
net = darknet.Darknet(num_classes = num_classes).to(device)

criterion = BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# In[19]:


train_total = len(tr_index)
valid_total = len(val_index)

train_batches = len(train_loader)
valid_batches = len(val_loader)

# Variables for lr scheduling and early stopping
best_valid_loss = 1024    # Any large number will suffice
patience = 0    # Bad epoch counter


# In[21]:


# %%time
epochs =100
for epoch in tqdm(range(1,epochs+1)):
    # Train
    net.train()
    
    train_loss = 0
    train_correct = 0
    
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        outputs = net(x)
        loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        # _, max_indices = outputs.max(1, keepdim = True)
        predicted = torch.sigmoid(outputs)
        predicted = outputs > 0.5
        predicted = torch.tensor(predicted,dtype = float, device = device)
        train_correct += predicted.eq(y).all(dim=1).sum().item()
        
    train_loss = train_loss / train_batches
    train_acc = train_correct / train_total
    
    # Validate
    net.eval()
    
    valid_loss = 0
    valid_correct = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            
            valid_loss += loss.item()
            # _, max_indices = outputs.max(1)
            predicted = torch.sigmoid(outputs)
            predicted = outputs > 0.5
            predicted = torch.tensor(predicted,dtype = float, device = device)
            valid_correct += predicted.eq(y).all(dim=1).sum().item()
            
    valid_loss = valid_loss / valid_batches
    valid_acc = valid_correct / valid_total
    
    # Save best model
    if best_valid_loss > valid_loss:
        torch.save(net.state_dict(), '../resnet.pt')
        best_valid_loss = valid_loss
        patience = 0

    if epoch%(epochs/100) == 0:
      print('[%d/%d] TrainLoss: %.3f, ValLoss: %.3f | TrainAcc: %.2f, ValAcc: %.2f' % (epoch, epochs, train_loss, valid_loss, train_acc, valid_acc))
    
    scheduler.step()
    
    # Break training loop if no improvement for 10 consecutive epochs
    if patience == 10:
        print('No improvement found. Escaped for loop at [%d/%d]' % (epoch,epochs))
        break
        
    patience += 1


# In[ ]:





# In[ ]:




