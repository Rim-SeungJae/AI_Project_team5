#!/usr/bin/env python
# coding: utf-8

# # Preprocessing

# ## resize

# In[ ]:


import os
import numpy as np
import json
import torch
from google.colab import drive
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


# In[ ]:


drive.mount('/content/drive')
FILE_PATH = "/content/drive/MyDrive/Colab Notebooks/해양 침적 쓰레기 이미지/Training"


# In[ ]:


os.chdir(FILE_PATH)


# In[ ]:


def resize_images(image_dir, output_dir, size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)

    for i, image in enumerate(images[:5]):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = img.resize(size, Image.ANTIALIAS)
                img.save(os.path.join(output_dir, image), img.format)
        if(i + 1) % 10000 == 0:
            print("[{}/{}] Image Resized".format(i + 1, num_images))


# In[ ]:


img_size = [32, 32]
resize_images(FILE_PATH, './data', img_size)


# In[ ]:


def tokenize(input_dir, output_path):
    """function for tokenizing and getting id for image

    Args: 
        input_dir: directory path of which contains images
        output_file: name of output file
    
    Returns:
        output file
    
    Notes: 
        image name format: {class name(contains space)}_{location: int}_{image no: int}.jpg

    """
    names = os.listdir(input_dir)
    explain = dict()
    full_names = dict()
    for name in names[:5]:
        tokens = name.split('.')[0]  # remove file formatter
        tokens = tokens.split('_')
        classes = []
        id=""
        for token in tokens:
            if(token.isdigit()):
                id += token
            else:
                classes.append(token)
        explain[id] = classes
        full_names[id] = name
    
    with open(output_path + "imgclass_map.json", 'w', encoding="utf-8") as f:
        json.dump(explain, f)
    with open(output_path + "imgname_map.json", 'w', encoding="utf-8") as f:
        json.dump(full_names, f)

    return explain, full_names


# In[ ]:


explain, full_names = tokenize(FILE_PATH, './data/')


# In[ ]:


print(full_names)


# In[ ]:


print(explain)


# - 클래스를 int로 변경
# - 변경한 결과를 json 파일로 저장
# - 추가로 그 클래스가 몇 개인지도 저장

# In[ ]:


class Debris(object):
    def __init__(self, debris_file, img_mapped):
        self.debris_file = debris_file
        self.img_mapped = img_mapped
        self.debris2idx = {}
        self.idx2debris=[]
        self.idx = 0
        self.debris_cnt = Counter()

        self.read_file()
    
    def read_file(self):
        if(os.path.exists(self.debris_file)):
            with open(self.debris_file, 'r') as f:
                debris = json.load(f)
                self.debris2idx = debris['debris2idx']
                self.idx2debris = debris['idx2debris']
                self.idx = len(self.idx2debris) + 1
                self.debris_cnt = Counter(debris['debris_cnt'])
        else:
            self.build_debris()
            with open(self.debris_file, 'w', encoding = 'utf-8') as f:
                json.dump({
                    'debris2idx': self.debris2idx,
                    'idx2debris': self.idx2debris,
                    'debris_cnt': self.debris_cnt
                }, f)
    
    def add_debris(self, debris):
        if not debris in self.debris2idx:
            self.debris2idx[debris] = self.idx
            self.idx2debris.append(debris)
            self.idx += 1

    def build_debris(self):
        for _, debris in self.img_mapped.items():
            for word in debris:
                self.add_debris(word)
            self.debris_cnt.update(debris)

    def __call__(self, debris):
        return self.debris2idx[debris]
    
    def __len__(self):
        return len(self.debris2idx)


# In[ ]:


class MarinTrashDS(Dataset):
    def __init__(self, explain, full_names, debris_file, trasform=None):
        self.debris = Debris(debris_file = debris_file, img_mapped=explain)
        self.transform = transform
        self.img_index =  [ele for ele in explain]
        self.num_classes = len(self.debris)

    def __getitem__(self, index):
        img_id = self.img_index[index]
        classes = explain[img_id]
        target = np.zeros(self.num_classes, dtype = np.float)
        for word in classes:           
            int_target = self.debris(word)
            target[int_target] = 1.0
        target = torch.Tensor(target)

        path = './data/' + full_names[img_id]        
        img = Image.open(path).convert('RGB')

        if self.transform is not None: 
            img = self.transform(img)

        return img, target
    
    def __len__(self):
        return len(self.img_index)


# In[ ]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = 0.5, std = 0.5)                                
])


# In[ ]:


train_dataset = MarinTrashDS(explain = explain, full_names = full_names, debris_file = './data/debris.json', trasform = transform)


# In[ ]:


tr_index, val_index = train_test_split(list(range(len(train_dataset))), test_size = 0.2, shuffle = True)
tr_sampler = SubsetRandomSampler(tr_index)
val_sampler = SubsetRandomSampler(val_index)

