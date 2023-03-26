""" train and test dataset

author baiyu
"""
import os,csv,pickle
import sys
import pickle
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import torch
import random
from torch.utils.data import Dataset



class CEGTrain(Dataset):
    """CEG train dataset  derived from
    torch.utils.data.DataSet
    """
    def _loader(path):
        '''定义读取文件的格式'''
        img = Image.open(path).resize((128, 128))
        if img.mode != 'L':
            img = img.convert('L')
#         print('img.mode:',img.mode)
        return img

    def __init__(self, path, transform,loader = _loader):
        super(CEGTrain, self).__init__()  # 对继承自父类的属性进行初始化
        #new_label_train.csv Kfold/0_train.csv new_label2_train ecg_sq_0113_更新label2
        fh = open(os.path.join(path, 'sub_train_train.csv'), 'r')  # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        reader = csv.reader(fh)
        data = []
        ids = []
        for row in reader:
            if len(row) != 0:
                data.append((row[0],int(row[1])))  # (图片信息，lable)
#                  #训练集加上术前几天的心电图 
#                 indx = row[0].split('/')[-1].split('_')[0]
#                 ids.append(indx)    
#         all_data = get_sqn(path,ids)
#         data += all_data
        self.data = data
        self.transform = transform
        self.loader = loader
    def __len__(self):
        '''返回数据集的长度'''
        return len(self.data)

    def __getitem__(self, index):
        fn, label = self.data[index]
        img = self.loader(path=fn)
        index = fn.split('/')[-1].split("_")[0]
        if self.transform is not None:
            img = self.transform(img)
        if torch.cuda.is_available():
            img = torch.tensor(img, dtype=torch.float32).cuda()
            label = torch.tensor(label, dtype=torch.float32).cuda()
        return (img,index),label
    
    def get_labels(self):
        labels = []
        data = self.data
        for d in data:
            fn,label = d
            labels.append(label)
        return labels
    
class ECGVal(Dataset):
    """CEG  test dataset, derived from
    torch.utils.data.DataSet
    """

    def _loader(path):
        '''定义读取文件的格式'''
        img = Image.open(path).resize((128, 128))
        if img.mode != 'L':
            img = img.convert('L')
#         print('img.mode:',img.mode)
        return img

    def __init__(self, path, transform, loader = _loader):
        super(ECGVal, self).__init__()  # 对继承自父类的属性进行初始化
        #new_label_test.csv  Kfold/0_test.csv ecg_sq_0113_更新label2 step1_all_random_test.csv
        #Kfold/560/2_test step1_good Kfold/560/0_test.csv
#         fh = open(os.path.join(path, 'step1_good.csv'), 'r')  
        fh = open(os.path.join(path, 'sub_train_test.csv'), 'r')
        
        reader = csv.reader(fh)
        data = []
        names = get_all(path)
        for row in reader:
            if len(row) != 0:
#                 old_name = row[0].split('/')[-1]
#                 image = random_choice(row[0],names)
#                 row[0] = row[0].replace(old_name,image)
#                 row[0] = row[0].replace('step1_good','step1_all')
                data.append((row[0],int(row[1])))  # (图片信息，lable)
        self.data = data
        self.transform = transform
        self.loader = loader

    def __len__(self):
        '''返回数据集的长度'''
        return len(self.data)

    def __getitem__(self, index):
        fn, label = self.data[index]
        img = self.loader(path=fn)
        index = fn.split('/')[-1].split("_")[0]
        if self.transform is not None:
            img = self.transform(img)
        if torch.cuda.is_available():
            img = torch.tensor(img, dtype=torch.float32).cuda()
            label = torch.tensor(label, dtype=torch.float32).cuda()
        return (img,index),label
    def get_labels(self):
        labels = []
        data = self.data
        for d in data:
            fn,label = d
            labels.append(label)
        return labels
    
class ECGTest(Dataset):
    """CEG  test dataset, derived from
    torch.utils.data.DataSet
    """

    def _loader(path):
        '''定义读取文件的格式'''
        img = Image.open(path).resize((128, 128))
        if img.mode != 'L':
            img = img.convert('L')
#         print('img.mode:',img.mode)
        return img

    def __init__(self, path, transform, loader = _loader):
        super(ECGTest, self).__init__()  # 对继承自父类的属性进行初始化
        #new_label_test.csv  Kfold/0_test.csv ecg_sq_0113_更新label2 step1_all_random_test.csv
        #Kfold/560/2_test step1_good Kfold/560/0_test.csv
#         fh = open(os.path.join(path, 'step1_all.csv'), 'r')  
        fh = open(os.path.join(path, '560_test.csv'), 'r')  
        
        reader = csv.reader(fh)
        data = []
        names = get_all(path)
        for row in reader:
            if len(row) != 0:
            #random choice
                #image = random_choice(row[0],names)
                #data.append((image,int(row[1]))) 
                data.append((row[0],int(row[1])))  # (图片信息，lable)
        self.data = data
        self.transform = transform
        self.loader = loader

    def __len__(self):
        '''返回数据集的长度'''
        return len(self.data)

    def __getitem__(self, index):
        fn, label = self.data[index]
        img = self.loader(path=fn)
        index = fn.split('/')[-1].split("_")[0]
        if self.transform is not None:
            img = self.transform(img)
        if torch.cuda.is_available():
            img = torch.tensor(img, dtype=torch.float32).cuda()
            label = torch.tensor(label, dtype=torch.float32).cuda()
        return (img,index),label
    def get_labels(self):
        labels = []
        data = self.data
        for d in data:
            fn,label = d
            labels.append(label)
        return labels
    
def get_sqn(path,ids):
    fh = open(os.path.join(path, '1103_good.csv'), 'r')  # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
    reader = csv.reader(fh)
    data = []
    for row in reader:
        if len(row) != 0:
            index = row[0].split('/')[-1].split('_')[0]
            for ind in ids:
                if int(index)==int(ind) and 'all' not in row[0]:
#                     print(row[0])
                    data.append((row[0],int(row[1])))  # (图片信息，lable)
    return data

from random import choice

def get_ABIDEn(path,ids):
    fh = open(os.path.join(path, '02526.csv'), 'r')  # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
    reader = csv.reader(fh)
    data = []
    for row in reader:
        if len(row) != 0:
            index = row[0].split('/')[-1].split('_')[0]
            for ind in ids:
                if int(index)==int(ind) and 'all' not in row[0]:
#                     print(row[0])
                    data.append((row[0],int(row[1])))  # (图片信息，lable)
    return data

def random_choice(data,names):
    index = data.split('/')[-1].split('_')[0]
#     print(index)
    same_id = []
    for name in names:
        ind =  name.split('/')[-1].split('_')[0]
        if ind == index:
            root =  data.split('/')[0]
            name = name.split('/')[-1]
#             print('exit path',root+'/'+name)
            same_id.append(root+'/'+name)
    
#     print('random:',random.choice(same_id))
    random.choice(same_id)
    random.choice(same_id)
    random.choice(same_id)
    random.choice(same_id)
    random.choice(same_id)
    return random.choice(same_id)

def get_all(path):
    fh = open(os.path.join(path, 'step1_all.csv'), 'r')  # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
    reader = csv.reader(fh)
    names = []
    for row in reader:
        if len(row) != 0:
            names.append(row[0])  # (图片信息，lable)
    return names