# train.py
#!/usr/bin/env	python3

""" train network using pytorch

"""

import os
import random
import sys
import argparse
import time
import torch
import torch.nn as nn
from sklearn.metrics import f1_score,roc_curve,roc_auc_score,auc
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from conf import settings
from utils_image import get_network, get_training_dataloader, get_test_dataloader,accuracy,get_val_dataloader
#,median_ci
from FocalLoss import FocalLoss
from torch.autograd import Variable
from sklearn.metrics import classification_report,confusion_matrix



import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TZ'] = 'UTC-8'
time.tzset()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(42)


def train(epoch):
    prob_all = []
    label_all = []
    y_scores = []
    correct = 0.0
    net.train()
    for batch_index, (images, labels) in enumerate(ecg_training_loader):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
        optimizer.zero_grad()
        outputs,features = net(images[0])
        loss = loss_function(outputs, labels.long())
        loss.backward()
        optimizer.step()
        if epoch ==1 :
            f = features.data.cpu().numpy()
            df = pd.DataFrame(f)
            df.to_csv('init_features_right2.csv', mode='a', header=False)
        
@torch.no_grad()
def eval_training(epoch,best_auc):
    
    start = time.time()
    net.eval()
    
    prob_all =[]
    label_all =[]
    y_scores = []
    val_loss = 0.0 # cost function error
    correct = 0.0
    first = [] # 模型打分结果中类别0的概率，是一个n行 ，1列的数组
    preds = []  # 模型的打分结果中类别1的概率，是一个n行 ，1列的数组
    outputs = []
    test_acc = 0.0 
    with torch.no_grad():
        index = 0 
        for i, (images, labels) in enumerate(ecg_val_loader):
            index = index + 1
            if args.gpu:
                images = images[0].cuda()
                labels = labels.cuda()
            image = Variable(images[0], requires_grad=True)
            labels = Variable(labels, requires_grad=True)
            with torch.no_grad():
                outputs,features = net(image)
            loss = loss_function(outputs, labels.long())
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(labels).sum()
            outputs = outputs.cpu().numpy()
            prob_all.extend(np.argmax(outputs,axis=1))
            y_scores.extend(outputs[:,1])
            label_all.extend(labels.cpu().numpy())
    finish = time.time()
    
    test_auc = roc_auc_score(label_all, prob_all)
#     if test_auc >0.72 and test_auc<0.74:
#         print('write:',test_auc)
#         target_names = ['labels_0','labels_1']
#         print(classification_report(label_all, prob_all.cpu(), labels=labels, target_names=target_names, digits=3))

#         wdf_auc = pd.DataFrame({'label_all':label_all,'prob_all':prob_all})
#         wdf_auc.to_csv('./data/AUC/wdf_auc2.csv')

    fpr, tpr, thresholds = roc_curve(label_all,y_scores,drop_intermediate=False)
#     print(fpr,tpr)
    print('test_auc:',test_auc)
    print('auc(fpr, tpr):',auc(fpr, tpr))
    f1 = f1_score(label_all, prob_all) #, average='weighted'
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    labels = [0,1]
    target_names = ['labels_0','labels_1']
    print('-'*50)
    print(classification_report(label_all, prob_all, labels=labels, target_names=target_names, digits=3))

    conf_mat = confusion_matrix(label_all, prob_all)
    print('TP:',conf_mat[1, 1])
    print('FN:',conf_mat[1, 0])
    print('TN:',conf_mat[0, 0])
    print('FP:',conf_mat[0, 1])
    Sensitivity = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[1, 0])
    Specificity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
    Recall = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    Precision = conf_mat[1, 1] / (conf_mat[0, 1] + conf_mat[1, 1])
#     print('-'*50)
#     print('conf_mat_auc',conf_mat)
    print('epoch {} ,Test  Accuracy: {:.4f},Sensitivity: {:.4f}, Specificity: {:.4f}, Recall: {:.4f}, Precision: {:.4f}, F1: {:.4f}, AUC: {:.4f}, loss:{:.4f}'.format(
        epoch,
        correct.float()  / len(ecg_test_loader.dataset),
        Sensitivity,
        Specificity,
        Recall,
        Precision,
        f1,
        test_auc,
        val_loss
    ))
    print('-'*50)
    return f1,test_auc, fpr, tpr, val_loss,conf_mat

@torch.no_grad()
def test(ecg_test_loader):
    net = get_network(args)
#     net.load_state_dict(torch.load('./checkpoint/0426/f1_model/1_model.pth'))
    net.load_state_dict(torch.load('./checkpoint/0426/auc_model/step3_model.pth'))
#     net.load_state_dict(torch.load('./checkpoint/0426/min_loss/1_model.pth'))
    device = torch.device('cuda:0')
    net.to(device)
    net.eval()
    
    prob_all =[]
    label_all =[]
    y_scores = []
    with torch.no_grad():
        for i, (images, label) in enumerate(ecg_test_loader):
            if torch.cuda.is_available():
                image = images[0].cuda()
                label = label.cuda()
            image = Variable(image, requires_grad=True)
            label = Variable(label, requires_grad=True)
            with torch.no_grad():
                output,features = net(image)
            output = output.cpu().numpy() 
            prob_all.extend(np.argmax(output,axis=1))
            y_scores.extend(output[:,1])
            label_all.extend(label.cpu().numpy())
        auc = roc_auc_score(label_all, y_scores)
    print("AUC:{:.4f}".format(auc))
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='mymodel_all',help='net type') #required=True,
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.0003, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()
    
    net = get_network(args)
    device = torch.device('cuda:0')
    net.to(device)
    #data preprocessing:
    ecg_training_loader = get_training_dataloader(
        settings.ECG_TRAIN_MEAN,
        settings.ECG_TRAIN_STD,
        num_workers=0,
        batch_size=args.b,
        shuffle=False
    )
    ecg_val_loader = get_val_dataloader(
        settings.ECG_TRAIN_MEAN,
        settings.ECG_TRAIN_STD,
        num_workers=0,
        batch_size=args.b,
        shuffle=False
    )
    ecg_test_loader = get_test_dataloader(
        settings.ECG_TRAIN_MEAN,
        settings.ECG_TRAIN_STD,
        num_workers=0,
        batch_size=args.b,
        shuffle=False
    )
    loss_function = FocalLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay =0.001)  # optimizer 使用 Adam,weight_decay=0.01

    #开始循环训练
    min_epoch = 5
    train_best_auc = 0.0
    train_best_f1 = 0.0
    best_auc = 0.0
    best_f1 = 0.0
    best_avloss = 999
    best_sn = 0.0
    min_loss = 10 #随便设置一个比较大的数
#     print('Train on model:',net)

    auc_list = []
    for epoch in range(1, settings.EPOCH + 1):
        train(epoch)
        f1,test_auc, fpr, tpr, val_loss,conf_mat = eval_training(epoch,best_auc)
        auc_list.append(test_auc)
        if f1 > best_f1:
            best_f1 = f1
            print('>>>>Test_best_f1 in epoch: {},f1:{}'.format(epoch,best_f1))
#             print('conf_mat',conf_mat)
#             torch.save(net.state_dict(),'./checkpoint/0426/f1_model/cnn.pth')
        print('==>',test_auc)
        # if epoch == 33:
        #     wdf = pd.DataFrame({'FPR':fpr,'TPR':tpr})
        #     wdf.to_csv('./data/AUC/0426/172.csv')
        # if epoch == 35:
        #     wdf = pd.DataFrame({'FPR':fpr,'TPR':tpr})
        #     wdf.to_csv('./data/AUC/0426/173.csv')
        # if epoch == 32:
        #     wdf = pd.DataFrame({'FPR':fpr,'TPR':tpr})
        #     wdf.to_csv('./data/AUC/0426/175.csv')
        # if epoch == 50:
        #     wdf = pd.DataFrame({'FPR':fpr,'TPR':tpr})
        #     wdf.to_csv('./data/AUC/0426/178.csv')
            
        if test_auc > best_auc:
            best_auc = test_auc
#             print('conf_mat_auc',conf_mat)
#             print('>>>>Test_best_auc in epoch: {}, auc{}'.format(epoch,test_auc))
            wdf = pd.DataFrame({'FPR':fpr,'TPR':tpr})
            wdf.to_csv('./data/AUC/2023/0.csv')
#             torch.save(net.state_dict(),'./checkpoint/0426/auc_model/cnn.pth')
#         if epoch >= min_epoch and val_loss < min_loss:
#             min_loss = val_loss
#             print("^^^^^^^^^^save model^^^^^^^^^")
#             torch.save(net.state_dict(),'./checkpoint/0426/min_loss/cnn.pth')
        
    print('test_best_auc: {:.3f}, test_best_f1: {:.3f}'.format(best_auc,best_f1))
#     torch.save(net.state_dict(),'./checkpoint/0426/all/cnn.pth')
    
#     test(ecg_test_loader)