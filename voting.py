import warnings
warnings.filterwarnings("ignore")
import torch
import os
import random
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import argparse
from conf import settings
from torch.autograd import Variable
from collections import Counter
from sklearn.metrics import f1_score,roc_curve,roc_auc_score,auc
from sklearn.metrics import classification_report,confusion_matrix
# from utils_image import get_network, get_training_dataloader,get_test_dataloader,accuracy,median_ci
from utils import get_network, get_training_dataloader,get_test_dataloader
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(42)

def get_model(path):
    net = get_network(args)
    net.load_state_dict(torch.load(path))
    return net


def Test(testLoader,BATCHSIZE):
    
    pre=[]
    prob_all =[]
    label_all =[]
    y_scores = []
    vote_correct = 0.0
    
#     net1 = torch.load('./checkpoint/image_clinical/auc0.pth')
#     net2 = torch.load('./checkpoint/image_clinical/auc1.pth')
#     net3 = torch.load('./checkpoint/image_clinical/auc2.pth')
#     net4 = torch.load('./checkpoint/image_clinical/auc3.pth')
#     net5 = torch.load('./checkpoint/image_clinical/auc4.pth')
    
    net1 = get_model('./checkpoint/image_clinical/auc0.pth')
    net2 = get_model('./checkpoint/image_clinical/auc1.pth')
    net3 = get_model('./checkpoint/image_clinical/auc2.pth')
    net4 = get_model('./checkpoint/image_clinical/auc3.pth')
    net5 = get_model('./checkpoint/image_clinical/auc4.pth')
    
#     net1 = get_model('./checkpoint/f1_model/0_model.pth')
#     net2 = get_model('./checkpoint/f1_model/1_model.pth')
#     net3 = get_model('./checkpoint/f1_model/2_model.pth')
#     net4 = get_model('./checkpoint/f1_model/3_model.pth')
#     net5 = get_model('./checkpoint/f1_model/4_model.pth')
    
#     net1 = get_model('./checkpoint/min_loss/0_model.pth')
#     net2 = get_model('./checkpoint/min_loss/1_model.pth')
#     net3 = get_model('./checkpoint/min_loss/2_model.pth')
#     net4 = get_model('./checkpoint/min_loss/3_model.pth')
#     net5 = get_model('./checkpoint/min_loss/4_model.pth')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    mlps = [net1.to(device), net2.to(device), net3.to(device), net4.to(device), net5.to(device)]
    
    mlps_correct = [0 for i in range(len(mlps))]
    mlp_score_1,mlp_score_2 ,mlp_score_3 ,mlp_score_4 ,mlp_score_5 = list(),list(),list(),list(),list()
    
    mlps_scores = [mlp_score_1,mlp_score_2 ,mlp_score_3 ,mlp_score_4 ,mlp_score_5]
    
    with torch.no_grad():
        for (images, txts, label) in ecg_test_loader:
            if args.gpu:
                images = images.cuda()
                txts = txts.cuda()
                label = labels.cuda()
            images = Variable(images, requires_grad=True)
            labels = Variable(label, requires_grad=True)
#         for i, (img, label) in enumerate(testLoader):
#             if "cuda:0":
#                 img = img[0].cuda()
#                 label = label.cuda()
#                 print('label.shape:',label.shape)
#                 print('img.size:',img.size)
            for i, mlp in enumerate(mlps):
                mlp.eval()
                output = mlp(images,x_txt=txts)
#                 print('output',output)
                _, prediction = torch.max(output, 1) # 按行取最大值
                
                pre_num = prediction.cpu().numpy()
                mlps_correct[i] += (pre_num == label.cpu().numpy()).sum()
                pre.append(pre_num)
                mlps_scores[i].extend(output[:,1].cpu().numpy())
            arr = np.array(pre)  # (3, 100)
#             print(arr[:, 0])  # [3 3 5]
#             print(Counter(arr[:, 0])) # Counter({3: 0, 5: 1})
#             print(Counter(arr[:, 0]).most_common(1))  # [(3, 0)]
#             print(Counter(arr[:, 0]).most_common(1)[0])  # (3, 0)
#             print(Counter(arr[:, 0]).most_common(1)[0][0]) # 
            pre.clear()
            result = [Counter(arr[:, i]).most_common(1)[0][0] for i in range(label.shape[0])]
            vote_correct += (result == label.cpu().numpy()).sum()
            label_all.extend(label.cpu().numpy())
            prob_all.extend(result)
       
        mean_scores = []
        for i  in range(len(mlps_scores[0])):
            sum_score = 0.0
            for j in  range(len(mlps_scores)):
                sum_score += mlps_scores[j][i]
            mean_score = (sum_score / len(mlps_scores))
            mean_scores.append(mean_score)
        mean_scores = np.asarray(mean_scores)
        test_auc = roc_auc_score(label_all, mean_scores)
        fpr, tpr, thresholds = roc_curve(label_all,mean_scores,drop_intermediate=False)
        f1 = f1_score(label_all, prob_all) 
        conf_mat = confusion_matrix(label_all, prob_all)
        Sensitivity = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[1, 0])
        Specificity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])

    # len(testLoader) = 100,为啥除以100？因为就是100个bathsize呀。
    print("集成模型的正确率: "+ str(vote_correct / len(label_all)))
    print("集成模型的AUC: " + str(test_auc))
    print("集成模型的F1: " + str(f1))
    print("集成模型的Sensitivity: " + str(Sensitivity))
    print("集成模型的Specificity: " + str(Specificity))
    
    for idx, coreect in enumerate(mlps_correct):
        print("模型"+str(idx)+"的正确率为："+str(coreect / len(label_all)))
    for idx, score in enumerate(mlps_scores):
        print("模型"+str(idx)+"的AUC为："+str(roc_auc_score(label_all,mlps_scores[idx])))
    wdf = pd.DataFrame({'FPR':fpr,'TPR':tpr})
    wdf.to_csv('./data/AUC/voting.csv')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
#     parser.add_argument('-net', type=str, default='mymodel_all',help='net type') #required=True,
    parser.add_argument('-net', type=str, default='mymodel',help='net type') #包含txt
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    args = parser.parse_args()
    
    ecg_test_loader = get_test_dataloader(
        settings.ECG_TRAIN_MEAN,
        settings.ECG_TRAIN_STD,
        num_workers=0,
        batch_size=args.b,
        shuffle=False
    )
    Test(ecg_test_loader,args.b)