import csv,os

def createImgIndex(dataPath,ratio):
    '''
    读取目录下面的图片制作包含图片信息、图片label的train.txt和val.txt
    dataPath: 图片目录路径
    ratio: val占比
    return：label列表
    '''
    print(dataPath)
    fileList = os.listdir(dataPath)
    with open('./data/sub_train_test.csv', 'w') as f:
        writer = csv.writer(f)
        for i in range(int(len(fileList)*ratio)):
            row = []
            if '.png' in fileList[i]: #and 'all' in fileList[i]:
                label = fileList[i].split('_')[-1].split('.')[0]
                #测试集仅纳入术前最近的一次心电图
                row.append(os.path.join(dataPath, fileList[i]))  # 图片路径
                row.append(label)
                if row is not None:
                    writer.writerow(row)
        f.close()
    with open('./data/sub_train_train.csv', 'w') as f:
        writer = csv.writer(f)
        for i in range(int(len(fileList) * ratio)+1, len(fileList)):
            row = []
            if '.png' in fileList[i]: #and 'sq' in fileList[i]:
                label = fileList[i].split('_')[-1].split('.')[0]
                row.append(os.path.join(dataPath,  fileList[i]))  # 图片路径
                row.append(label)
                if row is not None:
                    writer.writerow(row)
        f.close()
def createImgIndexAll(dataPath):
    '''
    读取目录下面的图片制作包含图片信息、图片label的train.txt和val.txt
    dataPath: 图片目录路径
    ratio: val占比
    return：label列表
    '''
    print(dataPath)
    fileList = os.listdir(dataPath)
    with open('./data/sub_test.csv', 'w') as f:
        writer = csv.writer(f)
        for i in range(int(len(fileList))):
            row = []
            if '.png' in fileList[i]: #and 'all' in fileList[i]:
                label = fileList[i].split('_')[-1].split('.')[0]
                row.append(os.path.join(dataPath, fileList[i]))  # 图片路径
                row.append(label)
                if row is not None:
                    writer.writerow(row)
        f.close()
if __name__ == '__main__':
    root = os.getcwd() + '/data/'
    # createImgIndexAll(root + '560/')
    # createImgIndex(root + 'step1_all/',0.3)
#     createImgIndex(root + '/image_d2_under_sampling',0.2)
#     createImgIndex(root + '/images',0.2)    
#     createImgIndex(root + '/sx',0.2)
####################20230212#################
    createImgIndex(root + 'new_sub_train/',0.3)
