import numpy as np
from sklearn.model_selection import KFold
import os
import pickle

def k_cross():
    '''
        返回10个不同的lab序列
    '''
    feature_path = r'/disk2/lrs/dengfeng.p/break_feature/Data/feature'
    tg_list = os.listdir(feature_path) ## 列出 tgfile_path 下的所有文件夹
    tg_list.sort() ## 排序
    arr_lab = []
    c = 0 ## 计数功能
    for i in tg_list:
        lab_path = feature_path + '/' + i
        arr_lab.append(lab_path)
        c+=1
    
    X = arr_lab
    kf = KFold(n_splits=10)
    arr_ = []
    for train, test in kf.split(X):
        arr_tool = []
        for i in range(train.size):
            arr_tool.append(X[train[i]])
        for i in range(test.size):
            arr_tool.append(X[test[i]])
    # return train,test
        arr_.append(arr_tool)
    arr_ = np.array(arr_)
    return arr_


if __name__ == "__main__":

    # # arr_lab = k_cross_lab()
    # # print(arr_lab)
   
    # # x,y =  k_cross()
    # # print(x,y)
    # k_cross = k_cross()
    # print(k_cross)
    # x = k_cross_lab()
    x1 = k_cross()
    print(x1)
    print('yunxing')
    pass