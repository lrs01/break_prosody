from audioop import cross
from cProfile import label
from operator import imod
from matplotlib.pyplot import cla
# from black import re
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import random
import sys
import sys
curPath = os.path.abspath(os.path.dirname('/home/lrs/dengfeng.p/break_feature/tool'))
sys.path.append(curPath)
from tool.k_Cross import k_cross
from pathlib import Path
from torch.utils.data import  Dataset,DataLoader
from get_f_l_pkl import get_feature_label,get_train_test_form
import warnings
warnings.filterwarnings('ignore')


def collate_fn_my(batch):

 
    seq_lens = [] ## 保存一个batch 中每个数据的长度
    batch_size = len(batch)
    feature_max = max([x[0].shape[0] for x in batch])#找到当前bathc 中最大的序列长度，也就是音节数量最多的序列
    feature_dim = batch[0][0].shape[-1] ## 目前是11个特征
    feature_padding = torch.zeros(batch_size,feature_dim,feature_max) # 生成一个最大的零矩阵，最后把每个batch的序列填充进去

    #--------------------------------------------------------------------#
    #padding 过程
    for i in range(batch_size):
        feature_len = batch[i][0].shape[0]
        seq_lens.append(feature_len)
        feature_padding[i,:,:feature_len] = batch[i][0].permute(1,0).unsqueeze(0)
    #--------------------------------------------------------------------#

    # tool = []
    # for item in batch:
    #     print(item[1])
    #     tool.append(item[1])
    # print(len(tool))
    target = [torch.LongTensor(item[1]) for item in batch]## list形式
    # target = np.array(target)
    # target = torch.from_numpy(target)
    # print(target)

    return feature_padding,target,seq_lens






    pass


class Dataset_l_f(Dataset):
    def __init__(self,train,flag):
        self.train = train
        self.flag = flag
        
    def __getitem__(self, index) :
        src = self.train[index]
        feature, label = get_feature_label(src,self.flag)
        return torch.FloatTensor(feature),label
    
        pass
    def __len__(self):

        return self.train.shape[0]

if __name__ == '__main__':
    c=0
    y = k_cross()
    train ,_ = get_train_test_form(y,0)
    mydataset = Dataset_l_f(train=train,flag=0)
    for b in DataLoader(mydataset,batch_size=4,collate_fn=collate_fn_my):
        f,l ,s= b
        print(f.shape,l.shape,s)
        exit()
        c+=1
    print(c)
    pass