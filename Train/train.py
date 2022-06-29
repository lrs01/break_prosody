
from itertools import count
from matplotlib.pyplot import flag
import torch 

import torch.nn as nn 
from torch.utils.data import  Dataset,DataLoader  
import os  
import math
import sys
import os,inspect
current_dir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.chdir(current_dir)
sys.path.append('../')
import numpy
from Model.Model import model
from Train.Dateset import Dataset_l_f
from tool.k_Cross import k_cross
from get_f_l_pkl import get_feature_label,get_train_test_form

def acc(predict,label):

    cor = 0
    count = 0
    for a,b in (zip(predict,label[0])):
        if b.item() == 0:
            count +=1
            continue
        if a.item() == b.item() :
            cor +=1
    return cor,count


def train(path_save_model,ind):

    device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
    model_ = model()
    # model_ = nn.DataParallel(model_) ## 多卡训练
    model_.to(device)

    y = k_cross()
    train ,_ = get_train_test_form(y,ind)
    mydataset = Dataset_l_f(train=train,flag=0)
    optimizer = torch.optim.Adam(model_.parameters(), lr=0.00005,weight_decay = 1e-3)
    loss_function = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,5,5,7,10]),size_average=True).to(device)
    loader = DataLoader(mydataset,batch_size=1,shuffle=True)

    epoch_size = 200

    for epoch in range(epoch_size):
        model_.train() ## 开启训练模式
        print(f"Epoch:{epoch}")
        loss_batch = 0
        acc_batch = 0
        total = 0
        for i , batch in enumerate(loader):
            feature,label = batch
            feature_ = feature.to(device)
            label_ = label.to(device)
            outputs = model_(feature_)
            loss = 0 
            loss = loss_function(outputs[0].permute(1,0).contiguous(),label_[0])
            _,maxdd = torch.max(outputs[0].permute(1,0).contiguous(),dim=1)
            # exit()
            # acc_k = torch.sum(maxdd == label_[0])
            acc_k ,count= acc(predict=maxdd,label=label_)
            acc_batch +=acc_k
            total += (label_.shape[1]-count)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_batch += loss
            if i % 20 == 0:
                print(' ',i,'batch_loss:',loss)
           
        print(f'acc_epoch:{acc_batch / total}')
        print(f"epoch_loss:{loss_batch}")
    
    path_save_model_ =path_save_model
    os.mknod(path_save_model_)
    torch.save(model_,path_save_model_)

if __name__ == '__main__':
    train()
    pass