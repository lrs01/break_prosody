
from itertools import count
from matplotlib.pyplot import flag
from prometheus_client import Summary
import torch 
from torch.utils.tensorboard import SummaryWriter ## pytorch 训练可视化
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
from Train.Dateset import Dataset_l_f, collate_fn_my
from tool.k_Cross import k_cross
from tool.data_deal import adjust_lr_rate
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

def acc_include_0(predict,label):
    '''
    此函数的目的是大概明确模型的对总共韵律标注的准确率
    '''
    cor = 0
    for a,b in  zip(predict,label):
        if a.item() == b.item():
            cor +=1
    return cor

def train(ind):
    ##———————————————————————————————参数设置—————————————————————————————————————————————————————##
    epoch_size = 600
    lr = 0.01
    batch_size = 6
    ##————————————————————————————————————————————————————————————————————————————————————————##
    device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
    model_ = model()
    model_ = nn.DataParallel(model_) ## 多卡训练
    model_.to(device)
    # writer = SummaryWriter(log_dir='/home/lrs/dengfeng.p/break_feature/log/t6003')  # tensorboard可视化
    y = k_cross() ## k-则分组，分成训练集和测试集
    train ,_ = get_train_test_form(y,ind) ## 得到训练集
    mydataset = Dataset_l_f(train=train,flag=0) ## flag=0代表训练集经过Z-score标准化
    optimizer = torch.optim.Adam(model_.parameters(), lr=lr,weight_decay = 1e-3)
    # scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)## 动态变化学习率
    loss_function = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,5,5,7,10]),size_average=True).to(device) ## 注意不能忘记把loss函数放到卡上
    loader = DataLoader(mydataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn_my)

    
    for epoch in range(epoch_size):
        ##————————————————————————————————————————————————————————————————————————————————————————##
        adjust_lr_rate(optimizer=optimizer,epoch=epoch,start_lr= lr)## 动态变化学习率
        model_.train() ## 开启训练模式
        print(f"Epoch:{epoch}")
        print("Epoch:{}  Lr:{:.2E}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        ##————————————————————————————————————epo参数————————————————————————————————————————————————————##
        loss_epo = 0
        acc_epo  = 0
        total_epo = 0
        acc_rate_epo =0
        #————————————————————————————————————————————————————————————————————————————————————————#
        for i , batch in enumerate(loader):
            feature,label,seq_lens = batch ## feature : padding之后的feature；label：list类型的；seq_lens:
            feature_ = feature.to(device)
            outputs = model_(feature_)
            #------------------------------------------------------------------##
            loss_batch = 0
            acc_batch = 0
            total_batch = 0 
            acc_rate_batch = 0
            #----------------------------------------------------------------##
            for k in range(len(seq_lens)):
                pred_k = outputs[k][:,:seq_lens[k]]
                pred_k = pred_k.permute(1,0).contiguous()
                label_k = label[k].to(device)
                loss_batch += loss_function(pred_k,label_k)
                _,predict_k = torch.max(pred_k,dim=1)
                acc_batch +=acc_include_0(predict=predict_k,label=label_k)
                total_batch +=seq_lens[k] 
            #—————————————————————————————————————————————————————————————————————#
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            #--——————————————————————————————————————————————————————————————————#
            loss_epo +=loss_batch
            acc_epo +=acc_batch
            total_epo +=total_batch
            acc_rate_batch = acc_batch / total_batch
            #--——————————————————————————————————————————————————————————————————#
            print(f'acc_batch:{acc_rate_batch}')
        #--——————————————————————————————————————————————————————————————————#
        acc_rate_epo = acc_epo / total_epo
        # scheduler.step()
        #--——————————————————————————————————————————————————————————————————#
        # writer.add_scalars("Train", {'loss_epo': loss_epo, 'acc_epo': acc_rate_epo, 'lr': optimizer.state_dict()['param_groups'][0]['lr']}, epoch)  # pytorch 训练可视化
        print(f'acc_rate_epo:{acc_rate_epo}')
        print(f"loss_epo:{loss_epo}")


    # path_save_model_ =path_save_model
    # os.mknod(path_save_model_)
    # torch.save(model_,path_save_model_)

if __name__ == '__main__':
    train(1)
    pass