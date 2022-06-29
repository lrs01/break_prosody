
from operator import mod
from matplotlib.pyplot import flag
from numpy import save
import torch 
import time
from time import sleep
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os,inspect
current_dir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.chdir(current_dir)
sys.path.append('../')
import numpy as np
from Dateset import Dataset_l_f
from tool.k_Cross import k_cross
from get_f_l_pkl import get_feature_label,get_train_test_form

## 将输出写入到相关的日志文件中
# import sys
# log_path = '/home/lrs/dengfeng.p/break_feature/log/m_1_5_log.txt'
# sys.stdout = open(log_path, mode='a+', encoding='utf-8')


def acc(predict,label):

    # cor = 0 ## 简单版本的ACC计算
    # count = 0

    cor_0 =0
    cor_1 =0
    cor_2 =0
    cor_3 =0
    cor_4 =0

    total_0 = 0
    total_1 = 0
    total_2 = 0
    total_3 = 0
    total_4 = 0

    acc_batch_0 = 0 
    acc_batch_1 = 0
    acc_batch_2 = 0
    acc_batch_3 = 0
    acc_batch_4 = 0

    for i in label[0]:
        if i==0:
            total_0 +=1
        elif i ==1 :
            total_1 +=1
        elif i ==2 :
            total_2 +=1
        elif i ==3 :
            total_3 +=1
        elif i ==4 :
            total_4 +=1
    
    for a,b in (zip(predict,label[0])):

        if a.item() == b.item():
            if b.item() ==0 :
                cor_0 +=1
            elif b.item() ==1 :
                cor_1 +=1
            elif b.item() == 2:
                cor_2 +=1
            elif b.item() ==3 :
                cor_3 +=1
            elif b.item() ==4 :
                cor_4 +=1
    
    if total_0 == 0:
        acc_batch_0 = 2
    else :
        acc_batch_0 = cor_0 / total_0

    if total_1 == 0:
        acc_batch_1 = 2
    else :
        acc_batch_1 = cor_1 / total_1
    
    if total_2 == 0:
        acc_batch_2 = 2
    else :
        acc_batch_2 = cor_2 / total_2
    
    if total_3 == 0:
        acc_batch_3 = 2
    else :
        acc_batch_3 = cor_3 / total_3
    
    if total_4 == 0:
        acc_batch_4 = 2
    else :
        acc_batch_4 = cor_4 / total_4

        
    

    return acc_batch_0, acc_batch_1, acc_batch_2, acc_batch_3, acc_batch_4

        

        ## 简单版本的ACC计算
        # if b.item() == 0:
        #     count +=1
        #     continue
        # if a.item() == b.item() :
        #     cor +=1
    # return total_0, total_1, total_2, total_3, total_4

def test_model(save_model,ind):
   
    device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
    y = k_cross()
    _,test = get_train_test_form(y,ind)
    mydataset = Dataset_l_f(train=test,flag=0)
    loader = DataLoader(mydataset,batch_size=1,shuffle=True)
    # model = save_model
    model = torch.load(save_model) ## 加载模型
    model.eval()## 模型测试模式
    model.to(device)
    count = 0
    cor = 0

    with torch.no_grad(): ## 测试不计算梯度
        acc_batch =0
        total = 0
        final_list = []
        for i ,batch in enumerate(loader):
            list_tool = []
            feature , label = batch
            feature_ = feature.to(device)
            label_ = label.to(device)
            outputs = model(feature_)
            # print(label_[0])
            # exit()

            _,maxdd = torch.max(outputs[0].permute(1,0).contiguous(),dim=1)
            # acc_k ,count =acc(predict=maxdd,label=label_)
            # acc_batch +=acc_k
            # total +=(label_.shape[1]-count) 
            # Acc = acc_batch/total
            
            acc_batch_0, acc_batch_1, acc_batch_2, acc_batch_3, acc_batch_4 = acc(predict=maxdd,label=label_)
            list_tool = [acc_batch_0, acc_batch_1, acc_batch_2, acc_batch_3, acc_batch_4]
            final_list.append(list_tool)
    return final_list

            
       
    # print(acc_batch/total)
    # return Acc



            
           
if __name__=="__main__":

    model_file_path = "/home/lrs/dengfeng.p/break_feature/Model/M_1_5_2"
    g = os.listdir(model_file_path)
    g.sort()
    acc_0 =[]
    acc_1 = []
    acc_2 = []
    acc_3 = []
    acc_4 = []

    for i in g:
        save_md = model_file_path + '/' + i
        # acc_i = test_model(save_md,int(i[-4]))
        # list_tool.append(acc_i)
        final_list_ = test_model(save_md,int(i[-4]))
        final_list_ = np.array(final_list_)
        for j in range(final_list_.shape[0]):
            if final_list_[j][0] != 2 :
                acc_0.append(final_list_[j][0])
            if final_list_[j][1] != 2 :
                acc_1.append(final_list_[j][1])
            if final_list_[j][2] != 2:
                acc_2.append(final_list_[j][2])
            if final_list_[j][3] != 2 :
                acc_3.append(final_list_[j][3])
            if final_list_[j][4] != 2 :
                acc_4.append(final_list_[j][4])

    acc_0 = np.mean(np.array(acc_0))
    acc_1 = np.mean(np.array(acc_1))
    acc_2 = np.mean(np.array(acc_2))
    acc_3 = np.mean(np.array(acc_3))
    acc_4 = np.mean(np.array(acc_4))
    print(acc_0,acc_1,acc_2,acc_3,acc_4)

    # print(final_list_.shape)

        # exit
    # list_tool = np.array(list_tool)
    # acc_mean = np.mean(list_tool)
    # print(acc_mean) ## 


    pass