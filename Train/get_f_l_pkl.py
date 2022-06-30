
import encodings
from re import X
import torch 
import os 
import numpy as np
import pickle
import sys
curPath = os.path.abspath(os.path.dirname('/disk2/lrs/dengfeng.p/break_feature/tool'))
sys.path.append(curPath)
from tool.k_Cross import k_cross
from tool.data_deal import z_score_1



def get_feature_label(tg_path,flag):  
    
    tg_path = str(tg_path)
    pkl = os.listdir(tg_path)
    list_tool = []
    for  i in range(len(pkl)):
        if pkl[i] != 'label.pkl':
            x_pa = os.path.join(tg_path, pkl[i])
            data = open(x_pa,'rb')
            outp = pickle.load(data) ## numpy
            list_tool.append(outp)
            data.close()
        else :
                x_pa = os.path.join(tg_path,pkl[i])
                data = open(x_pa,'rb')
                label_out = pickle.load(data)
                label_out = torch.from_numpy(label_out)
                data.close()

   
    #---------------------------------------------------------------------#
    # modified by lrs in 6/18
    ## 归一化操作，为了尽可能自限制测试集，只对训练集进行归一化
    if flag == 0:
        feature_np = np.hstack((list_tool[0],list_tool[1],list_tool[2]))
        feature_np_z = z_score_1(feature_np)
        feature_np_z_tensor = torch.from_numpy(feature_np_z).float()
        return feature_np_z_tensor,label_out

    #--------------------------------------------------------------------#
    elif flag == 1:

        t1 = torch.from_numpy(list_tool[0]).float()
        t2 = torch.from_numpy(list_tool[1]).float()
        t3 = torch.from_numpy(list_tool[2]).float()

        feature = torch.cat((t1,t2,t3),dim=1).float()
        return feature,label_out

def get_train_test_form(lab_path,iterm):
    '''
        返回指定训练和测试的表单
        此函数的输出结果可以作为get_feature_label的参数传入
    '''
    
    arr_tool_train = []
    arr_tool_test = []
    for i in range(lab_path[iterm].shape[0]): ## 注意numpy数组的shape输出是一个元组。
        if i < lab_path[iterm].shape[0]-1:
            path = str(lab_path[iterm][i])
            path_dir = os.listdir(path)
            for j in range(len(path_dir)):
                tgpath = os.path.join(path,path_dir[j])
                arr_tool_train.append(tgpath)
        else :
            path = str(lab_path[iterm][i])
            path_dir = os.listdir(path)
            for j in range(len(path_dir)):
                tgpath = os.path.join(path,path_dir[j])
                arr_tool_test.append(tgpath)

    # print(lab_path[0].shape[0])
    arr_tool_train = np.array(arr_tool_train) ## 本次训练表单
    arr_tool_test = np.array(arr_tool_test) ## 本次测试表单

    return arr_tool_train, arr_tool_test
    



if __name__ == '__main__':
    
    # test_path = '/home/lrs/dengfeng.p/break_feature/Data/feature/f001lab/f001001_01'
    # x,y,= get_feature_label(test_path,flag = 1)
    
    # print(type(x),type(y))
    # print(x,y.shape,)
    # # y = k_cross()
    # # x ,y = get_train_test_form(y,iterm=0)
    # # print(x.shape,y.shape)
    pass