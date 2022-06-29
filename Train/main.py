from train import train,acc
import torch
import random
import numpy as np
from test_k import acc,test_model
from train2 import train as tr1
# 种子，确保每次训练的随机数都一样
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



if __name__ == "__main__":
    setup_seed(2022)  # 使用随机数种子
    # for i in range(10):
    #     path_save_model = "/home/lrs/dengfeng.p/break_feature/Model/M_1_5_2" + '/'+f'm1_5_{i}.pt'
    #     train(path_save_model,i)
    #     # test_model(path_save_model,i)wxcf
    #     pass
    tr1(1)

    
        

    pass
