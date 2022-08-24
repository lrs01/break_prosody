from train import train,acc
import torch
import random
import numpy as np
from test_k import test_model,acc
from train2 import train as tr1,acc_include_0
# 种子，确保每次训练的随机数都一样
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



if __name__ == "__main__":
    setup_seed(2022)  # 使用随机数种子
    for i in range(5):
        
        path_save_model = "/disk2/lrs/dengfeng.p/break_feature/Model/ LA_3" + '/'+f'LA_{i}.pt'
        # tr1(path_save_model,i)
        # if i == 5 :
        #     break
        test_model(path_save_model,i+1)
        # pass
    # tr1(1)

    
        

    pass
