import torch
import numpy as np 
import pandas as pd
import numpy as np
import scipy.stats as stats
import spacy

# ---------简单的数据z-score-----------------------------#
def z_score(x):
  #-----------------------------------------------------#
  # 为数据进行归一化，归一化的数据是经过特征concat之后的数据
  #-----------------------------------------------------#

  x = (x - np.mean(x)) / np.std(x)
    
  return x


#———————————————————modified  by lrs at 6/18-----------------------##
def z_score_1(data):
  '''
  ## data : 数组
     axis : 纬度
  '''
  output_data = stats.zscore(data,axis=1)
  
  return output_data
#———————————————————modified  by lrs at 6/18-----------------------##
  pass


def adjust_lr_rate(optimizer, epoch, start_lr):
  	#学习率动态衰减****增加
        #lr = start_lr * (1-epoch/Epoch)**0.9
        # lr = start_lr * min((epoch+1)/150,1)
        if epoch <20:
          lr = start_lr
        elif epoch>=20 and epoch <50:
          lr = start_lr *0.5
        elif epoch >=50 and epoch < 120 :
          lr = start_lr * 0.1
        elif epoch>=120 and epoch <250:
          lr = start_lr *0.05
        elif epoch>=250 and epoch <400 :
          lr = start_lr * 0.01
        else :
          lr = start_lr *0.005
        # lr = start_lr * (0.1 * (epoch // 3))
        for param_group in optimizer.param_groups:
                param_group['lr'] = lr


def compute_s_k(input):
  '''
  计算偏度 和 峰度
  '''
  s = pd.Series(input)
  return s.skew(),s.kurt()




  




  pass
if __name__ == '__main__' :

  nlp = spacy.load('zh_core_web_sm')
  doc = nlp(u'我是一个中国人,我爱中国')
  for token in doc :
    print(token,token.pos_,token.pos)



  pass