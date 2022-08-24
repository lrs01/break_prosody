import math
from unicodedata import bidirectional 
import torch 
import torch.nn as nn
#导入常用库
import math
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


class LSTM_Attention(nn.Module):
    def __init__(self,feature_dim,hidden_dim,attention_dim,n_layers,num_class):
        super(LSTM_Attention,self).__init__()
        
        #从LSTM得到output之后，将output通过下面的linear层，然后就得到了Q,K,V
        #这里我是用的attention_size是等于hidden_dim的，这里可以自己换成别的attention_size
    
        self.rnn = nn.LSTM(input_size = feature_dim,hidden_size = hidden_dim,num_layers = n_layers,batch_first = True,bidirectional=True)
        #Linear层,因为是三分类，所以后面的维度为3
        self.fc1 = nn.Linear(attention_dim,attention_dim // 2)
        self.fc2 = nn.Linear(attention_dim// 2 ,num_class)
        self.att = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=16)
        self.relu = nn.ReLU()
        #dropout
        self.dropout = nn.Dropout(0.2)
        
    #用来计算attention
   

    def forward(self,x):   
        #进行LSTM
        x = x.permute(0,2,1)
        output,(h_n,c) = self.rnn(x) 
        # print(output.shape)
        attn_output, _ = self.att(output, output, output)
        # print(attn_output.shape)
        out = self.fc1(attn_output) #out.shape = [batch_size,num_class]
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


    
if __name__ == '__main__' :
        
        net = LSTM_Attention(feature_dim=19,hidden_dim=512,attention_dim = 1024,n_layers = 1,num_class=5)
        inp = torch.rand(5,19,2015)
        out = net(inp).shape
        print(out)
        pass

