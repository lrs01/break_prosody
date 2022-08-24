from heapq import merge
from importlib import import_module
from unicodedata import bidirectional
from pandas import merge_asof
import torch 
import torch.nn as nn 
# from final_Process_Utils import get_feature_label_from_wavfile
## 模型测试
class model(nn.Module):
    def __init__(self) :
        super(model,self).__init__()
        mark_classes = 5
        # inp = torch.rand(3,80,649) ## 模型输入size=[batchsize ,n_fft,frams]
        self.model1 = nn.Sequential(
            nn.Conv1d(19,128,kernel_size=3,padding=1),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(128),   
        )
        self.model2 = nn.LSTM(128,512,num_layers =2,batch_first=True,bidirectional=True)
        self.model3 = nn.Linear(1024,256)
        # self.dropout = nn.Dropout(0.2)
        self.model4 = nn.Linear(256,mark_classes)
        # self.fc = nn.Linear() 
        # self.model5 = nn.Linear(128,mark_classes)
    def use_attention (self,lstm_output, hidden_state):
        lstm_output = lstm_output.permute(1,0,2)
        merge_state = torch.cat([s for s in hidden_state],1)
        merge_state = merge_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output,merge_state)
        weights = nn.Softmax(weights.squeeze(2),dim=1).squeeze(2)
        
        return torch.bmm(torch.transpose(lstm_output,1,2),weights).squeeze(2)

    def forward(self,input):
        # input = input.permute(0,2,1).contiguous()
        # print(input.shape)
        # exit()
        x = self.model1(input)##[B,n_ mel,帧数]
        x = x.permute(0,2,1).contiguous()
        # x = self.model2(x)[0]
        #——————————————————————————————————————#
        x,(hidden,ceil) = self.model2(x) # 
        att_output = self.use_attention(x,hidden)
         #——————————————————————————————————————#
        return att_output

if __name__ == '__main__':

    inp = torch.rand(1,19,200)
    x = model()
    print(x(inp).shape)
    exit()

   
    pass