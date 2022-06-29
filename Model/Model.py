from importlib import import_module
from unicodedata import bidirectional
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
        # self.model5 = nn.Linear(128,mark_classes)
    def forward(self,input):
        # input = input.permute(0,2,1).contiguous()
        # print(input.shape)
        # exit()
        x = self.model1(input)##[B,n_ mel,帧数]
        x = x.permute(0,2,1).contiguous()
        x = self.model2(x)[0]
        x = self.model3(x)
        # x = self.dropout(x)
        out = self.model4(x).permute(0,2,1).contiguous()
        return out
    ## out : [batchsize, markclass, frames]
    # label = torch.LongTensor(np.random.randint(0,20,(2,200)))

    # lossf =  nn.CrossEntropyLoss()

    # loss =  lossf(out.view(2*200,mark_classes),label.view(2*200))
    # print(loss)
if __name__ == '__main__':

    inp = torch.rand(1,90,11)
    x = model()
    print(x(inp).shape)
    exit()

   
    pass