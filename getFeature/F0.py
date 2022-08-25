
from json import tool
import pickle 
from distutils.log import debug
import librosa
import numpy as np
import statistics as sts ## 计算标准差
from torch import double
from hpCram import hpCram
import soundfile as sf
import pyworld as pw
import warnings
import glob
import pandas as pd
import parselmouth
from parselmouth.praat import call
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
from F0Andenergy import PYwordl_F0
import textgrid

def tgpath_2_wavpath_(tgp:str):
    '''
    :param tg: tg绝对路径
    :return:  其对应的 语音文件相对路径
    '''
    return tgp.replace('lab','wav').replace('.TextGrid','.wav')
def get_pitch(tg_filepath):
    tgfilepath = tg_filepath
    wav_path = tgpath_2_wavpath_(tgfilepath) ## 查找到对应的wav文件的路径
    f0,t_f0 = PYwordl_F0(wav_path) ## 调用方法获取基频，对应基频时间信息
    tg_ = textgrid.TextGrid()
    tg_.read(tgfilepath)
    global PY
    ## 找到拼音层
    for i in range(len(tg_.tiers)):
        if tg_.tiers[i].name == 'PY' or tg_.tiers[i].name == 'py' or tg_.tiers[i].name == 'syllable' or tg_.tiers[i].name == 'sy' or tg_.tiers[i].name == '1':
             PY = i
    
    ## 找到 声韵母层
    for i_DE in range(len(tg_.tiers)):
        if tg_.tiers[i_DE].name == 'DE' or tg_.tiers[i_DE].name == 'de':
            DE = i_DE ## 0
            break
    ## 轻音部分
    Qing = ['b','d','g','p','t','k','z','zh','k','j','c','ch','q','f','s','sh','x','h']
    xmin = 0
    xmax = 0
    xmin_ = 0
    frame_shift = 0.01 ## 帧移10ms
    arr = []
    i = 0
    for j in range(len(tg_.tiers[PY])):
        l = []
		# if tg_.tiers[PY][len(tg_.tiers[PY])-1] != 'sil'
        if(tg_.tiers[PY][j].mark != 'sil' and tg_.tiers[PY][j].mark !='silv' and tg_.tiers[PY][j].mark != ''):
            if (j+1) < len(tg_.tiers[PY]):
                if tg_.tiers[PY][j+1].mark == 'tl':
                    xmin = tg_.tiers[PY][j].minTime
                    xmax = tg_.tiers[PY][j+1].maxTime
                    for i in range(len(tg_.tiers[DE])):
                        if tg_.tiers[DE][i].minTime == xmin and tg_.tiers[DE][i].maxTime!= tg_.tiers[PY][j].maxTime :
                            for k in range(len(Qing)):
                                if Qing[k] in tg_.tiers[DE][i].mark:
                                    xmin = tg_.tiers[DE][i].maxTime
                                    # print('最小时间',tg_.tiers[PY][j].minTime)
                                    # print(tg_.tiers[PY][j].mark)
                                    # print(tg_.tiers[DE][i].mark)
                                    # print('最终最小时间',xmin)
                                    # print('#######################')
                                    break
                        l.append(xmin)
                        l.append(xmax)    
                        break                        
                elif(tg_.tiers[PY][j].mark!='tl'):
                    xmin = tg_.tiers[PY][j].minTime
                    xmax = tg_.tiers[PY][j].maxTime
                    for i in range(len(tg_.tiers[DE])):
                        if tg_.tiers[DE][i].minTime == xmin and tg_.tiers[DE][i].maxTime!= tg_.tiers[PY][j].maxTime:
                            for k in range(len(Qing)):
                                if Qing[k] in tg_.tiers[DE][i].mark:
                                    xmin =tg_.tiers[DE][i].maxTime
                                    # print('最小时间',tg_.tiers[PY][j].minTime)
                                    # print(tg_.tiers[PY][j].mark)
                                    # print(tg_.tiers[DE][i].mark)
                                    # print('最终最小时间',xmin)
                                    # print('#######################')
                                    break
                        l.append(xmin)
                        l.append(xmax)      
                        break
            elif tg_.tiers[PY][j]!='tl':
                xmin = tg_.tiers[PY][j].minTime
                xmax = tg_.tiers[PY][j].maxTime
                for i in range(len(tg_.tiers[DE])):
                        if tg_.tiers[DE][i].minTime == xmin and tg_.tiers[DE][i].maxTime!= tg_.tiers[PY][j].maxTime:
                            for k in range(len(Qing)):
                                if Qing[k] in tg_.tiers[DE][i].mark:
                                    xmin =tg_.tiers[DE][i].maxTime
                        l.append(xmin)
                        l.append(xmax)
                        break
        if len(l)!=0:
            arr.append(l)
    arr = np.array(arr)
    final_f0 = []
    for i in range(arr.shape[0]):
        frame_s= []
        min_t = arr[i][0] ## 当前音节的开始时间
        max_t = arr[i][1] ## 当前音节的结束时间
        for i_ in range(t_f0.size):
            if t_f0[i_] > min_t and t_f0[i_] < max_t:
                frame_s.append(f0[i_])
        if frame_s == []:
            print(arr[i])
            print(i)
        
        ## 计算参数
        max_f = max(frame_s) ## 当前音节内的f0最大的值
        min_f = min(frame_s) ## 当前音节内的f0最小的值
        mean_f = np.mean(frame_s) ## 当前音节内f0的均值
        start_f = sum(frame_s[:10])/10 ## 当前音节首位f0值
        end_f = sum(frame_s[-10:])/10 ## 当前音节末尾f0值
        std_f = np.std(frame_s) ## 当前音节f0 标准差
        var_f = np.var(frame_s) ## 当前音节f0 方差

        #——————————————————————下一个音节的内的相关特征————————————————————————————————————————————#
        if i!= arr.shape[0]-1 : # 如果不是最后一个音节
            min_t_after = arr[i+1][0] # 下一个音节开始的时间
            max_t_after = arr[i+1][1] # 下一个音节结束的时间
            tool_after_pitch = []
            [tool_after_pitch.append(f0[i_after]) for i_after in range(t_f0.size) if t_f0[i_after] > min_t_after and t_f0[i_after] < max_t_after]

            max_range = max_f - max(tool_after_pitch)
            min_range = min_f - min(tool_after_pitch)
            mean_range = mean_f - np.mean(tool_after_pitch)
            start_end_range = end_f - sum(tool_after_pitch[:10]) / 10

        else :
            max_range = 0
            min_range = 0
            mean_range = 0
            start_end_range = 0
            
        ## f0特征
        f0_feature = [min_f,max_f,start_f,end_f,mean_f,std_f,var_f,max_range,min_range,mean_range,start_end_range]
        final_f0.append(f0_feature)
    final_f0 = np.array(final_f0)
    return final_f0 
    
def get_duration(tg_filepath):
    '''
    获取当前音节的duration
    音节之后duration时长
    当前音节的duration 和后面音节duration 的比值
    当前音节duration 和 后面duration 的比值
    *******暂时不考虑当前音节和前后音节的duration 的比值*******
    '''
    tg_filepath = tg_filepath
    tg_ = textgrid.TextGrid()
    tg_.read(tg_filepath)

    ## 找到拼音层
    for i in range(len(tg_.tiers)):
        if tg_.tiers[i].name == 'PY' or tg_.tiers[i].name == 'py' or tg_.tiers[i].name == 'syllable' or tg_.tiers[i].name == '1' or tg_.tiers[i].name == 'sy':
            PY = i
    
    xmin = 0
    xmax = 0
    xmin_ = 0
    frame_shift = 0.01 ## 帧移10ms
    arr = []
    i = 0
    for j in range(len(tg_.tiers[PY])):
        l = []
        
        if(tg_.tiers[PY][j].mark != 'sil' and tg_.tiers[PY][j].mark !='silv' and tg_.tiers[PY][j].mark!=''):
            if j+1 <len(tg_.tiers[PY]) :
                if(tg_.tiers[PY][j+1].mark == 'tl'):
                    xmin = tg_.tiers[PY][j].minTime
                    xmax = tg_.tiers[PY][j+1].maxTime
                    l.append(xmin)
                    l.append(xmax)    
                                        
                elif(tg_.tiers[PY][j].mark!='tl'):
                    xmin = tg_.tiers[PY][j].minTime
                    xmax = tg_.tiers[PY][j].maxTime
                    l.append(xmin)
                    l.append(xmax)   
            elif tg_.tiers[PY][j]!='tl':
                xmin = tg_.tiers[PY][j].minTime
                xmax = tg_.tiers[PY][j].maxTime
                l.append(xmin)
                l.append(xmax)
          
                
        if len(l)!=0:
            arr.append(l)
    arr = np.array(arr)
    
    #########################################
    arr_duration=[]
    for i in range(arr.shape[0]):
        arr_tool  = []
        min_t = arr[i][0] ## 当前音节的开始时间
        max_t = arr[i][1] ## 当前音节的结束时间
        s_d = max_t - min_t ## 获取当前音节duration
        arr_tool.append(s_d)
        #——————————————————## 获取当前音节之后的静音段的时长和当前音节和其后音节的时长比值——————————————————————————————————————————————————————#
        if i == arr.shape[0]-1: 
            len_PY = len(tg_.tiers[PY])
            sil_f = tg_.tiers[PY][len_PY-1].maxTime - tg_.tiers[PY][len_PY-1].minTime
            ## 如果是最后一个音节，比值为0
            duration_rate = 0
        else : 
            sil_f = arr[i+1][0] - arr[i][1]
            ## 如果不是最后一个音节，比值正常计算
            duration_rate = s_d / (arr[i+1][1] - arr[i+1][0])
        #——————————————————## 获取当前音节之后的静音段的时长和当前音节和其后音节的时长比值——————————————————————————————————————————————————————#
        arr_tool.append(sil_f)
        arr_tool.append(duration_rate)
        arr_duration.append(arr_tool)

    arr_duration = np.array(arr_duration)



    return  arr_duration
    pass


def get_intensity(tg_filepath):
    # 传入TextGrid文件
    tg_filepath1 = tg_filepath
    # 找到对应的wav文件
    wav_file = tgpath_2_wavpath_(tg_filepath1)
    y,sr = librosa.load(wav_file,sr = 16000, mono=False)
    # 取左声道
    y = y[0]
    intensity = librosa.feature.rms(y=y,hop_length= 160)
    intensity_ = intensity[0] 
    ## 得到每一帧的时间点
    _,t_f0 = PYwordl_F0(wav_file) 
    tg_ = textgrid.TextGrid()
    tg_.read(tg_filepath1)

    for i in range(len(tg_.tiers)):
        if tg_.tiers[i].name == 'PY' or tg_.tiers[i].name == 'py' or tg_.tiers[i].name == 'syllable' or tg_.tiers[i].name == '1' or tg_.tiers[i].name == 'sy':
            PY = i
    
    xmin = 0
    xmax = 0
    xmin_ = 0
    frame_shift = 0.01 ## 帧移10ms
    arr = []
    i = 0
    for j in range(len(tg_.tiers[PY])):
        l = []
        
        if(tg_.tiers[PY][j].mark != 'sil' and tg_.tiers[PY][j].mark !='silv' and tg_.tiers[PY][j].mark!=''):
            if j+1 <len(tg_.tiers[PY]) :
                if(tg_.tiers[PY][j+1].mark == 'tl'):
                    xmin = tg_.tiers[PY][j].minTime
                    xmax = tg_.tiers[PY][j+1].maxTime
                    l.append(xmin)
                    l.append(xmax)    
                                        
                elif(tg_.tiers[PY][j].mark!='tl'):
                    xmin = tg_.tiers[PY][j].minTime
                    xmax = tg_.tiers[PY][j].maxTime
                    l.append(xmin)
                    l.append(xmax)   
            elif tg_.tiers[PY][j]!='tl':
                xmin = tg_.tiers[PY][j].minTime
                xmax = tg_.tiers[PY][j].maxTime
                l.append(xmin)
                l.append(xmax)
          
                
        if len(l)!=0:
            arr.append(l)
    arr = np.array(arr)
    ################################
    arr_intensity = []
    for i in range(arr.shape[0]):
        arr_tool = []
        min_t = arr[i][0]
        max_t = arr[i][1]
        for i_ in range(t_f0.size):
            if t_f0[i_] > min_t and t_f0[i_]<max_t:
                arr_tool.append(intensity_[i_])
        # ——————————————————抽取当前音节的特征————————————————————————————————————————————#
        max_intensity = max(arr_tool) # 最大值
        min_intensity = min(arr_tool)# 最小值
        mean_intensity = np.mean(arr_tool) # 平均值
        arange_intensity = max_intensity - min_intensity # 范围
        # ——————————————————抽取后一个音节的平均intensity的平均值特征————————————————————————————————————————————#
        if i != arr.shape[0]-1: ## 当前音节不是最后一个音节
            arr_tool_after_i = []
            min_t_after_i = arr[i+1][0]
            max_t_after_i = arr[i+1][1]
            for i_after in range(t_f0.size):
                if t_f0[i_after] > min_t_after_i and t_f0[i_after] < max_t_after_i:
                    arr_tool_after_i.append(intensity_[i_after])
            mean_intensity_after = np.mean(arr_tool_after_i)
            one_after_rate = mean_intensity / mean_intensity_after
        else :
            one_after_rate = 0
        # ——————————————————将一个音节的特征组合起来————————————————————————————————————————————#
        intensity_f = [min_intensity,max_intensity,mean_intensity,arange_intensity,one_after_rate]
        arr_intensity.append(intensity_f)
    #———————————————一个样本的（wav文件的的特征）———————————————————————————————————————#
    arr_intensity = np.array(arr_intensity)
    return arr_intensity


def get_label(tg_filepath):

    tg_ = textgrid.TextGrid()
    tg_.read(tg_filepath)

    for i in range(len(tg_.tiers)):
        if tg_.tiers[i].name == "BI" or tg_.tiers[i].name == "bi" or tg_.tiers[i].name == "St":
            BI = i
    
    arr_BI = []
    for j in range(len(tg_.tiers[BI])): ## 将韵律边界标注的time，mark 放到数组中
        arr_tool = []
        time = tg_.tiers[BI][j].time
        mark = tg_.tiers[BI][j].mark
        arr_tool.append(time)
        arr_tool.append(mark)
        arr_BI.append(arr_tool)
    arr_BI = np.array(arr_BI)
    ## 找到拼音层
    for i in range(len(tg_.tiers)):
        if tg_.tiers[i].name == 'PY' or tg_.tiers[i].name == 'py' or tg_.tiers[i].name == 'syllable' or tg_.tiers[i].name == '1' or tg_.tiers[i].name == 'sy':
            PY = i
    
    xmin = 0
    xmax = 0
    xmin_ = 0
    arr = []
    i = 0
    for j in range(len(tg_.tiers[PY])):
        l = []
        
        if(tg_.tiers[PY][j].mark != 'sil' and tg_.tiers[PY][j].mark !='silv' and tg_.tiers[PY][j].mark!=''):
            if j+1 <len(tg_.tiers[PY]) :
                if(tg_.tiers[PY][j+1].mark == 'tl'):
                    xmin = tg_.tiers[PY][j].minTime
                    xmax = tg_.tiers[PY][j+1].maxTime
                    l.append(xmin)
                    l.append(xmax)    
                                        
                elif(tg_.tiers[PY][j].mark!='tl'):
                    xmin = tg_.tiers[PY][j].minTime
                    xmax = tg_.tiers[PY][j].maxTime
                    l.append(xmin)
                    l.append(xmax)   
            elif tg_.tiers[PY][j]!='tl' : 
                xmin = tg_.tiers[PY][j].minTime
                xmax = tg_.tiers[PY][j].maxTime
                l.append(xmin)
                l.append(xmax)
          
                
        if len(l)!=0:
            arr.append(l)
    arr = np.array(arr)

    arr_mark=[]
    for i in range(arr.shape[0]):
        arr_tool = []
        max_t = arr[i][1]
        # print(max_t)
        sign = 0
        for j in range(arr_BI.shape[0]):
            # print(arr_BI[j][0])
            # exit()
            if  float(arr_BI[j][0])  == max_t:
                arr_mark.append(int(arr_BI[j][1]))
                sign =1
                break
        if sign!=1:
            arr_mark.append(0)
        
    arr_mark = np.array(arr_mark)
    return arr_mark


def get_mel_syllable(tg_filepath):
    

    pass










if __name__=="__main__":
    ## 测试
    path1 = '/disk2/lrs/dengfeng.p/break_feature/Data/lab/f001lab/f001001_02.TextGrid'
    # path2 = '/home/lrs/dengfeng.p/break_feature/Data/lab/f002lab/f002001_01.TextGrid'
    
    arr_mark = get_label(path1)
    print(arr_mark) 
    
    # print(arr_BI.shape,arr.shape)
    # print(arr)
    # print(arr_BI)
    # print(arr_BI.shape[0]) 
    # tg_.read(path1)

    # ## 找到拼音层
    # for i in range(len(tg_.tiers)):
    #     if tg_.tiers[i].name == 'PY' or tg_.tiers[i].name == 'py':
    #         PY = i
    
    # for i in range(len(tg_.tiers[PY])):
    #     if tg_.tiers[PY][i].mark == '':
    #         print(tg_.tiers[PY][i].minTime)

    ndarr = get_pitch(path1)
    print(ndarr.shape)
    # # x = get_duration(path1)
    # print(x.shape)
    
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    a = get_intensity(path1)
    print(a.shape)

    # pkl = '/home/lrs/dengfeng.p/break_feature/Data/feature/m002lab/m002013_03/f0.pkl'
    # data = open(pkl,'rb')
    # readdata = pickle.load(data)发给
    # print(readdata.shape)
    # data.close()




    



                                







# print(tg_.tiers[PY][1].bounds())## (minTime,maxTime)
# print(tg_.tiers[PY][1].duration())## duration time
# print(tg_.tiers[PY][1].mark)## 'sil','qi',标注的内容
# print(tg_.tiers[PY][1].maxTime)
# print(tg_.tiers[PY].name) ## 层级标注是什么
# print(tg_.tiers[PY])
# print(tg_.tiers[PY][])

