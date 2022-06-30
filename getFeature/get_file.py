import os
import shutil
import pickle
from F0 import get_duration,get_pitch,tgpath_2_wavpath_,get_intensity,get_label
def t1():
    '''
    ## 创建 '/home/lrs/dengfeng.p/break_feature/Data/feature' 下文件夹
    '''
    tgfile_path = r'/disk2/lrs/dengfeng.p/break_feature/Data/lab'
    feature_path = r'/disk2/lrs/dengfeng.p/break_feature/Data/feature'
    tg_list = os.listdir(tgfile_path) ## 列出 tgfile_path 下的所有文件夹
    tg_list.sort() ## 排序
    # for i in tg_list:
    #     path = feature_path + '/' +i
    #     os.mkdir(path)
    c = 0 ## 计数功能
    for i in tg_list:
        tg = os.listdir(tgfile_path+'/'+i)
        tg.sort() ## 排序
        for j in tg:
            j_ = j.replace('.TextGrid', '') ## 将j 中的'.TextGrid' 替换为空字符
            tg_path = feature_path + '/' + i + '/' + j_ ## 组合路径
            tg_file_path = tgfile_path + '/' + i + '/' + j ## 
            ###
            print(tg_file_path)
            print("@@@@@@@@@@@@@@@@@@@@@@@$$$$$$$$$$$$$$$$$$$")
            f0_feature =get_pitch(tg_filepath=tg_file_path) ## 获取f0参数（数组）
            f0_feature_in =open(tg_path+'/'+'f0.pkl','wb') ## 创建新的pickle文件
            print(tg_path)
            pickle.dump(f0_feature,f0_feature_in) ## 将其写入新创建的pickle文件中
            f0_feature_in.close() ## 关闭文件
            
            ###
            duration_feature = get_intensity(tg_filepath=tg_file_path)
            d_feature_in =open(tg_path+'/'+'duration.pkl','wb')
            pickle.dump(duration_feature,d_feature_in)
            d_feature_in.close()

            
            

            c+=1
    print(c)
            # print(tg_path)
            # shutil.rmtree(tg_path) ## 删除指定目录下的所有文件夹
            # os.mkdir(tg_path) ## 创建路径为：tg_path 的文件夹

def t2():
    '''
    ## 创建 '/home/lrs/dengfeng.p/break_feature/Data/feature' 下文件夹
    '''
    tgfile_path = r'/disk2/lrs/dengfeng.p/break_feature/Data/lab'
    feature_path = r'/disk2/lrs/dengfeng.p/break_feature/Data/feature'
    tg_list = os.listdir(tgfile_path) ## 列出 tgfile_path 下的所有文件夹
    tg_list.sort() ## 排序
    # for i in tg_list:
    #     path = feature_path + '/' +i
    #     os.mkdir(path)
    c = 0 ## 计数功能
    for i in tg_list:
        tg = os.listdir(tgfile_path+'/'+i)
        tg.sort() ## 排序
        for j in tg:
            j_ = j.replace('.TextGrid', '') ## 将j 中的'.TextGrid' 替换为空字符
            tg_path = feature_path + '/' + i + '/' + j_ ## 组合路径
            tg_file_path = tgfile_path + '/' + i + '/' + j ## 
            ###
            print(tg_file_path)
            print("@@@@@@@@@@@@@@@@@@@@@@@$$$$$$$$$$$$$$$$$$$")
            intensity_feature =get_duration(tg_filepath=tg_file_path) ## 获取f0参数（数组）
            intensity_feature_in =open(tg_path+'/'+'intensity.pkl','wb') ## 创建新的pickle文件
            print(tg_path)
            pickle.dump(intensity_feature,intensity_feature_in) ## 将其写入新创建的pickle文件中
            intensity_feature_in.close() ## 关闭文件

            c+=1
    print(c)

def t3():
    '''
    ## 创建 '/home/lrs/dengfeng.p/break_feature/Data/feature' 下文件夹
    '''
    tgfile_path = r'/disk2/lrs/dengfeng.p/break_feature/Data/lab'
    feature_path = r'/disk2/lrs/dengfeng.p/break_feature/Data/feature'
    tg_list = os.listdir(tgfile_path) ## 列出 tgfile_path 下的所有文件夹
    tg_list.sort() ## 排序
    # for i in tg_list:
    #     path = feature_path + '/' +i
    #     os.mkdir(path)
    c = 0 ## 计数功能
    for i in tg_list:
        tg = os.listdir(tgfile_path+'/'+i)
        tg.sort() ## 排序
        for j in tg:
            j_ = j.replace('.TextGrid', '') ## 将j 中的'.TextGrid' 替换为空字符
            tg_path = feature_path + '/' + i + '/' + j_ ## 组合路径
            tg_file_path = tgfile_path + '/' + i + '/' + j ## 
            ###
            print(tg_file_path)
            print("@@@@@@@@@@@@@@@@@@@@@@@$$$$$$$$$$$$$$$$$$$")
            label_feature =get_label(tg_filepath=tg_file_path) ## 获取f0参数（数组）
            label_feature_in =open(tg_path+'/'+'label.pkl','wb') ## 创建新的pickle文件
            print(tg_path)
            pickle.dump(label_feature,label_feature_in) ## 将其写入新创建的pickle文件中
            label_feature_in.close() ## 关闭文件

            c+=1
    print(c)



if __name__ == "__main__":
    # t1()
    t2()
    # t3()


    pass
