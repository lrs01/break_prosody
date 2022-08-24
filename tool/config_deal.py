from unicodedata import name
import yaml

def read_yaml(path):
    '''
    输入：.yaml 文件；输入字典
    '''
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)
    file.close()

    return dict


if __name__ == "__main__":

    print(read_yaml(r"/disk2/lrs/dengfeng.p/break_feature/test_/config.yaml"))

    pass