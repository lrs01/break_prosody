import os

import IPython
import matplotlib
import matplotlib.pyplot as plt
import requests
import torch
import torchaudio



if __name__=='__main__':

    device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    waveform, sample_rate = torchaudio.load('/disk2/lrs/dengfeng.p/break_feature/0001_000051.wav')
    print(waveform.shape)
    waveform = waveform.to(device)
    with torch.inference_mode():
        features, _ = model.extract_features(waveform)
    tool = []
    for feature in features:
        tool.append(feature)
    tool = tuple(tool)
    finallyFeature = torch.concat(tool,dim=0)
    print(len(features))
    print(features[0].shape)
    print(finallyFeature.shape)
    # print()
    
