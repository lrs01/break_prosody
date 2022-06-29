
import librosa
import numpy as np
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
# Audio
# num_mels = 80
# # num_freq = 1024
# n_fft = 2048
# sr = 16000
# # frame_length_ms = 50.
# # frame_shift_ms = 12.5
# preemphasis = 0.97
# frame_shift = 0.0125 # seconds
# frame_length = 0.05 # seconds
# hop_length = int(sr*frame_shift) # samples.
# win_length = int(sr*frame_length) # samples.
# n_mels = 80 # Number of Meƒl banks to generate
# power = 1.2 # Exponent for amplifying the predicted magnitude
# min_level_db = -100
# ref_level_db = 20
# hidden_size = 256
# embedding_size = 512
# max_db = 100
# ref_db = 20


def get_spectrograms(fpath,hp):
  
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.
    Args:
      fpath: A string. The full path of a sound file.
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=16000)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    f0, _ = pw.dio(y.astype(np.float64), hp.sr, frame_period=hp.hop_length / hp.sr * 1000)

    energy = np.linalg.norm(mag, ord=None, axis=None, keepdims=False)

    return mel, mag, f0, energy

# import matplotlib.pyplot as plt


def YIN_F0(filename):
    '''YIN 方法提取基频'''
    y, sr = librosa.load(filename, sr=16000)
    f0 = librosa.yin(y, fmin=80, fmax=400)
    print(f0)
    # f0[np.isnan(f0)] = 0
    # times = librosa.times_like(f0)
    # plt.plot(times, f0, "_", linewidth=1)
    # plt.xlabel("Time(s)")
    # plt.ylabel("F0")

## PYworld 提取F0(注意使用了左声道的数据)
def PYwordl_F0(filename):
  ## fs=16k
  x,fs = sf.read(filename)
  x= x.T[0].astype(np.double)## numpy转置，转换成double类型
  # x,fs = librosa.load(filename,sr = 16000,mono=False)
  # x = x[0].astype(np.double)
  # sf.write('/home/lrs/dengfeng.p/break_feature/left2.wav',x,samplerate=16000)
  _f0, t = pw.dio(x, fs,f0_floor=75,f0_ceil=800,frame_period=10)
  f0 = pw.stonemask(x, _f0, t, fs)
  len_ = len(f0)## 帧数
  len_2 = len(t)
  return f0,t


def pitch(voiceID,f0min,f0max):
  sound = parselmouth.Sound(voiceID) # read the sound
  pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
  return pitch


def get_mel(wav_path, n_fft, n_mels, win_len, hop_len, preemphasis, ref_db, max_db):
  y,sr = librosa.load(wav_path,sr = 16000)
  # y= y[0]
  # print(y.shape)
  # exit()
  # 预加重
  # preemphasis = 0.97
  y = np.append(y[0], y[1:] - preemphasis * y[:-1])
  # stft
  linear = librosa.stft(y=y,n_fft=n_fft,hop_length=hop_len,win_length=win_len)
  # magnitude spectrogram
  mag = np.abs(linear)
  # mel spectrogram
  mel_basis = librosa.filters.mel(sr,n_fft,n_mels)
  mel = np.dot(mel_basis,mag)
  # to decibel
  mel = 20 * np.log10(np.maximum(1e-5, mel))
  mag = 20 * np.log10(np.maximum(1e-5, mag))
  #normalize 
  mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
  # Transpose 
  mel = mel.T.astype(np.float32)
  
  return mel

  pass

if __name__=="__main__":
  """
  使用pyworld提取基频
  """
  hp= hpCram()
  wav_src = '/home/lrs/dengfeng.p/break_feature/jupterT/0001_000051.wav'
  # print(PYwordl_F0(wav_src)[0].shape)
  mel = get_mel(wav_src,hp.n_fft,hp.n_mels,hp.win_length,hp.hop_length,hp.preemphasis,hp.ref_db,hp.max_db)
  print(mel.shape)

    