from typing import Optional, Callable

import torch
from torch import Tensor
from torch import nn
import torchaudio

import numpy as np

import os
import glob

import random

SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185
SR = 16000

def normalzieNoise(wav: Tensor,
                   noise: Tensor,
                   max_length: int = 16000) -> Tensor:
    len_wav = wav.shape[1]
    len_noise = noise.shape[1]
    if len_wav > len_noise:
        buf = torch.zeros_like(wav)
        start_point = int((len_wav - len_noise)*random.uniform(0, 1))
        end_point = start_point + len_noise
        buf[:, start_point: end_point] = noise
        noise = buf
    elif len_wav < len_noise:
        start_point = int((len_noise - len_wav)*random.uniform(0, 1))
        end_point = start_point + len_wav
        noise = noise[:, start_point: end_point]
    return noise[:, :max_length]

def addNoise(wav: Tensor,
             noise: Tensor,
             snr: list) -> Tensor:
    noise = normalzieNoise(wav, noise)
    addnsy = torchaudio.transforms.AddNoise()
    return addnsy(wav, noise, snr = torch.Tensor([random.uniform(*snr)]))

def pad_truncate(wav: Tensor,
                 max_length: int = 16000,
                 pad_value: int = 0) -> Tensor:
    wav_length = wav.shape[1]
    if wav_length < max_length:
        buff = torch.zeros([1, max_length])
        buff[:, :wav_length] = wav
        wav = buff
    elif wav_length > max_length:
        wav = wav[:, :max_length]
    return wav

def time_shift(wav: Tensor,
               shift: list,
               sr: int = 16000) -> Tensor:
    x_shift = int(random.uniform(*shift)*sr)
    padding = torch.zeros(1, np.abs(x_shift))
    if x_shift < 0:
        wav = torch.cat([padding, wav[:, :x_shift]], dim=-1)
    else:
        wav = torch.cat([wav[:, x_shift:], padding], dim=-1)
    return wav

class Augmentation:
    def __init__(self,
                 noise_dir: str,
                 noise_prob: float,
                 snr: list,
                 shift: list = None,
                 is_train: bool = False,
                 augment: bool = True,
                 transform: Optional[Callable] = None,
                 spec_augment: Optional[bool] = False,
                 *args, **kwargs) -> None:
        self.noise_paths = glob.glob(os.path.join(noise_dir, '*.wav'))
        self.is_train = is_train
        self.noise_prob = noise_prob
        self.augment = augment
        self.transform = transform
        self.add_noise = lambda x, noise: addNoise(x, noise, snr)
        self.pad_trunc = lambda x: pad_truncate(x, SR)
        self.shift = shift
        if shift:
            self.time_shift = lambda x: time_shift(x, shift)
        if spec_augment:
            self.spec_augment = torchaudio.transforms.SpecAugment(*args, **kwargs)
        else:
            self.spec_augment = None

    def __call__(self,
                 wav: Tensor,
                 label_idx: str) -> Tensor:
        # padding to SR
        wav = self.pad_trunc(wav)

        if self.augment:
            # time shifting for training
            if self.is_train:
                if self.shift:
                    wav = self.time_shift(wav)

            p = random.random()
            if label_idx == SILENCE_INDEX or (self.is_train and p<= self.noise_prob):
                noise, _ = torchaudio.load(random.choice(self.noise_paths))
                if label_idx == SILENCE_INDEX:
                    p = random.random()
                    wav = normalzieNoise(wav, noise*p)
                else:
                    wav = self.add_noise(wav, noise)

        if self.transform:
            wav = self.transform(wav)
        
        if self.transform and self.spec_augment:
            wav = self.spec_augment(wav)
        return wav

class LFBE_Delta(nn.Module):
    def __init__(self,
                 sample_rate: int,
                 n_mfcc: int,
                 n_mels: int,
                 melkwargs: dict) -> None:
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(sample_rate = sample_rate,
                                               n_mfcc = n_mfcc,
                                               melkwargs = melkwargs)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate = sample_rate,
                                                        n_mels = n_mels,
                                                        **melkwargs)
        self.todb = torchaudio.transforms.AmplitudeToDB()

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
        input: Tensor input: (N, C, T)
        """
        logmel = self.todb(self.mel(input))
        mfcc = self.mfcc(input)
        delta = torchaudio.functional.compute_deltas(mfcc)
        delta2 = torchaudio.functional.compute_deltas(delta)
        return torch.concat([logmel, delta, delta2], dim = 1)

def transform(transform_type: str, 
              *args, **kwargs):
    if transform_type == 'logmel':
        transform = nn.Sequential(torchaudio.transforms.MelSpectrogram(*args, **kwargs),
                                  torchaudio.transforms.AmplitudeToDB())
    elif transform_type == 'MFCC':
        transform = torchaudio.transforms.MFCC(*args, **kwargs)
    elif transform_type == 'LFBE_Delta':
        transform = LFBE_Delta(*args, **kwargs)
    else:
        raise "The transform type doesn't exist."
    return transform


