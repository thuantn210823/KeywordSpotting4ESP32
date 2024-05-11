import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchaudio
from torchaudio import datasets, transforms


import numpy as np
import pandas as pd

import os
import glob
from tqdm import tqdm

import random

from GSC12 import SpeechCommands12
from utils import zipzip

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

class Preprocessing:
    def __init__(self,
                 noise_dir: str,
                 noise_prob: float,
                 snr: list,
                 shift: list = None,
                 is_train: bool = False,
                 augment: bool = True,
                 transform = None) -> None:
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

    def __call__(self,
                 wav: Tensor,
                 label: str) -> Tensor:
        # padding to SR
        wav = self.pad_trunc(wav)

        if self.augment:
            # time shifting for training
            if self.is_train:
                if self.shift:
                    wav = self.time_shift(wav)

            p = random.random()
            if label == SILENCE_LABEL or (self.is_train and p<= self.noise_prob):
                noise, _ = torchaudio.load(random.choice(self.noise_paths))
                if label == SILENCE_LABEL:
                    p = random.random()
                    wav = normalzieNoise(wav, noise*p)
                else:
                    wav = self.add_noise(wav, noise)

        if self.transform:
            wav = self.transform(wav)

        return wav


from tqdm import tqdm

def GSC_preprocessing(dataset, output_directory, num_classes = 12, mul_factor = 1, set = 'train', csv_file_name = 'analysised_spec.csv'):
    """
    Preprocessing for each dataset

    mul_factor: increasing the number of data samples by mul_factor times.
    """
    out_df = {
        'link': [],
        'label': [],
    }
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    for idx in range(mul_factor):
        # def f(ix, ex):
        for ix, (wav, label) in tqdm(enumerate(dataset)):
            #if ix%1000 == 0:
            #    print(f'{ix}/{len(data_df)}')
            #row = data_df.iloc[ix]
            #label = row['label']

            fname = f'{set}_{label}_{ix}_{idx}.npz'
            #row = {
            #    'link': os.path.join(output_directory, f'label_{label}', fname),
            #    'label': label,
            #    'set': set
            #}
            out_df['link'].append(os.path.join(set, fname))
            out_df['label'].append(label)

            if os.path.exists(os.path.join(output_directory, fname)):
                continue

            #out_df = out_df._append(row, ignore_index = True)
            np.savez_compressed(os.path.join(output_directory, fname), wav.squeeze(0).numpy())

        # Parallel(n_jobs = os.cpu_count())(delayed(f)(i, ex) for i, ex in tqdm(enumerate(dataset)))

    out_df = pd.DataFrame(out_df)
    out_df.to_csv(csv_file_name, index = False)


def makeGSC12(data_dir: str,
              test_dir: str,
              root: str,
              download: bool,
              augment_configs: dict,
              csv_file_names: dict,
              zip_paths: dict,
              transform):
    
    f_pre = lambda is_train, augment: Preprocessing(noise_dir = '/content/GSC_12/_background_noise_',
                                                noise_prob = augment_configs['noise_prob'],
                                                snr = augment_configs['snr'],
                                                shift  = augment_configs['shift'],
                                                is_train = is_train,
                                                augment = augment,
                                                transform = transform
                                                )
    train_pre = f_pre(True, True)
    val_pre = f_pre(False, True)
    test_pre = f_pre(False, False)

    train_dataset = SpeechCommands12(data_dir, download = download, subset = 'training', transform = train_pre)
    val_dataset = SpeechCommands12(data_dir, download = False, subset = 'validation', transform = val_pre)
    test_dataset = SpeechCommands12(test_dir, download = download, subset = 'official_testing', transform = test_pre)

    if not os.path.exists(root):
        os.mkdir(root)
    
    train_path = os.path.join(root, 'train')
    val_path = os.path.join(root, 'val')
    test_path = os.path.join(root, 'test')

    # Make data
    GSC_preprocessing(train_dataset, train_path, 12, 4, 'train', csv_file_name = csv_file_names['train'])
    GSC_preprocessing(val_dataset, train_path, 12, 1, 'val', csv_file_name = csv_file_names['val'])
    GSC_preprocessing(test_dataset, train_path, 12, 1, 'test', csv_file_name = csv_file_names['test'])

    # Zipzip data
    zipzip(train_path, zip_paths['train'])
    zipzip(val_path, zip_paths['val'])
    zipzip(test_path, zip_paths['test'])

# Old version
def soundDataToFloat(SD):
    "Converts integer representation back into librosa-friendly floats, given a numpy array SD"
    return np.array([ np.float32((s>>2)/(32768.0)) for s in SD])

def GSC_12_preprocessing(dataset, output_directory, transform = None,
                       mul_factor = 1, set = 'train', csv_file_name = 'analysised_spec.csv'):
    """
    Preprocessing for each dataset

    mul_factor: increasing the number of data samples by mul_factor times.
    """
    out_df = pd.DataFrame({
        'link': [],
        'label': [],
        'set': [],
    })
    for idx in range(mul_factor):
        for ix, ex in tqdm(enumerate(dataset), desc = 'Processing...'):
            wav = ex['audio']
            wav = soundDataToFloat(wav.numpy())
            label = ex['label'].numpy()

            wav = torch.from_numpy(wav).unsqueeze(0)

            if transform:
                wav = transform(wav.float())

            fname = f'{set}_{label}_{ix}_{idx}.pt'
            row = {
                'link': os.path.join(output_directory, f'label_{label}', fname),
                'label': label,
                'set': set
            }

            out_df = out_df._append(row, ignore_index = True)
            torch.save(wav, os.path.join(output_directory, f'label_{label}', fname))

    out_df.to_csv(csv_file_name, index = False)

def GSC_35_preprocessing(dataset, output_directory, transform = None, warm_up = 0,
                       mul_factor = 1, set = 'train', csv_file_name = 'analysised_spec.csv'):
    """
    Preprocessing for each dataset

    mul_factor: increasing the number of data samples by mul_factor times.
    """
    out_df = pd.DataFrame({
        'link': [],
        'label': [],
        'set': [],
    })
    for idx in range(mul_factor):
        for ix, (wav, _, label, *_) in tqdm(enumerate(dataset), desc = 'Processing...'):
            #if ix < warm_up:
            #    continue
            label = labels.index(label)
            #wav = torch.from_numpy(wav).unsqueeze(0)

            fname = f'{set}_{label}_{ix}_{idx}.pt'
            row = {
                'link': os.path.join(output_directory, f'label_{label}', fname),
                'label': label,
                'set': set
            }

            out_df = out_df._append(row, ignore_index = True)

            if ix>= warm_up:
                if transform:
                    wav = transform(wav)
                torch.save(wav, os.path.join(output_directory, f'label_{label}', fname))

    out_df.to_csv(csv_file_name, index = False)

if __name__ == '__main__':
    pass

