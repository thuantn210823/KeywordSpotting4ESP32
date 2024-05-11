#######################################################################################################
# Google Speech Commands dataset
# 2 GCS versions are available include GSCv2 12 labels from Tensorflow and GSCv2 35 labels from Pytorch
# Author: Thuan Tran Ngoc
# Date: 15th, March 2024
# Version: 0.0
#######################################################################################################

import torch
from torch import nn
import torchaudio
import numpy as np

import os
import re
import gdown
import pandas as pd

from tqdm import tqdm

from utils import zipzip, unzipzip

TRAIN_TRANSFORM = None

class GSC(torch.utils.data.Dataset):
    def __init__(self, root, subset = 'train', zip_map = None, csv_map = None, unzip = True):
        super().__init__()
        local_path = os.path.join(root, subset)
        self.root = root
        if not os.path.exists(root):
            os.mkdir(root)
        if not os.path.exists(local_path):
            os.mkdir(local_path)
            unzipzip(zip_map[subset], local_path)
        if unzip:
            unzipzip(zip_map[subset], local_path)
        self.csv = pd.read_csv(csv_map[subset])
        self.subset = subset

    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        spec = np.load(os.path.join(self.root, row['link']))['arr_0']
        spec = torch.from_numpy(spec).unsqueeze(0)
        if self.subset == 'train':
            if TRAIN_TRANSFORM:
                spec = TRAIN_TRANSFORM(spec)
        return spec, row['label']

    def __len__(self):
        return len(self.csv)


def get_id(drive_url: str) -> str:
    return re.split('/', drive_url)[5]

def download_GSC(train_url: str,
                 val_url: str,
                 test_url: str,
                 output_directory: str,
                 end: str) -> dict:
    train_id = get_id(train_url)
    val_id = get_id(val_url)
    test_id = get_id(test_url)

    gdown_train = f"https://drive.google.com/uc?id={train_id}"
    gdown_val = f"https://drive.google.com/uc?id={val_id}"
    gdown_test = f"https://drive.google.com/uc?id={test_id}"
    
    map = {}

    for subset, url in zip(['train', 'val', 'test'], 
                            [gdown_train, gdown_val, gdown_test]):
        path = os.path.join(output_directory, subset) + end
        gdown.download(url, path, quiet = False)
        map[subset] = path
    
    return map
    
    


