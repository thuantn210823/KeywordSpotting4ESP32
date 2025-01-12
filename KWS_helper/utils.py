from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassConfusionMatrix

import os
import urllib
import sys
import tarfile
from zipfile import ZipFile
from tqdm import tqdm

from torchaudio._internal import download_url_to_file

# Create object of ZipFile
def zipzip(input_directory, zip_file_name):
    with ZipFile(zip_file_name, 'w') as zip_object:
    # Traverse all files in directory
        for folder_name, sub_folders, file_names in os.walk(input_directory):
            for filename in tqdm(file_names, desc = 'zipping...' ):
                # Create filepath of files in directory
                file_path = os.path.join(folder_name, filename)
                # Add files to zip file
                zip_object.write(file_path, os.path.basename(file_path))

    if os.path.exists(zip_file_name):
        print(f"{zip_file_name} created")
    else:
        print(f"{zip_file_name} created")

def unzipzip(zip_file_name, output_directory):
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    with ZipFile(zip_file_name, 'r') as zip_obj:
        zip_obj.extractall(output_directory)

    print(f"Extracted {zip_file_name}")

def download(data_url, dest_directory):
    filename = os.path.split(data_url)[-1]
    def _progress(count, block_size, total_size):
        sys.stdout.write(
            '\r>> Downloading %s %.1f%%' %
            (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    if not os.path.exists(dest_directory):
        os.mkdir(dest_directory)
    filepath = os.path.join(dest_directory, filename)

    filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def _download(data_url: str, 
              dest_directory: str) -> None:
    """
    Download data files from given data url.

    Args:
    data_url: str
    dest_directory: str
        where your files will be at.
    """
    filename = os.path.split(data_url)[-1]

    if not os.path.exists(dest_directory):
        os.mkdir(dest_directory)
    filepath = os.path.join(dest_directory, filename)

    download_url_to_file(data_url, filepath)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def predict(model, dataset, num_classes):
    test_model = model.eval()
    metric = 0
    for idx, (x, y) in tqdm(enumerate(dataset)):
        confmat = MulticlassConfusionMatrix(num_classes = num_classes)
        pred = test_model(x)
        metric += confmat(pred, y)
    return metric, confmat

def plot_confmat(mat, num_classes, confmat_name, default_dir, save = True):
    fig, ax = plt.subplots(figsize = (5, 5))
    im = ax.matshow(mat)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, mat[i][j].item(),
                  ha="center", va="center", color="w", fontsize = 10)
    ax.figure.colorbar(im, ax = ax)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    fig.savefig(os.path.join(default_dir, confmat_name + '.png'), dpi = 1000)
    return fig, ax

def FRR_FAR(cofmat: Tensor,
            label_index: int):
    sum_ = sum(cofmat[label_index, :])
    res = 1-(cofmat[label_index, label_index])/sum_
    return res.item()*100    