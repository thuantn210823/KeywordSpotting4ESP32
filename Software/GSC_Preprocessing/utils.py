import os
import urllib
import sys
import tarfile

from zipfile import ZipFile
from tqdm import tqdm

from torchaudio._internal import download_url_to_file

#import torch
#import tensorflow as tf

#import onnx
#from onnx_tf.backend import prepare

#from scc4onnx import order_conversion

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

#def pytorch2onnx(pytorch_model, onnx_path):
    