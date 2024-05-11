import torch
from torch import Tensor
from torchmetrics.classification import MulticlassConfusionMatrix

import numpy as np
import matplotlib.pyplot as plt

import os
from tqdm import tqdm

def test(trainer, model, dataloaders, ckpt_directory):
    ckpt_paths = os.listdir(ckpt_directory)
    loss = []
    acc = []
    for ckpt_path in ckpt_paths:
        logger = trainer.test(model = model,
                              dataloaders = dataloaders,
                              ckpt_path = os.path.join(ckpt_directory, ckpt_path),
                              verbose = False)
        loss.append(logger[0]['test_loss'])
        acc.append(logger[0]['test_acc'])
    return loss, acc, sum(loss)/len(loss), sum(acc)/len(acc)

def predict_directory(model, dataloader, num_classes, ckpt_directory):
    ckpt_paths = os.listdir(ckpt_directory)
    metrics = []
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(os.path.join(ckpt_directory, ckpt_path))
        model.load_state_dict(ckpt['state_dict'])
        test_model = model.eval()
        metric = 0
        for idx, (x, y) in tqdm(enumerate(dataloader)):
            confmat = MulticlassConfusionMatrix(num_classes = num_classes)
            #if idx%10 == 0:
            #    print(f'{idx}/{len(dataloader)}')
            pred = test_model(x)
            metric += confmat(pred, y)
        metrics.append(metric)
    return metrics

def predict(model, dataloader, num_classes, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    test_model = model.eval()
    metric = 0
    for idx, (x, y) in tqdm(enumerate(dataloader)):
        confmat = MulticlassConfusionMatrix(num_classes = num_classes)
        #if idx%10 == 0:
        #    print(f'{idx}/{len(dataloader)}')
        pred = test_model(x)
        metric += confmat(pred, y)
    return metric

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