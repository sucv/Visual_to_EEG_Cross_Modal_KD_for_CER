import os
import shutil
from pathlib import Path
import pickle

import numpy as np
import torch


def load_npy(path, feature=None):
    filename = path
    if feature is not None:
        filename = os.path.join(path, feature + ".npy")

    data = np.load(filename, mmap_mode='c')
    return data


def load_pickle(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data


def save_to_pickle(path, data, replace=False):
    if replace:
        with open(path, 'wb') as handle:
            pickle.dump(data, handle)
    else:
        if not os.path.isfile(path):
            with open(path, 'wb') as handle:
                pickle.dump(data, handle)


def copy_file(input_filename, output_filename):
    if not os.path.isfile(output_filename):
        shutil.copy(input_filename, output_filename)


def expand_index_by_multiplier(index, multiplier):
    expanded_index = []
    for value in index:
        expanded_value = [i for i in np.arange(value * multiplier, (value + 1) * multiplier)]
        expanded_index.extend(expanded_value)
    # expanded_index = np.round((np.asarray(index) * multiplier))
    return list(expanded_index)


def get_filename_from_a_folder_given_extension(folder, extension, string=""):
    file_list = []
    for file in sorted(os.listdir(folder)):
        if file.endswith(extension):
            if string in file:
                file_list.append(os.path.join(folder, file))

    return file_list


def ensure_dir(file_path):
    directory = file_path
    if file_path[-3] == "." or file_path[-4] == ".":
        directory = os.path.dirname(file_path)
    Path(directory).mkdir(parents=True, exist_ok=True)


class CCCLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gold, pred, weights=None):

        gold_mean = torch.mean(gold, 1, keepdim=True, out=None)
        pred_mean = torch.mean(pred, 1, keepdim=True, out=None)
        covariance = (gold - gold_mean) * (pred - pred_mean)
        gold_var = torch.var(gold, 1, keepdim=True, unbiased=True, out=None)
        pred_var = torch.var(pred, 1, keepdim=True, unbiased=True, out=None)
        ccc = 2. * covariance / (
                (gold_var + pred_var + torch.mul(gold_mean - pred_mean, gold_mean - pred_mean)) + 1e-50)
        ccc_loss = 1. - ccc

        if weights is not None:
            ccc_loss *= weights

        return torch.mean(ccc_loss)


