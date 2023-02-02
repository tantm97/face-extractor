# -*- coding: utf-8 -*-
# Standard library imports
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
from data_processing import dataset
from torch.utils import data


def create_exp_dir(path, visual_folder=False):
    """Create experiment directory.

    Args:
        path: A string determines the path of an experiment directory.
        visual_folder: A boolean indicating if we create visual folder or not. Defaults to False.
    """
    os.makedirs(path, exist_ok=True)
    if visual_folder is True:
        os.mkdir(path + "/visual")  # for visual results

    # print('Experiment dir : {}'.format(path))


def init_seeds(seed):
    """Setup environment for training.

    Args:
        seed: An integer random seeds for reproducibility.
    """
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def prepare_device(gpu_id="0"):
    """Setup GPU device if it is available, move the model into the configured device

    Args:
        gpu_id: An integer selects the GPU's ids. Defaults to '0'.

    Returns:
        A string determines devices and a list of GPUs ids.
    """
    n_gpu = torch.cuda.device_count()
    n_gpu_use = len(gpu_id)

    if n_gpu_use > 0 and (type(gpu_id) == list):
        gpu_id = str(gpu_id).replace("[", "").replace("]", "").replace(" ", "")
    # print(n_gpu_use, gpu_id)
    if n_gpu_use > 0 and n_gpu == 0:
        logging.warning("Warning: There's no GPU available on this machine," "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        logging.warning(
            "Warning: The number of GPU's configured to use is {}, but only {} are available "
            "on this machine.".format(n_gpu_use, n_gpu)
        )
        n_gpu_use = n_gpu

    try:
        # device = torch.device('cuda:' + str(gpu_id) if n_gpu_use > 0 else 'cpu')
        device = torch.device("cuda" if n_gpu_use > 0 else "cpu")
    except RuntimeError as err:
        logging.error(err)
        raise

    list_ids = list(range(n_gpu_use))
    return device, list_ids


def get_dataloaders(t_params):
    """Get dataloader.

    Args:
        t_params: A dictionary of train params.

    Returns:
        train Dataloader and test Dataloader.
    """

    drop_last_train = False
    drop_last_val = False

    train_ds = dataset.Dataset(t_params, t_params["dataset_dir"], t_params["train_list"], phase="train")

    if len(train_ds) - (len(train_ds) // t_params["batch_size"]) * t_params["batch_size"] <= 2:
        drop_last_train = True

    # dataloader bug: https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/4
    train_dl = data.DataLoader(
        train_ds,
        shuffle=True,
        drop_last=drop_last_train,
        batch_size=t_params["batch_size"],
        num_workers=t_params["num_workers"],
    )

    val_ds = dataset.Dataset(t_params, t_params["dataset_dir"], t_params["val_list"], phase="val")

    if len(val_ds) - (len(val_ds) // t_params["batch_size"]) * t_params["batch_size"] <= 2:
        drop_last_val = True

    val_dl = data.DataLoader(
        val_ds, shuffle=False, drop_last=drop_last_val, batch_size=t_params["batch_size"], num_workers=t_params["num_workers"]
    )

    return train_dl, val_dl


def get_time():
    """Get current time.

    Returns:
        A string of Datetime.
    """
    return (str(datetime.now())[:-10]).replace(" ", "-").replace(":", "-")
