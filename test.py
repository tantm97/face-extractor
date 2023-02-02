# -*- coding: utf-8 -*-
# Standard library imports
import argparse
import os
import sys

import torch
import torch.multiprocessing
import yaml
from torch.utils import data

from backbone import backbone
from data_processing import dataset
from utils import evaluate, utils


def get_test_dataloader(test_params):
    drop_last_test = False

    test_ds = dataset.Dataset(test_params, test_params["dataset_dir"], test_params["test_list"], phase="test")

    if len(test_ds) - (len(test_ds) // test_params["batch_size"]) * test_params["batch_size"] <= 2:
        drop_last_test = True

    test_dl = data.DataLoader(
        test_ds,
        shuffle=False,
        drop_last=drop_last_test,
        batch_size=test_params["batch_size"],
        num_workers=test_params["num_workers"],
    )

    return test_dl


def parse_args():
    """Arguments for testing.

    Returns:
        Arguments.
    """
    parser = argparse.ArgumentParser(description="Testing Arguments")
    parser.add_argument("--config", default="configs/base.yml", type=str, help="configs file path (default: None)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Debug runtime error dataloader: https://github.com/pytorch/pytorch/issues/11201
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parse_args()

    assert args.config is not None, "Please specify a configs file"

    nn_config_path = args.config
    with open(nn_config_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    device, n_gpu_ids = utils.prepare_device(cfg["gpu_id"])

    model = backbone.get_backbone(cfg["model_params"])
    model = model.to(device)

    test_params = cfg["test_params"]
    checkpoint = torch.load(test_params["weights"], map_location=device)
    model.load_state_dict(checkpoint)

    test_dl = get_test_dataloader(test_params)

    save_visual_dir = "%s/%s_%s.csv" % (test_params["test_dir"], test_params["weights"].rsplit("/", 2)[-2], utils.get_time())
    os.makedirs(save_visual_dir.rsplit("/", 1)[0], exist_ok=True)

    model.eval()
    test_acc = evaluate.evaluate(model, test_dl, device, save_visual_dir)

    print("Accuracy = ", test_acc)
