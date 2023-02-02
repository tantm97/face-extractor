# -*- coding: utf-8 -*-
# Standard library imports
import os

import cv2
import numpy as np
import torchvision
import yaml
from torch.utils import data
from torchvision import transforms as T


def compute(img, min_percentile, max_percentile):
    """Calculate the quantile, the purpose is to remove the abnormal situation at both ends of the histogram.

    Args:
        img: An image needs to calculate the quantile.
        min_percentile: An integer indicates the value of a min percentile.
        max_percentile: An integer indicates the value of a max percentile.

    Returns:
        An integer indicates values of max percentile pixel and min percentile pixel.
    """

    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)

    return max_percentile_pixel, min_percentile_pixel


def enhance_brightness(src):
    """Image brightness enhancement. More detail in: https://programmersought.com/article/26004081155/.

    Args:
        src:  An image needs to enhance brightness.

    Returns:
        An image is enhanced brightness.
    """

    # if get_lightness(src)>130:
    #     print("The brightness of the picture is sufficient, no enhancement")
    #     return src

    # Calculate the quantile first and remove a few outliers in the pixel value. This quantile can be configured by yourself.
    # For example, the red color of the histogram in 1 has a value from 0 to 255, but in fact the pixel value is mainly within 0 to 20.

    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)

    # Remove values outside the quantile range

    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel

    # Stretch the quantile range from 0 to 255. 255*0.1 and 255*0.9 are taken here
    # because pixel values may overflow, so it is best not to set it to 0 to 255.

    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)

    return out


def get_lightness(src):
    """Calculate brightness.

    Args:
        src: An image needs to calculate brightness.

    Returns:
        A float indicates a brightness value.
    """

    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()

    return lightness


class Dataset(data.Dataset):
    """Override Dataset class.

    Attributes:
        phase: A string indicating a phase is training, validating, or testing.
        input_shape: A tuple of the input shape.
        imgs: A list containing the image paths and ground-truth.
        transforms: Image transformations.
    """

    def __init__(self, t_params, root, data_list_file, phase="train"):
        """Init Dataset class."""
        self.phase = phase
        self.input_shape = tuple(t_params["input_shape"])

        with open(os.path.join(data_list_file), "r") as fd:
            imgs = fd.readlines()

        imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs)

        normalize = T.Normalize(mean=t_params["rgb_mean"], std=t_params["rgb_std"])

        if self.phase == "train":
            self.transforms = T.Compose(
                [
                    T.ToPILImage(),
                    T.Resize(self.input_shape[1:]),
                    # T.RandomCrop(self.input_shape[1:]),
                    T.RandomHorizontalFlip(p=0.5),
                    # T.RandomApply([T.ColorJitter(brightness=(0.4), contrast=(0.55, 1.0))], p=0.4),
                    # T.Grayscale(),
                    T.ToTensor(),
                    normalize,
                ]
            )
        else:
            self.transforms = T.Compose([T.ToPILImage(), T.Resize(self.input_shape[1:]), T.ToTensor(), normalize])

    def __getitem__(self, index):
        """Get sample."""
        sample = self.imgs[index]
        splits = sample.split()
        # if len(splits) < 2:
        #     print(splits)
        img_path = splits[0]
        image = cv2.imread(img_path)
        image = enhance_brightness(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image)
        label = int(splits[1])
        return image.float(), label

    def __len__(self):
        """Get length."""
        return len(self.imgs)


if __name__ == "__main__":
    with open("../configs/base.yml") as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    t_params = cfg["train_params"]
    dataset = Dataset(t_params=t_params, root=t_params["dataset_dir"], data_list_file=t_params["train_list"], phase="train")

    trainloader = data.DataLoader(dataset, batch_size=16)
    for i, (image, label) in enumerate(trainloader):
        print(image.shape)

        img = torchvision.utils.make_grid(image).numpy()

        img = np.transpose(img, (1, 2, 0))  # chw -> hwc
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imshow("img", img)
        # cv2.waitKey(0)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
