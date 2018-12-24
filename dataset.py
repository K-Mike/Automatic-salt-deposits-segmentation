import torch
import os
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset

from . import config as cf


class SaltDataset(Dataset):
    """
    Custom Pytorch Dataset implementation

    (Based on https://github.com/ternaus/robot-surgery-segmentation)

    Parameters
    ----------
    image_files : list of str
        image files to use in the dataset

    mask_files : list of str
        mask files to use in the dataset

    transform : func (default=None)
        transform image/mask tuple after load from disk

    mode : str (default='train')
        dataset operational mode, if not 'train', mask is dropped

    binary_factor : int (default=255)
        mask positive value
    """
    def __init__(self, image_files, mask_files=None, transform=None, mode='train', binary_factor=255):
        if mask_files is not None:
            assert len(image_files) == len(mask_files)

        # Remove bad images
        self.image_files = list()
        self.mask_files = list() if mask_files is not None else None
        for i, image_path in enumerate(image_files):
            img_name = os.path.basename(image_path)

            if img_name not in cf.BAD_TRAIN_IMAGES:
                self.image_files.append(image_files[i])
                if mask_files is not None:
                    self.mask_files.append(mask_files[i])

        self.transform = transform
        self.mode = mode
        self.binary_factor = binary_factor
        # Load depth information
        depths = pd.read_csv(os.path.join(cf.INPUT_DIR, 'depths.csv'))
        self.id_depth_dict = depths.set_index('id').to_dict()['z']
        self.min_depth = depths['z'].min()
        self.max_depth = depths['z'].max()
        # Create gradient depth image
        self.depth_grad = np.zeros(cf.INPUT_IMAGE_SIZE)
        h = self.depth_grad.shape[0]
        for i, v in enumerate(np.linspace(0.001, 1, h)):
            self.depth_grad[i] = v

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_full = load_image(self.image_files[idx])
        img_id = os.path.basename(self.image_files[idx])[:-4]
        # Add relative depth as the second channel
        relative_depth = (self.id_depth_dict[img_id] - self.min_depth) / (self.max_depth - self.min_depth)
        img_full[:, :, 1] = int(relative_depth * 255)
        # Add image * depth_gradient as the third channel
        img_full[:, :, 2] = (img_full[:, :, 2].astype(float) * self.depth_grad).astype('uint8')

        if self.mode == 'train':
            mask_full = load_mask(self.mask_files[idx], self.binary_factor)
            img, mask = self.transform(img_full, mask_full)

            return to_float_tensor(img), torch.from_numpy(np.expand_dims(mask, 0)).float()
        else:
            img, _ = self.transform(img_full)
            return to_float_tensor(img), os.path.basename(self.image_files[idx])


def to_float_tensor(img):
    return torch.from_numpy(np.moveaxis(img, -1, 0)).float()


def load_image(path):
    """Read image from disk in conventional RGB format"""
    img = cv2.imread(path)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path, factor=255):
    """Read mask from disk in grayscale and normalize to [0; 1]"""
    mask = cv2.imread(path, 0)

    if mask is not None:
        mask = (mask / factor).astype(np.uint8)

    return mask
