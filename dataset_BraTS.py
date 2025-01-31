import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import imageio as iio
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

def normalise_intensity(image, ROI_thres=0.1):
    pixel_thres = np.percentile(image, ROI_thres)
    ROI = np.where(image > pixel_thres, image, 0) # If image value is greater than pixel threshold, return image value, otherwise return 0
    mean = np.mean(ROI)
    std = np.std(ROI)
    ROI_norm = (ROI - mean) / (std + 1e-8) # Normalise ROI
    ROI_norm = np.clip(ROI_norm, 0, 3)  # Clamping values to be between 0 and 3

    return ROI_norm

def map_labels(label):
    label_map = {0: 0, 85: 1, 170: 2, 255: 3}
    mapped_label = label.copy()
    for k, v in label_map.items():
        mapped_label[label == k] = v
    return mapped_label

class BraTS_dataset(Dataset):
    def __init__(self, base_dir):
        self.img_path_all = glob(base_dir + '/BraTS-GLI-t1c/*.png')
        self.mask_path_all = [img_path.replace('t1c', 'seg') for img_path in self.img_path_all]

    def __len__(self):
      return len(self.img_path_all)

    def __getitem__(self, index):
        image = iio.imread(self.img_path_all[index])
        image = normalise_intensity(image)
        label = iio.imread(self.mask_path_all[index])
        label = map_labels(label)
        image = torch.from_numpy(image)
        label = torch.from_numpy(label) 
        return image, label
