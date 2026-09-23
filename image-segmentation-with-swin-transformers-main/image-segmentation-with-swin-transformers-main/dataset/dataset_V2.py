from __future__ import print_function, division

from PIL import Image
import numpy as np
from PIL import ImageFilter
import os
import math
from tqdm import tqdm
import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
import torchvision.transforms as T

from numba import jit


gts_for_participants_patches_dir = "/content/drive/MyDrive/Colab Notebooks/semantic_labelling/Dataset/ISPRS_semantic_labeling_Vaihingen_processed/gts_for_participants/"
top_patches_dir = "/content/drive/MyDrive/Colab Notebooks/semantic_labelling/Dataset/ISPRS_semantic_labeling_Vaihingen_processed/top/"


class VaihingenDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, type = 'train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.top_dir = self.root_dir + '/top'
        self.gts_dir = self.root_dir + '/gts_for_participants'
        self.transform = transform
        self.gts_filename_list = []
        self.top_filename_list = []

        for root, dirs, files in os.walk(self.gts_dir, topdown=False):
            for name in files:
                self.gts_filename_list.append(os.path.join(root, name))

        for root, dirs, files in os.walk(self.top_dir, topdown=False):
            for name in files:
                self.top_filename_list.append(os.path.join(root, name))

        self.file_with_mask = list(set(list(map(lambda x : os.path.basename(x[:x.find("_patch_")]+".tif"), self.gts_filename_list ))))
        self.files_without_mask = list(set(list(map(lambda x : os.path.basename(x[:x.find("_patch_")]+".tif"), self.top_filename_list))) - set(self.file_with_mask))

        if self.transform == None:
            self.transform = T.Compose([
                T.ToTensor()
            ])

        self.type = type

        if self.type == "train":
            self.file_to_iter = self.file_with_mask[:12]
        elif self.type== "valid":
            self.file_to_iter = self.file_with_mask[12:14]
        elif self.type== "test":
            self.file_to_iter = self.file_with_mask[14:]
        else:
            raise ValueError("type must be train, valid, or test")

        self.patches_to_iter = list(filter(lambda x : os.path.basename(x[:x.find("_patch_")])+".tif" in self.file_to_iter  , self.gts_filename_list))
        self.patches_to_iter = list(map(lambda x : os.path.basename(x), self.patches_to_iter))

    def __len__(self):
        return len(self.patches_to_iter)

    def __getitem__(self, idx):
        filename = self.patches_to_iter[idx]
        gts_filepath = self.gts_dir + "/" + filename
        gts_img = Image.open(gts_filepath)
        top_filepath = self.top_dir + "/" + filename
        top_img = Image.open(top_filepath)
        top_tensor = self.transform(top_img)
        gts_tensor = self.transform(gts_img)

        return top_tensor, gts_tensor




NClasses = 6
Background = np.array([255,0,0],dtype=np.float32) // 255        # channel 0
ImSurf = np.array ([255,255,255],dtype=np.float32) // 255       # channel 1
Car = np.array([255,255,0],dtype=np.float32) // 255             # channel 2
Building = np.array([0,0,255],dtype=np.float32) // 255          # channel 3
LowVeg = np.array([0,255,255],dtype=np.float32) // 255          # channel 4
Tree = np.array([0,255,0],dtype=np.float32) // 255              # channel 5


# process gts image in 256x256x3 to 256x256x6
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def to_class_channels_jit(sample):
    gts_6c = np.zeros((6,256,256))
    for i in range(sample.shape[1]):
        for j in range(sample.shape[2]):
            if np.array_equal(sample[:,i,j] , Background):
                gts_6c[0,i,j] = 1
            elif np.array_equal(sample[:,i,j] , ImSurf):
                gts_6c[1,i,j] = 1
            elif np.array_equal(sample[:,i,j] , Car):
                gts_6c[2,i,j] = 1
            elif np.array_equal(sample[:,i,j] , Building):
                gts_6c[3,i,j] = 1
            elif np.array_equal(sample[:,i,j] , LowVeg):
                gts_6c[4,i,j] = 1
            elif np.array_equal(sample[:,i,j] , Tree):
                gts_6c[5,i,j] = 1
    return gts_6c


# process gts image in 256x256x6 to 256x256x3
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def from_class_to_rgb_jit(sample):
    gts_3c = np.zeros((3,256,256))
    for i in range(sample.shape[1]):
        for j in range(sample.shape[2]):
            if sample[0,i,j] == 1:
                gts_3c[:,i,j] = Background
            elif sample[1,i,j] == 1:
                gts_3c[:,i,j] = ImSurf
            elif sample[2,i,j] == 1:
                gts_3c[:,i,j] = Car
            elif sample[3,i,j] == 1:
                gts_3c[:,i,j] = Building
            elif sample[4,i,j] == 1:
                gts_3c[:,i,j] = LowVeg
            elif sample[5,i,j] == 1:
                gts_3c[:,i,j] = Tree
    return gts_3c
