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


NClasses = 6
Background  = np.array([255,0,0],dtype=np.float32) // 255        # channel 0
ImSurf      = np.array ([255,255,255],dtype=np.float32) // 255   # channel 1
Car         = np.array([255,255,0],dtype=np.float32) // 255      # channel 2
Building    = np.array([0,0,255],dtype=np.float32) // 255        # channel 3
LowVeg      = np.array([0,255,255],dtype=np.float32) // 255      # channel 4
Tree        = np.array([0,255,0],dtype=np.float32) // 255        # channel 5


# process gts image in 256x256x3 to 256x256x6
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def to_class_channels_jit(sample):
    gts_6c = np.zeros((6,sample.shape[1],sample.shape[2]))
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
    gts_3c = np.zeros((3,sample.shape[1],sample.shape[2]))
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


# process gts image in 1x256x256 to 3x256x256
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def from_one_class_to_rgb_jit(sample):
    gts_3c = np.zeros((3,sample.shape[1],sample.shape[2]))
    for i in range(sample.shape[1]):
        for j in range(sample.shape[2]):
            if sample[0,i,j] == 0:
                gts_3c[:,i,j] = Background
            elif sample[0,i,j] == 1:
                gts_3c[:,i,j] = ImSurf
            elif sample[0,i,j] == 2:
                gts_3c[:,i,j] = Car
            elif sample[0,i,j] == 3:
                gts_3c[:,i,j] = Building
            elif sample[0,i,j] == 4:
                gts_3c[:,i,j] = LowVeg
            elif sample[0,i,j] == 5:
                gts_3c[:,i,j] = Tree
    return gts_3c

# process gts image in 3x256x256 to 1x256x256
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def from_rgb_to_class_jit(sample):
    gts_3c = np.zeros((1,sample.shape[1],sample.shape[2]),dtype=np.uint8)
    for i in range(sample.shape[1]):
        for j in range(sample.shape[2]):
            if np.all(sample[:,i,j] == Background):
                gts_3c[0,i,j] = 0
            elif np.all(sample[:,i,j] == ImSurf):
                gts_3c[0,i,j] = 1
            elif np.all(sample[:,i,j] == Car):
                gts_3c[0,i,j] = 2
            elif np.all(sample[:,i,j] == Building):
                gts_3c[0,i,j] = 3
            elif np.all(sample[:,i,j] == LowVeg):
                gts_3c[0,i,j] = 4
            elif np.all(sample[:,i,j] == Tree):
                gts_3c[0,i,j] = 5

    return gts_3c
