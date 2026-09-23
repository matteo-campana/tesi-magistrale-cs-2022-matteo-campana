
from PIL import Image

import numpy as np
from PIL import ImageFilter
import os
import math
from tqdm import tqdm
import glob
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
import torchvision.transforms as T
from matplotlib.pyplot import figure

import numpy as np
import torch
import torch.utils.data as data
import glob
# import tifffile as tiff

import numba
from numba import jit
import torch.nn.functional as F

from utils.dataset_utils import *


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".tif", '.png', '.jpg', '.npy'])

class Dataset_Semantic_Segmentation(data.Dataset):
    def __init__(self, data_path='', size_w=224, size_h=224, flip=0,
                batch_size=1, transform = None, dataset_length=-1 ):
        super(Dataset_Semantic_Segmentation, self).__init__()

        self.src_list = np.array(sorted(glob.glob(os.path.join(data_path,'imgs/*.npy'))))
        self.lab_list = np.array(sorted(glob.glob(os.path.join(data_path,'masks/*.npy'))))
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip
        self.index = 0
        self.batch_size = batch_size
        self.transform = transform
        self.resize = T.Resize(size=(size_w,size_h))

        self.dataset_length = dataset_length
        if self.dataset_length >= 0:
            assert self.dataset_length <= len(self.src_list) , "sample index out of bound"
            self.src_list = self.src_list[:self.dataset_length]
            self.lab_list = self.lab_list[:self.dataset_length]

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, idx):

        image = np.load(self.src_list[idx]).transpose([1,2,0])
        label = np.load(self.lab_list[idx])
        label = label.reshape(1,label.shape[0],label.shape[1])

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        else:
            image = self.resize(torch.tensor(image).permute( 2, 0, 1).contiguous()).float()
            label = self.resize(torch.tensor(label)).float()

        return image, label

    def data_iter_index(self, index=1000):
        batch_index = np.random.choice(len(self.src_list),index)
        x_batch = self.src_list[batch_index]
        y_batch = self.lab_list[batch_index]
        data_series = []
        label_series = []

        try:
            for i in tqdm(range(index),desc="Loading Images from Memory... "):

                im = np.load(x_batch[i]).transpose([1,2,0]) #/ 255.0

                if self.transform:
                  im = self.transform(im)

                data_series.append(im)

                gts = np.load(y_batch[i])
                gts = gts.reshape(1,gts.shape[0],gts.shape[1])
                #gts = from_one_class_to_rgb_jit(gts)
                label_series.append(gts)

                self.index += 1

        except OSError:
            return None, None


        data_series = torch.from_numpy(np.array(data_series)).type(torch.FloatTensor)
        data_series = data_series.permute(0, 3, 1, 2).contiguous()
        data_series = F.interpolate(data_series, size=(self.size_w,self.size_h))

        label_series = torch.from_numpy(np.array(label_series)).type(torch.FloatTensor)
        label_series = F.interpolate(label_series, size=(self.size_w,self.size_h))

        torch_data = data.TensorDataset(data_series, label_series)

        data_iter = data.DataLoader(
            dataset=torch_data,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,
            num_workers=0,
        )

        return data_iter

    def data_iter(self):
        data_series = []
        label_series = []

        try:
            for i in tqdm(range(len(self.src_list)),desc="Loading Images from Memory... "):

                im = np.load(self.src_list[i]).transpose([1,2,0])# / 255.0

                if self.transform:
                  im = self.transform(im)

                data_series.append(im)

                gts = np.load(self.lab_list[i])
                gts = gts.reshape(1,gts.shape[0],gts.shape[1])
                #gts = from_one_class_to_rgb_jit(gts)
                label_series.append(gts)

                self.index += 1

        except OSError:
            return None, None

        data_series = torch.from_numpy(np.array(data_series)).type(torch.FloatTensor)
        data_series = data_series.permute(0, 3, 1, 2).contiguous()
        data_series = F.interpolate(data_series, size=(self.size_w,self.size_h))

        label_series = torch.from_numpy(np.array(label_series)).type(torch.FloatTensor)
        label_series = F.interpolate(label_series, size=(self.size_w,self.size_h))

        torch_data = data.TensorDataset(data_series, label_series)

        data_iter = data.DataLoader(
            dataset=torch_data,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,
            num_workers=0,
        )


        return data_iter
