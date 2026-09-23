from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
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
from tqdm import tqdm
import torch.nn.functional as F

import numpy as np
import torch
import torch.utils.data as data
import glob
# import tifffile as tiff


'''
# TO MAKE IT WORKS WITH DINO CODE

from dino_augment_transforms import DataAugmentationDINO

global_crops_scale, local_crops_scale, local_crops_number = [0.14,1], [0.05, 0.14],   2,
augment = DataAugmentationDINO(global_crops_scale, local_crops_scale, local_crops_number)

from torchvision import transforms

transforms_dino = transforms.Compose([
            transforms.ToPILImage(),
            DataAugmentationDINO( [0.14,1], [0.05, 0.14],   2),
            transforms.Lambda(lambda crops: torch.stack([crop for crop in crops]))
        ])

from dataset import train_dataset

train_data_path = "/content/drive/MyDrive/Colab Notebooks/semantic_labelling/Dataset/Vaihingen_processed/train"

dataset = dataset_pretrain(data_path=train_data_path,
                        size_w=256,
                        size_h=256,
                        batch_size=8,
                        transform = transforms_dino
                        )

train_dataloader = dataset.data_iter_index(8)
'''


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".tif", '.png', '.jpg', '.npy'])

class dataset_pretrain(data.Dataset):
    def __init__(self, data_path='', size_w=256, size_h=256,
                batch_size=1, transform = None):
        super(dataset_pretrain, self).__init__()
        self.src_list = np.array(sorted(glob.glob(os.path.join(data_path,'imgs/*.npy'))))
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.index = 0
        self.batch_size = batch_size
        self.transform = transform

    def data_iter_index(self, index=1000):
        batch_index = np.random.choice(len(self.src_list), index)
        x_batch = self.src_list[batch_index]
        data_series = []
        data_series_teacher = []
        data_series_student = []

        try:
            for i in tqdm(range(index),desc="Loading images from memory"):

                im = np.load(x_batch[i]).transpose([1,2,0]) #/ 255.0

                if self.transform:
                    im = self.transform(im)
                    data_series_teacher.append(im[:2])
                    data_series_student.append(im[2:])
                else:
                    data_series.append(im)

                self.index += 1

        except OSError:
            return None, None

        if not self.transform:
            data_series = torch.from_numpy(np.array(data_series)).type(torch.FloatTensor)
            data_series = data_series.permute(0, 3, 1, 2).contiguous()
            torch_data = data_series
        else:
            data_series_teacher = torch.cat(data_series_teacher)
            data_series_student = torch.cat(data_series_student)
            torch_data = data.TensorDataset(data_series_teacher, data_series_student)

        data_iter = data.DataLoader(
            dataset=torch_data,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,
            num_workers=0,
        )

        return data_iter

    def data_iter(self):
        data_series = []
        data_series_teacher = []
        data_series_student = []

        try:
            for i in tqdm(range(len(self.src_list)),desc="Loading images from memory"):

                im = np.load(self.src_list[i]).transpose([1,2,0])# / 255.0

                if self.transform:
                    im = self.transform(im)

                    data_series_teacher.append(im[:2])
                    data_series_student.append(im[2:])
                else:
                    data_series.append(im)

                self.index += 1

        except OSError:
            return None, None

        if not self.transform:
            data_series = torch.from_numpy(np.array(data_series)).type(torch.FloatTensor)
            data_series = data_series.permute(0, 3, 1, 2).contiguous()
            torch_data = data_series
        else:
            data_series_teacher = torch.cat(data_series_teacher)
            data_series_student = torch.cat(data_series_student)
            torch_data = data.TensorDataset(data_series_teacher, data_series_student)

        data_iter = data.DataLoader(
            dataset=torch_data,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,
            num_workers=0,
        )

        return data_iter





class evaluation_dataset_pretrain(data.Dataset):
    def __init__(self, data_path='', size_w=224, size_h=224,
                batch_size=1, transform = None):
        super(evaluation_dataset_pretrain, self).__init__()
        self.src_list = np.array(sorted(glob.glob(os.path.join(data_path,'imgs/*.npy'))))
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.index = 0
        self.batch_size = batch_size
        self.transform = transform

    def data_iter_index(self, index=1000):
        batch_index = np.random.choice(len(self.src_list), index)
        x_batch = self.src_list[batch_index]
        data_series = []

        try:
            for i in tqdm(range(index),desc="Loading images from memory"):

                im = np.load(x_batch[i]).transpose([1,2,0]) #/ 255.0

                if self.transform:
                    im = self.transform(im)

                data_series.append(im)

                self.index += 1

        except OSError:
            return None, None

        data_series = torch.from_numpy(np.array(data_series)).type(torch.FloatTensor)
        data_series = data_series.permute(0, 3, 1, 2).contiguous()
        torch_data = F.interpolate(data_series, size=(self.size_w,self.size_h))


        data_iter = data.DataLoader(
            dataset=torch_data,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,
            num_workers=0,
        )

        return data_iter

    def data_iter(self):
        data_series = []

        try:
            for i in tqdm(range(len(self.src_list)),desc="Loading images from memory"):

                im = np.load(self.src_list[i]).transpose([1,2,0])# / 255.0

                if self.transform:
                    im = self.transform(im)

                data_series.append(im)

                self.index += 1

        except OSError:
            return None, None


        data_series = torch.from_numpy(np.array(data_series)).type(torch.FloatTensor)
        data_series = data_series.permute(0, 3, 1, 2).contiguous()
        torch_data = F.interpolate(data_series, size=(self.size_w,self.size_h))


        data_iter = data.DataLoader(
            dataset=torch_data,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,
            num_workers=0,
        )

        return data_iter
