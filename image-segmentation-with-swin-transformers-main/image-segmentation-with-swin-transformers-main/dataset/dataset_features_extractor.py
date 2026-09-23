import torch
from torch.utils.data import Dataset
import numpy as np
import torch
import glob
import os
from torchvision import transforms
from dino_augment_transforms import DataAugmentationDINO

class FeaturesExtractorDataset(Dataset):
    def __init__(self, data_path='', size_w=224, size_h=224, flip=0, dataset_length=-1,
                 global_crops_scale = [0.14,1], local_crops_scale= [0.05, 0.14], local_crops_number= 6):
        super(FeaturesExtractorDataset, self).__init__()

        self.src_list = np.array(sorted(glob.glob(os.path.join(data_path,'imgs/*.npy'))))
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.resize = transforms.Resize(size=(size_w,size_h))

        self.transforms_dino = transforms.Compose([
                    transforms.ToPILImage(),
                    DataAugmentationDINO( global_crops_scale, local_crops_scale, local_crops_number),
                    transforms.Lambda(lambda crops: torch.stack([crop for crop in crops]))
                ])

        self.dataset_length = dataset_length
        if self.dataset_length >= 0:
            assert self.dataset_length <= len(self.src_list) , "sample index out of bound"
            self.src_list = self.src_list[:self.dataset_length]

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, idx):

        image = np.load(self.src_list[idx]).transpose([1,2,0])
        image = self.resize(torch.tensor(image).permute( 2, 0, 1).contiguous()).float()

        im_dino = self.transforms_dino(image)
        global_crops, local_crops = im_dino[:2], im_dino[2:]

        return global_crops, local_crops
