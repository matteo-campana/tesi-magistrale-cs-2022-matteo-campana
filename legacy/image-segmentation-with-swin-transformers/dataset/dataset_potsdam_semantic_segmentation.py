import os
import numpy as np
import torch
import torch.utils.data as data
import glob
import torchvision.transforms as T
from tqdm import tqdm
from pathlib import Path


class DatasetPotsdamSemantiSegmentatin(data.Dataset):
    def __init__(self, data_path='', size_w=224, size_h=224, transform = None, dataset_length=-1 ,consistency_check_flag=False):
        super(DatasetPotsdamSemantiSegmentatin, self).__init__()

        self.src_list = np.array(sorted(glob.glob(os.path.join(data_path,'imgs/*.npy'))))
        self.lab_list = np.array(sorted(glob.glob(os.path.join(data_path,'masks/*.npy'))))
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h

        self.transform = transform
        self.resize = T.Resize(size=(size_w,size_h))

        if consistency_check_flag:
            # Dataset Consistency Check
            filenames_train_imgs = sorted(list(map(lambda x : Path(x).stem.replace("_RGB","") ,self.src_list)))
            filenames_train_masks = sorted(list(map(lambda x : Path(x).stem.replace("_label","") ,self.lab_list)))
            consistency_check = list(filter(lambda x : not x in filenames_train_masks,filenames_train_imgs))
            assert len(consistency_check) == 0

            # Dataset Consistency Check
            for x,y in tqdm(zip(filenames_train_imgs,filenames_train_masks),desc="dataset consistency check"):
                if x != y:
                    print("error, {} not equal to {}".format(x,y))
                    assert False, "[ERROR] Consistency check FAILED , tuple ({} ,{}) not accepted".format(x,y)

        self.dataset_length = dataset_length
        if self.dataset_length >= 0:
            assert self.dataset_length <= len(self.src_list) , "sample index out of bound"
            self.src_list = self.src_list[:self.dataset_length]
            self.lab_list = self.lab_list[:self.dataset_length]

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, idx):

        image = np.load(self.src_list[idx]) / 255
        label = np.load(self.lab_list[idx])

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        else:
            image = self.resize(torch.tensor(image)).float()
            label = self.resize(torch.tensor(label)).float()

        return image, label
