from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from .resnet_layers import block, ResNet, VectorQuantizer, VectorQuantizerEMA


class Encoder(nn.Module): # [3, 4]
    def __init__(self, block, layers, image_channels, num_classes):
        super(Encoder, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size = 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride=2, padding = 1)

        # Encoder layers
        self.layer1 = self._make_layer(block, layers[0], out_channels = 64, stride = 1)
        self.layer2 = self._make_layer(block, layers[1], out_channels = 128, stride = 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downdample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downdample = nn.Sequential( nn.Conv2d(self.in_channels, out_channels*4, kernel_size = 1), nn.BatchNorm2d(out_channels*4))


        layers.append( block(self.in_channels,out_channels,identity_downdample,stride ) )
        self.in_channels = out_channels * 4

        for i in range(num_residual_blocks - 1 ):
            layers.append( block(self.in_channels, out_channels) )

        return nn.Sequential(*layers)


class Decoder(nn.Module): # [6, 3]
    def __init__(self, block, layers, image_channels, num_classes):
        super(Decoder, self).__init__()
        self.in_channels = 512

        # Decoder layers
        self.layer3 = self._make_layer(block, layers[0], out_channels = 256, stride = 2)
        self.layer4 = self._make_layer(block, layers[1], out_channels = 512, stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4,num_classes)

    def forward(self, x):
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downdample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downdample = nn.Sequential( nn.Conv2d(self.in_channels, out_channels*4, kernel_size = 1), nn.BatchNorm2d(out_channels*4))

        layers.append( block(self.in_channels,out_channels,identity_downdample,stride ) )
        self.in_channels = out_channels * 4

        for i in range(num_residual_blocks - 1 ):
            layers.append( block(self.in_channels, out_channels) )

        return nn.Sequential(*layers)



def ResNet50_Encoder(img_channels = 3, num_classes=1000):
    return Encoder(block, [3,4], img_channels, num_classes)

def ResNet50_Decoder(img_channels = 3, num_classes=1000):
    return Decoder(block, [6, 3], img_channels, num_classes)


class ResNetVAE(nn.Module):
    def __init__(self,):
        super(Model, self).__init__()

        self._encoder = ResNet50_Encoder()
        self._decoder = ResNet50_Decoder()

    def forward(self, x):
        x = self._encoder(x)
        x = self._decoder(x)

        return x
