import torch
import torchvision
import torchvision.transforms as T
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels
from torch.nn import Module, Conv2d, Parameter, Softmax
from functools import partial

from models.resnet_vae_v2 import ResNet50VAE, ResNetEncoder, ResNetDecoder, block, decoder_block
from models.vector_qunatizer import VectorQuantizer, VectorQuantizerEMA

import os
from collections import OrderedDict


def conv3otherRelu(in_planes, out_planes, kernel_size=None, stride=None, padding=None):
    # 3x3 convolution with padding and relu
    if kernel_size is None:
        kernel_size = 3

    assert isinstance(kernel_size, (int, tuple)), 'kernel_size is not in (int, tuple)!'

    if stride is None:
        stride = 1
    assert isinstance(stride, (int, tuple)), 'stride is not in (int, tuple)!'

    if padding is None:
        padding = 1
    assert isinstance(padding, (int, tuple)), 'padding is not in (int, tuple)!'

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.ReLU(inplace=True)  # inplace=True
    )

def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class PAM_Module(Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(PAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        # self.exp_feature = exp_feature_map
        # self.tanh_feature = tanh_feature_map
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


class CAM_Module(Module):
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        batch_size, chnnels, width, height = x.shape
        proj_query = x.view(batch_size, chnnels, -1)
        proj_key = x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(batch_size, chnnels, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, chnnels, height, width)

        out = self.gamma * out + x
        return out


class PAM_CAM_Layer(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(PAM_CAM_Layer, self).__init__()
        self.conv1 = conv3otherRelu(in_ch, in_ch)

        self.PAM = PAM_Module(in_ch)
        self.CAM = CAM_Module()

        self.conv2P = nn.Sequential(nn.Dropout2d(0.1, False), conv3otherRelu(in_ch, in_ch, 1, 1, 0))
        self.conv2C = nn.Sequential(nn.Dropout2d(0.1, False), conv3otherRelu(in_ch, in_ch, 1, 1, 0))

        self.conv3 = nn.Sequential(nn.Dropout2d(0.1, False), conv3otherRelu(in_ch, out_ch , 1, 1, 0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2P(self.PAM(x)) + self.conv2C(self.CAM(x))
        return self.conv3(x)


# <---------------------------------------------------------------------------->
# MAResUNet50
# <---------------------------------------------------------------------------->

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, nonlinearity):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class MAResUNet50_VQ_SS(nn.Module):
    def __init__(self,
                 decoder_block, layers,
                 nonlinearity =  partial(F.relu, inplace=True),
                 pretrain = "imagenet",
                 num_channels=3, num_classes=6,
                 commitment_cost = 0.25,decay = 0.99):

        super(MAResUNet50_VQ_SS, self).__init__()
        self.name = 'MAResUNet50_VQ_SS'

        if pretrain == "imagenet":
            base_model = tmodels.resnet50(pretrained = True)
        elif pretrain == "imagenet-dino":
            base_model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        elif pretrain == "none":
            base_model = tmodels.resnet50(pretrained = False)
        elif pretrain == "EsViT":
            base_model = tmodels.resnet50(pretrained = True)
            os.system("wget -nc https://chunyleu.blob.core.windows.net/output/ckpts/esvit/resnet/resnet50/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/resume_from_ckpt0200/checkpoint.pth")
            checkpoint_resnet50 = torch.load('./checkpoint.pth')#, map_location=torch.device('cpu'))
            student_network_ckpt = checkpoint_resnet50["student"]
            resnet50_state_dict_dino_student = OrderedDict()
            for key in student_network_ckpt.keys():
                if "module.backbone.backbone" in key:
                    #print(key.replace("module.backbone.backbone.",""))
                    resnet50_state_dict_dino_student[key.replace("module.backbone.backbone.","")] = student_network_ckpt[key]
            base_model.load_state_dict(resnet50_state_dict_dino_student,strict=False)

        resnet = base_model
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        filters = [256, 512, 1024, 2048]

        att_filters = list(map(lambda x : x // 2 ,filters))

        self.attention4 = PAM_CAM_Layer(filters[3], filters[3])
        self.attention3 = PAM_CAM_Layer(filters[2], filters[2]//2)
        self.attention2 = PAM_CAM_Layer(filters[1], filters[1]//2)
        self.attention1 = PAM_CAM_Layer(filters[0], filters[0]//2)

        self.decoder = ResNetDecoder(decoder_block, layers, num_channels)

        self.decoder4 = self.decoder.layer1
        self.decoder3 = self.decoder.layer2
        self.decoder2 = self.decoder.layer3
        self.decoder1 = self.decoder.layer4

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0]//4, 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        num_embeddings = 2048
        embedding_dim = 2048

        if decay > 0.0:
            self.vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self.vq_vae = VectorQuantizer(num_embeddings, embedding_dim,commitment_cost)

    def forward(self, x):
        # Encoder
        x1 = self.firstconv(x)
        x1 = self.firstbn(x1)
        x1 = self.firstrelu(x1)
        x1 = self.firstmaxpool(x1)

        e1 = self.encoder1(x1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.attention4(e4)

        loss, quantized, perplexity, encodings = self.vq_vae(e4)

        # Decoder
        d4 = self.decoder4(quantized) + self.attention3(e3)
        d3 = self.decoder3(d4) + self.attention2(e2)
        d2 = self.decoder2(d3) + self.attention1(e1)
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return loss, quantized, perplexity, encodings, out
# <---------------------------------------------------------------------------->
# NETWORK BUILDER PRE-TRAINING
# <---------------------------------------------------------------------------->

def MAResUNet50_VQ_SS_(num_channels=3,num_classes = 6, commitment_cost = 0.25, decay = 0.99,pretrain = "imagenet"):
    return MAResUNet50_VQ_SS(num_channels=num_channels, num_classes=num_classes, decoder_block=decoder_block, layers=[3, 6, 4, 3],commitment_cost = commitment_cost,decay = decay,pretrain = pretrain)
# <--------------------------------------------------------
