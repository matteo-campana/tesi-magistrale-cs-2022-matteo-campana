from models.resnet_vae_v2 import ResNet50VAE, ResNetEncoder, ResNetDecoder, block, decoder_block
from models.vector_qunatizer import VectorQuantizer, VectorQuantizerEMA
import torch
import torch.nn as nn
import torchvision
from torchvision.utils import make_grid
from torchvision import transforms as T

# <---------------------------------------------------------------------------->
# RESNET VARIATIONAL AUTO ENCODER FULL MODEL
# <---------------------------------------------------------------------------->

class ResNetVAE_VQ(nn.Module):
    def __init__(self, block ,decoder_block, layers, image_channels,commitment_cost = 0.25,decay = 0.99):
        super(ResNetVAE_VQ, self).__init__()

        self.encoder = ResNetEncoder(block, layers, image_channels)
        self.decoder = ResNetDecoder(decoder_block, layers, image_channels)
        num_embeddings = 2048
        embedding_dim = 2048

        if decay > 0.0:
            self.vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self.vq_vae = VectorQuantizer(num_embeddings, embedding_dim,commitment_cost)


    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, encodings = self.vq_vae(z)
        x = self.decoder(quantized)

        return loss, quantized, perplexity, encodings, x

# <---------------------------------------------------------------------------->
# RESNET VARIATIONAL AUTO ENCODER FULL MODEL SEMANTIC SEGMENTATION
# <---------------------------------------------------------------------------->


class ResNetVAE_SS_VQ(nn.Module):
    def __init__(self, block ,decoder_block, layers, image_channels,commitment_cost = 0.25,decay = 0.99):
        super(ResNetVAE_SS_VQ, self).__init__()

        self.encoder = ResNetEncoder(block, layers, image_channels)
        self.decoder = ResNetDecoder(decoder_block, layers, image_channels)
        self.proj    = nn.ConvTranspose2d(image_channels,6, kernel_size=1, stride=1, padding=0, bias=False)
        num_embeddings = 2048
        embedding_dim = 2048
        self.finalbn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()

        if decay > 0.0:
            self.vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self.vq_vae = VectorQuantizer(num_embeddings, embedding_dim,commitment_cost)


    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, encodings = self.vq_vae(z)
        x = self.decoder(quantized)
        x = self.proj(x)
        x = self.finalbn(x)
        x = self.relu(x)

        return loss, quantized, perplexity, encodings, x

# <---------------------------------------------------------------------------->
# NETWORK BUILDER PRE-TRAINING
# <---------------------------------------------------------------------------->

def ResNet50VAE_VQ(img_channel=3,commitment_cost = 0.25,decay = 0.99):
    return ResNetVAE_VQ(block, decoder_block, [3, 6, 4, 3], img_channel, commitment_cost ,decay )

def ResNet101VAE_VQ(img_channel=3,commitment_cost = 0.25,decay = 0.99):
    return ResNetVAE_VQ(block, decoder_block, [3, 23, 4, 3], img_channel, commitment_cost ,decay )

def ResNet152VAE_VQ(img_channel=3,commitment_cost = 0.25,decay = 0.99):
    return ResNetVAE_VQ(block, decoder_block, [3, 36, 8, 3], img_channel, commitment_cost ,decay )

# <---------------------------------------------------------------------------->
# NETWORK BUILDER SEMANTIC SEGMENTATION
# <---------------------------------------------------------------------------->


def ResNet50VAE_SS_VQ(img_channel=3,commitment_cost = 0.25,decay = 0.99):
    return ResNetVAE_SS_VQ(block, decoder_block, [3, 6, 4, 3], img_channel, commitment_cost ,decay )

def ResNet101VAE_SS_VQ(img_channel=3,commitment_cost = 0.25,decay = 0.99):
    return ResNetVAE_SS_VQ(block, decoder_block, [3, 23, 4, 3], img_channel, commitment_cost ,decay )

def ResNet152VAE_SS_VQ(img_channel=3,commitment_cost = 0.25,decay = 0.99):
    return ResNetVAE_SS_VQ(block, decoder_block, [3, 36, 8, 3], img_channel, commitment_cost ,decay )
