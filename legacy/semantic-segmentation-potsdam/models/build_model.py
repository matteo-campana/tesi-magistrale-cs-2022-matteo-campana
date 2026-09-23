from .MAResUNet import MAResUNet50VAE, MAResUNet101VAE, MAResUNet152VAE, MAResUNet50VAE_SS, MAResUNet101VAE_SS, MAResUNet152VAE_SS, MAResUNet18_
from .MAResUNet_vq import MAResUNet50_VQ, MAResUNet101_VQ, MAResUNet152_VQ, MAResUNet50_VQ_SS, MAResUNet101_VQ_SS, MAResUNet152_VQ_SS, MAResUNet18_VQ_SS_
from .MAResUNet50 import MAResUNet50_SS_
from .MAResUNet50_vq import MAResUNet50_VQ_SS_

import torch
import torch.nn as nn
import torch.nn.functional as F

def build_model(model_name="maresunet18"):

    model = None

# <---------------------------------------------------------------------------->
# MAResUNet SEMANTIC-SEGMENTATION
# <---------------------------------------------------------------------------->
    if model_name == "maresunet18":
        model = MAResUNet18_(img_channel=3,num_classes = 6)
    elif model_name == "maresunet50":
        #model =MAResUNet50VAE_SS(img_channel=3)
        model = MAResUNet50_SS_(num_channels=3,num_classes = 6)
    elif model_name == "maresunet101":
        model = MAResUNet101VAE_SS(img_channel=3)
    elif model_name == "maresunet152":
        model = MAResUNet152VAE_SS(img_channel=3)

# <---------------------------------------------------------------------------->
# MAResUNet WITH VECTOR QUNATIZER SEMANTIC-SEGMENTATION
# <---------------------------------------------------------------------------->

    elif model_name == "maresunet18-vq":
        model = MAResUNet18_VQ_SS_(img_channel=3,commitment_cost = 0.25,decay = 0.99)
    elif model_name == "maresunet50-vq":
        #model =MAResUNet50_VQ_SS(img_channel=3,commitment_cost = 0.25,decay = 0.99)
        model = MAResUNet50_VQ_SS_(num_channels=3,num_classes = 6, commitment_cost = 0.25, decay = 0.99)
    elif model_name == "maresunet101-vq":
        model =MAResUNet101_VQ_SS(img_channel=3,commitment_cost = 0.25,decay = 0.99)
    elif model_name == "maresunet152-vq":
        model =MAResUNet152_VQ_SS(img_channel=3,commitment_cost = 0.25,decay = 0.99)

# <---------------------------------------------------------------------------->
# MAResUNet WITH PRETRAIN DINO FACEBOOK
# <---------------------------------------------------------------------------->
    elif model_name == "maresunet50-pretrain-dino-fair":
        model = MAResUNet50_SS_(num_channels=3,num_classes = 3,pretrain = "imagenet-dino")
    elif model_name == "maresunet50-vq-pretrain-dino-fair":
        model = MAResUNet50_VQ_SS_(num_channels=3,num_classes = 3, commitment_cost = 0.25, decay = 0.99,pretrain = "imagenet-dino")
    elif model_name == "maresunet50-ss-pretrain-dino-fair":
        model = MAResUNet50_SS_(num_channels=3,num_classes = 6,pretrain = "imagenet-dino")
    elif model_name == "maresunet50-ss-vq-pretrain-dino-fair":
        model = MAResUNet50_VQ_SS_(num_channels=3,num_classes = 6, commitment_cost = 0.25, decay = 0.99,pretrain = "imagenet-dino")

# <---------------------------------------------------------------------------->
# MAResUNet WITH PRETRAIN EsViT MICROSOFT
# <---------------------------------------------------------------------------->
    elif model_name == "maresunet50-pretrain-EsViT":
        model = MAResUNet50_SS_(num_channels=3,num_classes = 3,pretrain = "EsViT")
    elif model_name == "maresunet50-vq-pretrain-EsViT":
        model = MAResUNet50_VQ_SS_(num_channels=3,num_classes = 3, commitment_cost = 0.25, decay = 0.99,pretrain = "EsViT")
    elif model_name == "maresunet50-ss-pretrain-EsViT":
        model = MAResUNet50_SS_(num_channels=3,num_classes = 6,pretrain = "EsViT")
    elif model_name == "maresunet50-ss-vq-pretrain-EsViT":
        model = MAResUNet50_VQ_SS_(num_channels=3,num_classes = 6, commitment_cost = 0.25, decay = 0.99,pretrain = "EsViT")


    return model
