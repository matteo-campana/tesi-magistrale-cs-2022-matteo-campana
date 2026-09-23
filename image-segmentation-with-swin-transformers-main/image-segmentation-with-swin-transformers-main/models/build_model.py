from .resnet_vae_v2 import ResNet50VAE, ResNet101VAE, ResNet152VAE, ResNet50VAE_SS, ResNet101VAE_SS, ResNet152VAE_SS
from .resnet_vq_vae_v2 import ResNet50VAE_VQ, ResNet101VAE_VQ, ResNet152VAE_VQ, ResNet50VAE_SS_VQ, ResNet101VAE_SS_VQ, ResNet152VAE_SS_VQ
from .swin_transformer_vae import SwinTransformerVAE, SwinTransformerVAE_SS
from .swin_transformer_vq_vae import SwinTransformerVQVAE, SwinTransformerVQVAE_SS
from .MAResUNet import MAResUNet50VAE, MAResUNet101VAE, MAResUNet152VAE, MAResUNet50VAE_SS, MAResUNet101VAE_SS, MAResUNet152VAE_SS, MAResUNet18_
from .MAResUNet_vq import MAResUNet50_VQ, MAResUNet101_VQ, MAResUNet152_VQ, MAResUNet50_VQ_SS, MAResUNet101_VQ_SS, MAResUNet152_VQ_SS, MAResUNet18_VQ_SS_
from .MAResUNet50 import MAResUNet50_SS_
from .MAResUNet50_vq import MAResUNet50_VQ_SS_
from .MASWinTNet import MASWinTNet_Tiny_SS_, MASWinTNet_Nano_SS_, MASWinTNet_Tiny_, MASWinTNet_Nano_
from .MASWinTNet_vq import MASWinTNetVQ_Tiny_SS_, MASWinTNetVQ_Nano_SS_, MASWinTNetVQ_Tiny_, MASWinTNetVQ_Nano_
from .SWinTUNet import SWinTUNet_
from .SWinTUNetVQ import SWinTUNetVQ_

import torch
import torch.nn as nn
import torch.nn.functional as F

def build_model(model_name="swin-transformer-vq-vae"):

    model = None

# <---------------------------------------------------------------------------->
# RESNET PRE-TRAINING
# <---------------------------------------------------------------------------->

    if model_name == "resnet50-vae":
        model =ResNet50VAE(img_channel=3)
    elif model_name == "resnet101-vae":
        model =ResNet101VAE(img_channel=3)
    elif model_name == "resnet152-vae":
        model =ResNet152VAE(img_channel=3)

# <---------------------------------------------------------------------------->
# RESNET WITH VECTOR QUNATIZER PRE-TRAINING
# <---------------------------------------------------------------------------->

    elif model_name == "resnet50-vq-vae":
        model =ResNet50VAE_VQ(img_channel=3,commitment_cost = 0.25,decay = 0.99)
    elif model_name == "resnet101-vq-vae":
        model =ResNet101VAE_VQ(img_channel=3,commitment_cost = 0.25,decay = 0.99)
    elif model_name == "resnet152-vq-vae":
        model =ResNet152VAE_VQ(img_channel=3,commitment_cost = 0.25,decay = 0.99)

# <---------------------------------------------------------------------------->
# RESNET SEMANTIC-SEGMENTATION
# <---------------------------------------------------------------------------->

    elif model_name == "resnet50-ss-vae":
        model =ResNet50VAE_SS(img_channel=3)
    elif model_name == "resnet101-ss-vae":
        model =ResNet101VAE_SS(img_channel=3)
    elif model_name == "resnet152-ss-vae":
        model =ResNet152VAE_SS(img_channel=3)

# <---------------------------------------------------------------------------->
# RESNET WITH VECTOR QUNATIZER SEMANTIC-SEGMENTATION
# <---------------------------------------------------------------------------->

    elif model_name == "resnet50-ss-vq-vae":
        model =ResNet50VAE_SS_VQ(img_channel=3,commitment_cost = 0.25,decay = 0.99)
    elif model_name == "resnet101-ss-vq-vae":
        model =ResNet101VAE_SS_VQ(img_channel=3,commitment_cost = 0.25,decay = 0.99)
    elif model_name == "resnet152-ss-vq-vae":
        model =ResNet152VAE_SS_VQ(img_channel=3,commitment_cost = 0.25,decay = 0.99)

# #######################################################################################################################################
# #######################################################################################################################################
# #######################################################################################################################################

# <---------------------------------------------------------------------------->
# MAResUNet PRE-TRAINING
# <---------------------------------------------------------------------------->
    elif model_name == "maresunet18":
        model = MAResUNet18_(img_channel=3,num_classes = 3)
    elif model_name == "maresunet50":
        #model = MAResUNet50VAE(img_channel=3)
        model = MAResUNet50_SS_(num_channels=3,num_classes = 3)
    elif model_name == "maresunet101":
        model = MAResUNet101VAE(img_channel=3)
    elif model_name == "maresunet152":
        model = MAResUNet152VAE(img_channel=3)

# <---------------------------------------------------------------------------->
# MAResUNet WITH VECTOR QUNATIZER PRE-TRAINING
# <---------------------------------------------------------------------------->
    elif model_name == "maresunet18-vq":
        model = MAResUNet18_VQ_SS_(img_channel=3,num_classes = 3, commitment_cost = 0.25,decay = 0.99)
    elif model_name == "maresunet50-vq":
        #model =MAResUNet50_VQ(img_channel=3,commitment_cost = 0.25,decay = 0.99)
        model = MAResUNet50_VQ_SS_(num_channels=3,num_classes = 3, commitment_cost = 0.25, decay = 0.99)
    elif model_name == "maresunet101-vq":
        model =MAResUNet101_VQ(img_channel=3,commitment_cost = 0.25,decay = 0.99)
    elif model_name == "maresunet152-vq":
        model =MAResUNet152_VQ(img_channel=3,commitment_cost = 0.25,decay = 0.99)

# #######################################################################################################################################
# #######################################################################################################################################
# #######################################################################################################################################

# <---------------------------------------------------------------------------->
# MAResUNet SEMANTIC-SEGMENTATION
# <---------------------------------------------------------------------------->
    elif model_name == "maresunet18-ss":
        model = MAResUNet18_(img_channel=3,num_classes = 6)
    elif model_name == "maresunet50-ss":
        #model =MAResUNet50VAE_SS(img_channel=3)
        model = MAResUNet50_SS_(num_channels=3,num_classes = 6)
    elif model_name == "maresunet101-ss":
        model = MAResUNet101VAE_SS(img_channel=3)
    elif model_name == "maresunet152-ss":
        model = MAResUNet152VAE_SS(img_channel=3)


# <---------------------------------------------------------------------------->
# MAResUNet WITH VECTOR QUNATIZER SEMANTIC-SEGMENTATION
# <---------------------------------------------------------------------------->

    elif model_name == "maresunet18-ss-vq":
        model = MAResUNet18_VQ_SS_(img_channel=3,commitment_cost = 0.25,decay = 0.99)
    elif model_name == "maresunet50-ss-vq":
        #model =MAResUNet50_VQ_SS(img_channel=3,commitment_cost = 0.25,decay = 0.99)
        model = MAResUNet50_VQ_SS_(num_channels=3,num_classes = 6, commitment_cost = 0.25, decay = 0.99)
    elif model_name == "maresunet101-ss-vq":
        model =MAResUNet101_VQ_SS(img_channel=3,commitment_cost = 0.25,decay = 0.99)
    elif model_name == "maresunet152-ss-vq":
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


# #######################################################################################################################################
# #######################################################################################################################################
# #######################################################################################################################################


# <---------------------------------------------------------------------------->
# SWIN-TRANSFORMERS NANO PRE-TRAINING
# <---------------------------------------------------------------------------->


    elif model_name == "swin-transformer-nano-vae":
        model = SwinTransformerVAE( img_size=224,
                                    patch_size=4,
                                    in_chans=3,
                                    num_classes=6,
                                    embed_dim=96,
                                    depths=[1, 1, 3, 1],
                                    num_heads=[1, 2, 4, 8],
                                    window_size=7,
                                    mlp_ratio=4.,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    drop_rate=0.,
                                    attn_drop_rate=0.,
                                    drop_path_rate=0.1,
                                    norm_layer=nn.LayerNorm,
                                    ape=False,
                                    patch_norm=True,
                                    use_checkpoint=False)

# <---------------------------------------------------------------------------->
# SWIN-TRANSFORMERS NANO SEMANTIC-SEGMENTATION
# <---------------------------------------------------------------------------->


    elif model_name == "swin-transformer-nano-vae-ss":
        model = SwinTransformerVAE_SS( img_size=224,
                                    patch_size=4,
                                    in_chans=3,
                                    num_classes=6,
                                    embed_dim=96,
                                    depths=[1, 1, 3, 1],
                                    num_heads=[1, 2, 4, 8],
                                    window_size=7,
                                    mlp_ratio=4.,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    drop_rate=0.,
                                    attn_drop_rate=0.,
                                    drop_path_rate=0.1,
                                    norm_layer=nn.LayerNorm,
                                    ape=False,
                                    patch_norm=True,
                                    use_checkpoint=False)

# <---------------------------------------------------------------------------->
# SWIN-TRANSFORMERS PRE-TRAINING
# <---------------------------------------------------------------------------->


    elif model_name == "swin-transformer-vae":
        model = SwinTransformerVAE( img_size=224,
                                    patch_size=4,
                                    in_chans=3,
                                    num_classes=6,
                                    embed_dim=96,
                                    depths=[2, 2, 6, 2],
                                    num_heads=[3, 6, 12, 24],
                                    window_size=7,
                                    mlp_ratio=4.,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    drop_rate=0.,
                                    attn_drop_rate=0.,
                                    drop_path_rate=0.1,
                                    norm_layer=nn.LayerNorm,
                                    ape=False,
                                    patch_norm=True,
                                    use_checkpoint=False)

# <---------------------------------------------------------------------------->
# SWIN-TRANSFORMERS SEMANTIC-SEGMENTATION
# <---------------------------------------------------------------------------->


    elif model_name == "swin-transformer-vae-ss":
        model = SwinTransformerVAE_SS( img_size=224,
                                    patch_size=4,
                                    in_chans=3,
                                    num_classes=6,
                                    embed_dim=96,
                                    depths=[2, 2, 6, 2],
                                    num_heads=[3, 6, 12, 24],
                                    window_size=7,
                                    mlp_ratio=4.,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    drop_rate=0.,
                                    attn_drop_rate=0.,
                                    drop_path_rate=0.1,
                                    norm_layer=nn.LayerNorm,
                                    ape=False,
                                    patch_norm=True,
                                    use_checkpoint=False)

# <---------------------------------------------------------------------------->
# SWIN-TRANSFORMERS NANO WITH VECTOR QUNATIZER PRE-TRAINING
# <---------------------------------------------------------------------------->


    elif model_name == "swin-transformer-nano-vq-vae":
        model = SwinTransformerVQVAE( img_size=224,
                                    patch_size=4,
                                    in_chans=3,
                                    num_classes=6,
                                    embed_dim=96,
                                    depths=[1, 1, 3, 1],
                                    num_heads=[1, 2, 4, 8],
                                    window_size=7,
                                    mlp_ratio=4.,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    drop_rate=0.,
                                    attn_drop_rate=0.,
                                    drop_path_rate=0.1,
                                    norm_layer=nn.LayerNorm,
                                    ape=False,
                                    patch_norm=True,
                                    use_checkpoint=False)

# <---------------------------------------------------------------------------->
# SWIN-TRANSFORMERS NANO WITH VECTOR QUNATIZER SEMANTIC-SEGMENTATION
# <---------------------------------------------------------------------------->


    elif model_name == "swin-transformer-nano-vq-vae-ss":
        model = SwinTransformerVQVAE_SS( img_size=224,
                                    patch_size=4,
                                    in_chans=3,
                                    num_classes=6,
                                    embed_dim=96,
                                    depths=[1, 1, 3, 1],
                                    num_heads=[1, 2, 4, 8],
                                    window_size=7,
                                    mlp_ratio=4.,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    drop_rate=0.,
                                    attn_drop_rate=0.,
                                    drop_path_rate=0.1,
                                    norm_layer=nn.LayerNorm,
                                    ape=False,
                                    patch_norm=True,
                                    use_checkpoint=False)


# <---------------------------------------------------------------------------->
# SWIN-TRANSFORMERS WITH VECTOR QUNATIZER PRE-TRAINING
# <---------------------------------------------------------------------------->

    elif model_name == "swin-transformer-vq-vae":
        model = SwinTransformerVQVAE( img_size=224,
                                    patch_size=4,
                                    in_chans=3,
                                    num_classes=6,
                                    embed_dim=96,
                                    depths=[2, 2, 6, 2],
                                    num_heads=[3, 6, 12, 24],
                                    window_size=7,
                                    mlp_ratio=4.,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    drop_rate=0.,
                                    attn_drop_rate=0.,
                                    drop_path_rate=0.1,
                                    norm_layer=nn.LayerNorm,
                                    ape=False,
                                    patch_norm=True,
                                    commitment_cost = 0.25,
                                    decay = 0.99,
                                    use_checkpoint=False)

# <---------------------------------------------------------------------------->
# SWIN-TRANSFORMERS WITH VECTOR QUNATIZER SEMANTIC-SEGMENTATION
# <---------------------------------------------------------------------------->

    elif model_name == "swin-transformer-vq-vae-ss":
        model = SwinTransformerVQVAE_SS( img_size=224,
                                    patch_size=4,
                                    in_chans=3,
                                    num_classes=6,
                                    embed_dim=96,
                                    depths=[2, 2, 6, 2],
                                    num_heads=[3, 6, 12, 24],
                                    window_size=7,
                                    mlp_ratio=4.,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    drop_rate=0.,
                                    attn_drop_rate=0.,
                                    drop_path_rate=0.1,
                                    norm_layer=nn.LayerNorm,
                                    ape=False,
                                    patch_norm=True,
                                    commitment_cost = 0.25,
                                    decay = 0.99,
                                    use_checkpoint=False)

# #######################################################################################################################################
# #######################################################################################################################################
# #######################################################################################################################################


# <---------------------------------------------------------------------------->
# MASWiTNet PRE-TRAINING
# <---------------------------------------------------------------------------->
    elif model_name == "MASWinTNet":
        model = MASWinTNet_Tiny_()
    elif model_name == "MASWinTNet-nano":
        model = MASWinTNet_Nano_()
    elif model_name == "MASWinTNet-vq":
        model = MASWinTNetVQ_Tiny_()
    elif model_name == "MASWinTNet-nano-vq":
        model = MASWinTNetVQ_Nano_()

# <---------------------------------------------------------------------------->
# MASWiTNet SEMANTIC-SEGMENTATION
# <---------------------------------------------------------------------------->


    elif model_name == "MASWinTNet-nano-ss":
        model = MASWinTNet_Nano_SS_()
    elif model_name == "MASWinTNet-nano-vq-ss":
        model = MASWinTNetVQ_Nano_SS_()

    elif model_name == "MASWinTNet-ss":
        model = MASWinTNet_Tiny_SS_()
    elif model_name == "MASWinTNet-vq-ss":
        model = MASWinTNetVQ_Tiny_SS_()

# <---------------------------------------------------------------------------->
# MASWiTNet SEMANTIC-SEGMENTATION WITH IMAGENET PRETRAIN
# <---------------------------------------------------------------------------->


    elif model_name == "MASWinTNet-nano-ss-pretrain-imagenet":
        model = MASWinTNet_Nano_SS_(pretrain = "imagenet")
    elif model_name == "MASWinTNet-nano-vq-ss-pretrain-imagenet":
        model = MASWinTNetVQ_Nano_SS_(pretrain = "imagenet")


    elif model_name == "MASWinTNet-ss-pretrain-imagenet":
        model = MASWinTNet_Tiny_SS_(pretrain = "imagenet", freeze_encoder=False)
    elif model_name == "MASWinTNet-vq-ss-pretrain-imagenet":
        model = MASWinTNetVQ_Tiny_SS_(pretrain = "imagenet", freeze_encoder=False)

# <---------------------------------------------------------------------------->
# MASWiTNet SEMANTIC-SEGMENTATION WITH EsViT PRETRAIN
# <---------------------------------------------------------------------------->


    elif model_name == "MASWinTNet-nano-ss-pretrain-EsViT":
        model = MASWinTNet_Nano_SS_(pretrain = "EsViT")
    elif model_name == "MASWinTNet-nano-vq-ss-pretrain-EsViT":
        model = MASWinTNetVQ_Nano_SS_(pretrain = "EsViT")

    elif model_name == "MASWinTNet-ss-pretrain-EsViT":
        model = MASWinTNet_Tiny_SS_(pretrain = "EsViT", freeze_encoder=False)
    elif model_name == "MASWinTNet-vq-ss-pretrain-EsViT":
        model = MASWinTNetVQ_Tiny_SS_(pretrain = "EsViT", freeze_encoder=False)

# <---------------------------------------------------------------------------->
# MASWiTNet SEMANTIC-SEGMENTATION WITH EsViT PRETRAIN
# <---------------------------------------------------------------------------->

    elif model_name == "SwinTUNet-ss":
        model = SWinTUNet_(pretrain = "None", freeze_encoder=False)
    elif model_name == "SwinTUNet-ss-pretrain-imagenet":
        model = SWinTUNet_(pretrain = "imagenet", freeze_encoder=False)
    elif model_name == "SwinTUNet-ss-pretrain-EsViT":
        model = SWinTUNet_(pretrain = "EsViT", freeze_encoder=False)


    elif model_name == "SwinTUNet-vq-ss":
        model = SWinTUNetVQ_(pretrain = "None", freeze_encoder=False)
    elif model_name == "SwinTUNet-vq-ss-pretrain-imagenet":
        model = SWinTUNetVQ_(pretrain = "imagenet", freeze_encoder=False)
    elif model_name == "SwinTUNet-vq-ss-pretrain-EsViT":
        model = SWinTUNetVQ_(pretrain = "EsViT", freeze_encoder=False)


# <---------------------------------------------------------------------------->
# SEMANTIC-SEGMENTATION WITH PRETRAIN AND FREEZED ENCODER
# <---------------------------------------------------------------------------->

    elif model_name == "SwinTUNet-ss-fe-pretrain-imagenet":
        model = SWinTUNet_(pretrain = "imagenet", freeze_encoder=True)
    elif model_name == "SwinTUNet-ss-fe-pretrain-EsViT":
        model = SWinTUNet_(pretrain = "EsViT", freeze_encoder=True)

    elif model_name == "SwinTUNet-vq-ss-fe-pretrain-imagenet":
        model = SWinTUNetVQ_(pretrain = "imagenet", freeze_encoder=True)
    elif model_name == "SwinTUNet-vq-ss-fe-pretrain-EsViT":
        model = SWinTUNetVQ_(pretrain = "EsViT", freeze_encoder=True)

    elif model_name == "MASWinTNet-ss-fe-pretrain-imagenet":
        model = MASWinTNet_Tiny_SS_(pretrain = "imagenet", freeze_encoder=True)
    elif model_name == "MASWinTNet-vq-ss-fe-pretrain-imagenet":
        model = MASWinTNetVQ_Tiny_SS_(pretrain = "imagenet", freeze_encoder=True)

    elif model_name == "MASWinTNet-ss-fe-pretrain-EsViT":
        model = MASWinTNet_Tiny_SS_(pretrain = "EsViT", freeze_encoder=True)
    elif model_name == "MASWinTNet-vq-ss-fe-pretrain-EsViT":
        model = MASWinTNetVQ_Tiny_SS_(pretrain = "EsViT", freeze_encoder=True)



    return model
