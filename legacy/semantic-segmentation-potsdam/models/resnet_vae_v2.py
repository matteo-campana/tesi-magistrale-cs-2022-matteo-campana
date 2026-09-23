import torch
import torch.nn as nn

# <---------------------------------------------------------------------------->
# RESNET ENCODER BLOCK
# <---------------------------------------------------------------------------->

class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

# <---------------------------------------------------------------------------->
# RESNET ENCODER
# <---------------------------------------------------------------------------->

class ResNetEncoder(nn.Module):
    def __init__(self, block, layers, image_channels):#, num_classes):
        super(ResNetEncoder, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

# <---------------------------------------------------------------------------->
# RESNET DECODER BLOCK
# <---------------------------------------------------------------------------->


class decoder_block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(decoder_block, self).__init__()
        self.expansion = 4

        self.conv1 = nn.ConvTranspose2d(
            in_channels,
            intermediate_channels // self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels // self.expansion)

        if stride > 1:
            self.conv2 = nn.ConvTranspose2d(
                intermediate_channels // self.expansion,
                intermediate_channels // self.expansion,
                kernel_size=3,
                stride=stride,
                padding=1,
                output_padding = 1,
                bias=False
            )
        else:
            self.conv2 = nn.ConvTranspose2d(
                intermediate_channels // self.expansion,
                intermediate_channels // self.expansion,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            )

        self.bn2 = nn.BatchNorm2d(intermediate_channels // self.expansion)

        self.conv3 = nn.ConvTranspose2d(
            intermediate_channels // self.expansion,
            intermediate_channels // self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels // self.expansion)

        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

# <---------------------------------------------------------------------------->
# RESNET DECODER
# <---------------------------------------------------------------------------->

class ResNetDecoder(nn.Module):
    def __init__(self, decoder_block, layers, image_channels):
        super(ResNetDecoder, self).__init__()
        self.in_channels = 2048

        self.conv1 = nn.ConvTranspose2d(64, image_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(image_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            decoder_block, layers[0], intermediate_channels=2048, stride=2
        )
        self.layer2 = self._make_layer(
            decoder_block, layers[1], intermediate_channels=1024, stride=2
        )
        self.layer3 = self._make_layer(
            decoder_block, layers[2], intermediate_channels=512, stride=2
        )
        self.layer4 = self._make_layer(
            decoder_block, layers[3], intermediate_channels=256, stride=2
        )

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        return x

    def _make_layer(self, decoder_block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels // 4:
            identity_downsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.in_channels,
                    intermediate_channels // 4,
                    kernel_size=1,
                    stride=stride,
                    output_padding = 1,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels // 4),
            )

        layers.append(
            decoder_block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels // 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(decoder_block(self.in_channels, intermediate_channels,identity_downsample=None, stride=1))

        return nn.Sequential(*layers)


# <---------------------------------------------------------------------------->
# RESNET VARIATIONAL AUTO ENCODER FULL MODEL
# <---------------------------------------------------------------------------->

class ResNetVAE(nn.Module):
    def __init__(self, block ,decoder_block, layers, image_channels):
        super(ResNetVAE, self).__init__()

        self.encoder = ResNetEncoder(block, layers, image_channels)
        self.decoder = ResNetDecoder(decoder_block, layers, image_channels)

        self.net = nn.Sequential(
            self.encoder,
            self.decoder
        )

    def forward(self, x):
        x = self.net(x)

        return x

# <---------------------------------------------------------------------------->
# RESNET VARIATIONAL AUTO ENCODER FULL MODEL SEMANTIC SEGMENTATION
# <---------------------------------------------------------------------------->


class ResNetVAE_SS(nn.Module):
    def __init__(self, block ,decoder_block, layers, image_channels):
        super(ResNetVAE_SS, self).__init__()

        self.encoder = ResNetEncoder(block, layers, image_channels)
        self.decoder = ResNetDecoder(decoder_block, layers, image_channels)
        self.proj    = nn.ConvTranspose2d(image_channels,6, kernel_size=1, stride=1, padding=0, bias=False)
        self.finalbn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.proj(x)
        x = self.finalbn(x)
        x = self.relu(x)

        return x

# <---------------------------------------------------------------------------->
# NETWORK BUILDER PRE-TRAINING
# <---------------------------------------------------------------------------->

def ResNet50VAE(img_channel=3):
    return ResNetVAE(block, decoder_block, [3, 6, 4, 3], img_channel)

def ResNet101VAE(img_channel=3):
    return ResNetVAE(block, decoder_block, [3, 23, 4, 3], img_channel)

def ResNet152VAE(img_channel=3):
    return ResNetVAE(block, decoder_block, [3, 36, 8, 3], img_channel)


# <---------------------------------------------------------------------------->
# NETWORK BUILDER SEMANTIC SEGMENTATION
# <---------------------------------------------------------------------------->

def ResNet50VAE_SS(img_channel=3):
    return ResNetVAE_SS(block, decoder_block, [3, 6, 4, 3], img_channel)

def ResNet101VAE_SS(img_channel=3):
    return ResNetVAE_SS(block, decoder_block, [3, 23, 4, 3], img_channel)

def ResNet152VAE_SS(img_channel=3):
    return ResNetVAE_SS(block, decoder_block, [3, 36, 8, 3], img_channel)
