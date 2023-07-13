import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    Module of the Convolutional Block used in UNet
    Two Seccessive 3x3 convolutional layers with ReLU
    activation on each

    Incorporates batch normalization to provide network
    stability while training

    Adding padding so that output feature maps match input feature
    maps. Prevents nececity of cropping during skip connections
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, padding=1
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.batch_norm1(x))

        x = self.conv2(x)
        x = self.relu (self.batch_norm2(x))
        return x


class EncoderBlock(nn.Module):
    """
    Module of an Encoder Block used in UNet
    Following a ConvBlock, 2x2 Max Pool (with
    stride of 2) is used to further downsample the 
    feature maps

    Forward method returns both pooled and unpooled
    pass of the ConvBlock, with the unpooled output
    being used for the skip connections
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBlock(
            in_channels=in_channels, out_channels=out_channels
        )
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, input):
        no_pool = self.conv_block(input)
        pool = self.pool(no_pool)
        return pool, no_pool
        

class DecoderBlock(nn.Module):
    """
    Module of a Decoder Block used in UNet
    Following a ConvBlock, 2x2 Up Convolution
    (with stride of 2) is used to upsample
    the feature maps from the latent space
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBlock(
            in_channels=in_channels, out_channels=out_channels
        )
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=2, stride=2, padding=0
        )
    
    def forward(self, input, skip):
        x = self.up(input)
        x = torch.cat([x, skip], axis=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    """
    UNet module incorporating contracting path
    (comprised of EncoderBlock objects), latent space
    EncoderBlock, and expanding path (comprised of
    DecoderBlock objects)
    Incorporates skip connections
    """
    def __init__(self):
        super().__init__()
        self.encode1 = EncoderBlock(3, 64)
        self.encode2 = EncoderBlock(64, 128)
        self.encode3 = EncoderBlock(128, 256)
        self.encode4 = EncoderBlock(256, 512)
        
        self.latent = ConvBlock(512, 1024)

        self.decode4 = DecoderBlock(1024, 512)
        self.decode3 = DecoderBlock(512, 256)
        self.decode2 = DecoderBlock(256, 128)
        self.decode1 = DecoderBlock(128, 64)

        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, input):
        """
        Encoder 
        """
        pool1, no_pool1 = self.encode1(input)
        pool2, no_pool2 = self.encode2(pool1)
        pool3, no_pool3 = self.encode3(pool2)
        pool4, no_pool4 = self.encode4(pool3)

        # Latent space (deepest part of network)
        latent = self.latent(pool4)

        """
        Decode (with skip connections, the no_pool objects)
        """
        decode = self.decode4(latent, no_pool4)
        decode = self.decode3(decode, no_pool3)
        decode = self.decode2(decode, no_pool2)
        decode = self.decode1(decode, no_pool1)

        output = self.output(decode)

        return output

