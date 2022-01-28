# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

# =====================
# === DISCRIMINATOR ===
# =====================

class CNNBlock(nn.Module):
    '''
    Convolution Block.
    According to the paper: Conv2D -> BatchNorm2D -> LeakyReLU
    '''
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)



# Input of the discriminator is RGB image
# Input: 256x256 -> 30x30
# For the discriminator we also used the target image according to the paper
class Discriminator(nn.Module):
    '''
    Discriminator model.
    This model has to determine if the input image is a fake one of a real one.
    For the initial Conv they don't use BatchNorm2D.
    '''
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        # For the first block they don't use batch norm
        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            nb_stride = 1 if feature == features[-1] else 2
            layers.append(CNNBlock(in_channels, feature, stride=nb_stride))

            in_channels = feature

        # output layer
        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'),
        )

        self.model = nn.Sequential(*layers)

    # y fake or real
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial_block(x)
        return self.model(x)


# =====================
# ===== GENERATOR =====
# =====================

# Generator model
class UNetBlock(nn.Module):
    '''
    One Conv block for UNet model (wether downsampling and upsampling)
    Upsampling is done using Conv 2D Transpose
    '''
    def __init__(self, in_channels, out_channels, down=True, act='relu', use_dropout=False):
        super().__init__()

        layers = []

        if down:
            layers.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode='reflect'))
        else:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False))

        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2))

        if use_dropout:
            layers.append(nn.Dropout(0.5))

        self.layers_pred = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers_pred(x)

class Generator(nn.Module):
    '''
    Generator model with UNet blocks (Encoder / Decoder)
    '''
    def __init__(self, in_channels=3, features=64):
        super().__init__()

        # Encoder
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        ) # 128

        self.down1 = UNetBlock(features, 2*features, down=True, act='leaky', use_dropout=False) # 64
        self.down2 = UNetBlock(2*features, 4*features, down=True, act='leaky', use_dropout=False) # 32
        self.down3 = UNetBlock(4*features, 8*features, down=True, act='leaky', use_dropout=False) # 16
        self.down4 = UNetBlock(8*features, 8*features, down=True, act='leaky', use_dropout=False) # 8
        self.down5 = UNetBlock(8*features, 8*features, down=True, act='leaky', use_dropout=False) # 4
        self.down6 = UNetBlock(8*features, 8*features, down=True, act='leaky', use_dropout=False) # 2

        # BottleNeck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(8*features, 8*features, 4, 2, 1, padding_mode='reflect'),  # 1
            nn.ReLU()
        )

        # Decoder
        # Twice more input channels because we use residual learning
        self.up1 = UNetBlock(8*features, 8*features, down=False, act='relu', use_dropout=True)
        self.up2 = UNetBlock(16*features, 8*features, down=False, act='relu', use_dropout=True)
        self.up3 = UNetBlock(16*features, 8*features, down=False, act='relu', use_dropout=True)
        self.up4 = UNetBlock(16*features, 8*features, down=False, act='relu', use_dropout=True)
        self.up5 = UNetBlock(16*features, 4*features, down=False, act='relu', use_dropout=True)
        self.up6 = UNetBlock(8*features, 2*features, down=False, act='relu', use_dropout=True)
        self.up7 = UNetBlock(4*features, features, down=False, act='relu', use_dropout=True)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(2*features, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # Encoder
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)

        # Bottleneck
        bottleneck = self.bottleneck(d7)

        # Decoder
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], dim=1))
        up3 = self.up3(torch.cat([up2, d6], dim=1))
        up4 = self.up4(torch.cat([up3, d5], dim=1))
        up5 = self.up5(torch.cat([up4, d4], dim=1))
        up6 = self.up6(torch.cat([up5, d3], dim=1))
        up7 = self.up7(torch.cat([up6, d2], dim=1))

        # Final layer
        return self.final_layer(torch.cat([up7, d1], dim=1))