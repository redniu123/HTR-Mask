"""
HTR-Customized ResNet18 for Handwritten Text Recognition

This is NOT the standard torchvision ResNet18!
Key modifications for HTR:
1. Modified stride configuration to handle text line images
2. Asymmetric downsampling: compress Height aggressively, preserve Width longer
3. Output shape: (B, C, 1, W/4) for input (B, 1, H, W)

For input (B, 1, 64, 512) with nb_feat=768:
    Output: (B, 768, 1, 128)
    
Stride Analysis:
    Layer       | Stride    | H dimension      | W dimension
    ------------|-----------|------------------|------------------
    conv1       | (2, 1)    | 64 → 32          | 512 → 512
    maxpool     | (2, 1)    | 32 → 16          | 512 → 512
    layer1      | (2, 1)    | 16 → 8           | 512 → 512
    layer2      | (2, 2)    | 8 → 4            | 512 → 256
    layer3      | (2, 2)    | 4 → 2            | 256 → 128
    maxpool     | (2, 1)    | 2 → 1            | 128 → 128
    ------------|-----------|------------------|------------------
    Total       |           | 64× (64→1)       | 4× (512→128)
"""

import torch
import torch.nn as nn


def conv3x3(in_planes: int, out_planes: int, stride=1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34"""
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    """
    HTR-Customized ResNet18 Feature Extractor
    
    Args:
        nb_feat: Output feature dimension (default 384, use 768 for HTR-VT)
        
    Input: (B, 1, 64, 512) - grayscale text line image
    Output: (B, nb_feat, 1, 128) - feature sequence for ViT encoder
    """

    def __init__(self, nb_feat: int = 384):
        super(ResNet18, self).__init__()
        self.inplanes = nb_feat // 4
        
        # Initial conv: downsample H only (stride=(2,1))
        # Input: (B, 1, 64, 512) -> Output: (B, C/4, 32, 512)
        self.conv1 = nn.Conv2d(1, nb_feat // 4, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nb_feat // 4, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        
        # Maxpool: downsample H only (stride=(2,1))
        # (B, C/4, 32, 512) -> (B, C/4, 16, 512)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1)
        
        # layer1: downsample H only (stride=(2,1))
        # (B, C/4, 16, 512) -> (B, C/4, 8, 512)
        self.layer1 = self._make_layer(BasicBlock, nb_feat // 4, 2, stride=(2, 1))
        
        # layer2: downsample both H and W (stride=2)
        # (B, C/4, 8, 512) -> (B, C/2, 4, 256)
        self.layer2 = self._make_layer(BasicBlock, nb_feat // 2, 2, stride=2)
        
        # layer3: downsample both H and W (stride=2)
        # (B, C/2, 4, 256) -> (B, C, 2, 128)
        self.layer3 = self._make_layer(BasicBlock, nb_feat, 2, stride=2)

    def _make_layer(self, block, planes: int, blocks: int, stride=1) -> nn.Sequential:
        """Create a residual layer with specified number of blocks"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 1, 64, 512)
            
        Returns:
            Feature tensor (B, nb_feat, 1, 128)
        """
        # Stem: (B, 1, 64, 512) -> (B, C/4, 16, 512)
        x = self.conv1(x)   # (B, C/4, 32, 512)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # (B, C/4, 16, 512)

        # Residual layers
        x = self.layer1(x)  # (B, C/4, 8, 512)
        x = self.layer2(x)  # (B, C/2, 4, 256)
        x = self.layer3(x)  # (B, C, 2, 128)
        
        # Final pooling: collapse H to 1
        x = self.maxpool(x) # (B, C, 1, 128)
        
        return x




