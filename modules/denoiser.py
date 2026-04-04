import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import matplotlib.pyplot as plt

# ResNet Block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

# Model
class EmbeddingDenoiseModel(nn.Module):
    def __init__(self):
        super(EmbeddingDenoiseModel, self).__init__()
        self.block1 = ResNetBlock(4, 32)
        self.block2 = ResNetBlock(32, 32)
        self.block3 = ResNetBlock(32, 32)
        self.block4 = ResNetBlock(32, 32)
        self.output_conv = nn.Conv3d(32, 1, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.block1(x)
        x = self.dropout(x)
        x = self.block2(x)
        x = self.dropout(x)
        x = self.block3(x)
        x = self.dropout(x)
        x = self.block4(x)
        x = self.dropout(x)
        x = self.output_conv(x)
        return x.squeeze(1)  # Remove channel dim
