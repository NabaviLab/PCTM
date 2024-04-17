
import torch
import torch.nn as nn

# Channel Attention Module, often used in CNN architectures to enhance feature representation
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Adaptive average pooling to squeeze spatial dimensions
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Fully connected layers to learn channel-wise dependencies
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu1(y)
        y = self.fc2(y)
        return x * self.sigmoid(y)
