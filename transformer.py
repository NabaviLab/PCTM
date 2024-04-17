
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from resnet import ResNetBlock
from utilities import ChannelAttention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src):
        return self.transformer_encoder(src)

class TwinResTransformer(nn.Module):
    def __init__(self):
        super(TwinResTransformer, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Blocks as per the given architecture table
        self.layer1 = self._make_layer(64, 256, 3, stride=1)
        self.layer2 = self._make_layer(256, 512, 4, stride=2)
        self.layer3 = self._make_layer(512, 1024, 6, stride=2)
        self.layer4 = self._make_layer(1024, 2048, 3, stride=2)

        self.transformer_encoder = TransformerEncoderBlock(d_model=2048, nhead=8, num_layers=2)
        self.channel_attention = ChannelAttention(in_planes=2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2 * 2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        
    def _make_layer(self, in_planes, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(ResNetBlock(self.in_planes, in_planes, planes, stride, downsample))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(self.in_planes, in_planes, planes))

        return nn.Sequential(*layers)

    def forward_one(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

    def forward(self, x1, x2):
        x1 = self.forward_one(x1)
        x2 = self.forward_one(x2)
        
        x1 = self.avgpool(x1)
        x2 = self.avgpool(x2)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        x = torch.cat((x1, x2), dim=1)
        
        # Positional encoding and transformer encoder
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x.unsqueeze(1))
        
        x = self.channel_attention(x.squeeze(1))
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        
        return x
