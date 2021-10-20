import torch 
import torch.nn as nn

# upsample layer to replace deconv
class ConvUpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, scale_factor=2, mode="nearest"):
        super(ConvUpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        return self.conv(self.upsample(x))

# channel-wise spatial attention module
class CSAttention(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, scale=1.0):
        super(CSAttention, self).__init__()

        middle_channels = int(in_channels*scale)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, stride=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, stride=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out_conv = self.conv(x)
        out_att = self.attention(x)
        return out_conv*out_att


