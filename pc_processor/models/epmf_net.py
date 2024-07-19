# !/usr/bin/env python3
# source code for PMFNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from .salsanext import SalsaNext
from .pmf_net import ResidualBasedFusionBlock, ASPP, ResNet

class SparseVariantConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, groups=1, dilation=1, bias=True):
        super(SparseVariantConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, padding=padding, 
            stride=stride, groups=groups, dilation=dilation)
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=0, dilation=dilation)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).float())
        else:
            self.bias = None
        self._init_weight()

    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")

    def forward(self, x, mask):
        x = x * mask
        with torch.no_grad():
            # compute normalize value
            ones_w = torch.ones_like(self.conv.weight)
            mask_conv = F.conv2d(
                mask.expand_as(x), ones_w, 
                bias=None, stride=self.conv.stride,
                padding=self.conv.padding, groups=self.conv.groups, 
                dilation=self.conv.dilation)
            mask_conv = 1. / torch.clamp(mask_conv, min=1e-5)
            # compute dialted mask
            mask = self.pool(F.pad(mask, 
                (self.conv.padding[1], self.conv.padding[1], self.conv.padding[0], self.conv.padding[0])))

        x = self.conv(x)
        if self.bias is not None:
            x = x + self.bias.view(1, self.bias.size(0), 1, 1).expand_as(x)
        x = x * mask
        
        return x, mask

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, stride=1):
        super(ResContextBlock, self).__init__()
        self.conv1 = SparseVariantConv(in_filters, out_filters, 3, padding=1, stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = SparseVariantConv(out_filters, out_filters, (3, 3), padding=(1, 1))
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = SparseVariantConv(out_filters, out_filters, (3, 3), padding=(2, 2), dilation=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        mask = x.abs().sum(1).ne(0).float().unsqueeze(1)
        shortcut, mask = self.conv1(x, mask)
        shortcut = self.act1(shortcut)

        resA, mask = self.conv2(shortcut, mask)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA, mask = self.conv3(resA1, mask)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)
        output = shortcut + resA2
        output = output*mask
        return output
    
class SalsaNextFusion(SalsaNext):
    def __init__(self, in_channels=8, nclasses=20, base_channels=32, img_feature_channels=[]):
        super(SalsaNextFusion, self).__init__(in_channels=in_channels, base_channels=base_channels,
                                              nclasses=nclasses, softmax=True)
    
        self.downCntx = ResContextBlock(in_channels, base_channels)
        self.downCntx2 = ResContextBlock(base_channels, base_channels)
        self.downCntx3 = ResContextBlock(base_channels, base_channels, stride=2)
        self.fusionblock_1 = ResidualBasedFusionBlock(self.base_channels*1, img_feature_channels[0])
        self.fusionblock_2 = ResidualBasedFusionBlock(self.base_channels*2, img_feature_channels[1])
        self.fusionblock_3 = ResidualBasedFusionBlock(self.base_channels*4, img_feature_channels[2])
        self.fusionblock_4 = ResidualBasedFusionBlock(self.base_channels*8, img_feature_channels[3])

        self.aspp = ASPP(self.base_channels * 8, self.base_channels * 8)

        self.extraUpSample = nn.Sequential(
            nn.Conv2d(base_channels, 4*base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*base_channels),
            nn.PixelShuffle(2)
        )

    def forward(self, x, img_feature=[]):
        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        downCntx = self.fusionblock_1(downCntx, img_feature[0])
        down0c, down0b = self.resBlock1(downCntx)

        down0c = self.fusionblock_2(down0c, img_feature[1])
        down1c, down1b = self.resBlock2(down0c)
        
        down1c = self.fusionblock_3(down1c, img_feature[2])
        down2c, down2b = self.resBlock3(down1c)
        
        down2c = self.fusionblock_4(down2c, img_feature[3])
        down3c, down3b = self.resBlock4(down2c)

        down5c = self.aspp(self.resBlock5(down3c))

        up4e = self.upBlock1(down5c, down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)
        up1e = self.extraUpSample(up1e)
        logits = self.logits(up1e)
        if self.softmax:
            logits = F.softmax(logits, dim=1)
        return logits, down5c


class RGBDecoder(nn.Module):
    def __init__(self, in_channels=[], nclasses=4, base_channels=64, lidar_base_channels=32):
        super(RGBDecoder, self).__init__()

        self.aspp = ASPP(in_channels[3], in_channels[3])
        self.extraUpSample = nn.Sequential(
            nn.Conv2d(lidar_base_channels*8, lidar_base_channels*8, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(lidar_base_channels*8),
            nn.PixelShuffle(2)
        )

        self.up_4a = nn.Sequential(
            nn.Conv2d(in_channels[3] + lidar_base_channels*2, base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )

        
        self.up_3a = nn.Sequential(
            nn.Conv2d(in_channels[2] + base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear")

        )
        self.up_2a = nn.Sequential(
            nn.Conv2d(in_channels[1] + base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.up_1a = nn.Sequential(
            nn.Conv2d(in_channels[0] + base_channels, base_channels, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.conv = nn.Conv2d(base_channels, nclasses, kernel_size=3, padding=1)

    def forward(self, inputs, lidar_feature):
        fuse_feature = torch.cat((self.extraUpSample(lidar_feature), self.aspp(inputs[3])), dim=1)
        up_4a = self.up_4a(fuse_feature)
        up_3a = self.up_3a(torch.cat((up_4a, inputs[2]), dim=1))
        up_2a = self.up_2a(torch.cat((up_3a, inputs[1]), dim=1))
        up_1a = self.up_1a(torch.cat((up_2a, inputs[0]), dim=1))
        out = self.conv(up_1a)
        out = F.softmax(out, dim=1)
        return out

class EPMFNet(nn.Module):
    def __init__(self, pcd_channels=5, img_channels=3, nclasses=20, base_channels=32, 
                 imagenet_pretrained=True, image_backbone="resnet34"):
        super(EPMFNet, self).__init__()
        
        if "resnet" in image_backbone:
            self.camera_stream_encoder = ResNet(
                in_channels=img_channels,
                pretrained=imagenet_pretrained,
                backbone=image_backbone)
        else:
            raise NotImplementedError(image_backbone)
        
        self.camera_stream_decoder = RGBDecoder(
            self.camera_stream_encoder.feature_channels, 
            nclasses=nclasses, 
            base_channels=self.camera_stream_encoder.expansion*16,
            lidar_base_channels=base_channels)


        self.lidar_stream = SalsaNextFusion(
            in_channels=pcd_channels, nclasses=nclasses, base_channels=base_channels,
            img_feature_channels=self.camera_stream_encoder.feature_channels)

    def forward(self, pcd_feature, img_feature):

        img_mid_feature = self.camera_stream_encoder(img_feature)
        lidar_pred, lidar_feature = self.lidar_stream(pcd_feature, img_mid_feature)

        camera_pred = self.camera_stream_decoder(img_mid_feature, lidar_feature)

        return lidar_pred, camera_pred

if __name__ == "__main__":
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    model = EPMFNet(image_backbone="resnet34").cuda()
    print(model)
    test_pcd = torch.ones((1, 5, 640, 640)).cuda()
    test_rgb = torch.ones((1, 3, 640, 640)).cuda()
    pred, pred_rgb = model(test_pcd, test_rgb)
    print(pred.size(), pred_rgb.size())