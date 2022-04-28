import torch
from torch import nn
from torchvision import models
import os

class UnetEncoder(nn.Module):
    def __init__(self, backbone):
        super(UnetEncoder, self).__init__()
        self.bb1 = self.addBN(backbone[0:4])
        self.bb2 = self.addBN(backbone[4:9])
        self.bb3 = self.addBN(backbone[9:16])
        self.bb4 = self.addBN(backbone[16:23])
        self.bb5 = self.addBN(backbone[23:])
        
    def forward(self, X):
        y1 = self.bb1(X)
        y2 = self.bb2(y1)
        y3 = self.bb3(y2)
        y4 = self.bb4(y3)
        y5 = self.bb5(y4)
        return y5, y4, y3, y2, y1

    def addBN(self, Sequential):
        output = []
        for layer in Sequential:
            if type(layer) == nn.Conv2d:
                output.extend([layer, nn.BatchNorm2d(layer.out_channels)])
            else:
                output.append(layer)
        return nn.Sequential(*output)


class UnetDecoderBlock(nn.Module):
    def __init__(self, channel, is_change=True):
        """channel代表该bolck输出向量的通道数, is_change代表该block的FCN是否改变通道数"""
        super(UnetDecoderBlock, self).__init__()
        if is_change:
            self.Trans = nn.ConvTranspose2d(channel * 2, channel, kernel_size=2, stride=2)
        else:
            self.Trans = nn.ConvTranspose2d(channel, channel, kernel_size=2, stride=2)
        self.Conv1 = nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1)
        self.Conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        
    def forward(self, X1, X2):
        """X1是上一个decoderblock传来的tensor，X2是从backbone传入的特征向量"""
        X = self.Trans(X1)
        X = torch.cat([X2, X], dim=1)
        return self.Conv2(self.Conv1(X))

    @staticmethod
    def bilinear_kernel(in_channels, out_channels, kernel_size):
        """要求输入通道和输出通道数相同"""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = (torch.arange(kernel_size).reshape(-1, 1),torch.arange(kernel_size).reshape(1, -1))
        filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
        weight = torch.zeros((in_channels, out_channels,kernel_size, kernel_size))
        weight[range(in_channels), range(out_channels), :, :] = filt
        return weight


class UnetDecoder(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.Up1 = UnetDecoderBlock(512, False)
        self.Up2 = UnetDecoderBlock(256, True)
        self.Up3 = UnetDecoderBlock(128, True)
        self.Up4 = UnetDecoderBlock(64, True)
        self.fc = nn.Conv2d(64, num_class, kernel_size=1)
    
    def forward(self, X1, X2, X3, X4, X5):
        """X1~X5分别为encoder从最后一个sequential开始到第一个的输出"""
        X1 = self.Up1(X1, X2)
        X1 = self.Up2(X1, X3)
        X1 = self.Up3(X1, X4)
        X1 = self.Up4(X1, X5)
        X = self.fc(X1)
        return X


class Unet(nn.Module):
    def __init__(self, num_class = 2, backbone : str = "vgg16"):
        super().__init__()
        backbone_model = Unet.get_backbone(backbone)
        self.encoder = UnetEncoder(backbone_model)
        self.decoder = UnetDecoder(num_class)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(*x)

    @staticmethod
    def get_backbone(back_name):
        if back_name == "vgg16":
            vgg_model = models.vgg16(pretrained=True)
            backbone = nn.Sequential(*list(vgg_model.children())[0][:-1])
        else:
            raise ValueError(f"backbone {back_name} doesn't exist")

        return backbone

if __name__ == "__main__":
    unet = Unet()
    x = torch.randn([8, 3, 704, 1216])
    y = unet(x)

    print(y.shape)