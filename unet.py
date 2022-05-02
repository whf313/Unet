import torch
from torch import nn
from torchvision import models


class UnetEncoder(nn.Module):
    def __init__(self, back_name):
        super(UnetEncoder, self).__init__()
        backbone = []
        backbone_raw = UnetEncoder.get_backbone(back_name=back_name)
        if back_name == "vgg16":
            for block in backbone_raw:
                backbone.append(UnetEncoder.add_bn(block))
        else:
            backbone = backbone_raw
        self.backbone = nn.Sequential(*backbone)
        
    def forward(self, x):
        y = []
        for block in self.backbone:
            x = block(x)
            y.append(x)
        y.reverse()
        return y

    @staticmethod
    def add_bn(sequential):
        output = []
        for layer in sequential:
            if type(layer) == nn.Conv2d:
                output.extend([layer, nn.BatchNorm2d(layer.out_channels)])
            else:
                output.append(layer)
        return nn.Sequential(*output)

    @staticmethod
    def get_backbone(back_name):
        backbone = []
        if back_name == "vgg16":
            vgg_model = models.vgg16(pretrained=True)
            features = list(vgg_model.children())[0]
            backbone = [features[0:4], features[4:9], features[9:16], features[16:23], features[23:-1]]
        elif back_name == "resnet34":
            res_model = models.resnet34(pretrained=True)
            stage_0 = nn.Sequential(*list(res_model.children())[0:4])
            backbone.append(stage_0)
            for layer in list(res_model.children())[4:8]:
                backbone.append(layer)
        else:
            raise ValueError(f"backbone {back_name} doesn't exist")
        return backbone


class UnetDecoderBlock(nn.Module):
    def __init__(self, channel, is_change=True, stride=2):
        """channel代表该bolck输出向量的通道数, is_change代表该block的FCN是否改变通道数, is_up代表通过转置卷积feature map是否增大"""
        super(UnetDecoderBlock, self).__init__()
        in_channel = 2 * channel if is_change else channel

        self.net1 = nn.Sequential(
            nn.ConvTranspose2d(in_channel, channel, kernel_size=2, stride=stride),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.attn = Attention_block(F_g=channel, F_l=channel, F_int=channel // 2)
        self.net2 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        
    def forward(self, x1, x2):
        """X1是上一个decoderblock传来的tensor，X2是从backbone传入的特征向量"""
        x = self.net1(x1)
        x2 = self.attn(x2, x)
        x = torch.cat([x2, x], dim=1)
        x = self.net2(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, g, x):
        # g为encoder传入的fm，x为decoder每次转置卷积后的上采样结果
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class UnetDecoder(nn.Module):
    Trans_info = {
        "vgg16": [
            (512, False, 2), (256, True, 2), (128, True, 2), (64, True, 2), 64
        ],
        "resnet34": [
            (256, True, 2), (128, True, 2), (64, True, 2), (64, False, 1), 32
        ]
    }

    def __init__(self, num_class, back_name):
        super().__init__()
        decode = []
        info, last_dim = UnetDecoder.Trans_info[back_name][:-1], UnetDecoder.Trans_info[back_name][-1]
        for channel, is_change, is_up in info:
            decode.append(UnetDecoderBlock(channel=channel, is_change=is_change, stride=is_up))

        decode.append(nn.Conv2d(last_dim, num_class, kernel_size=1))
        self.decoder = nn.Sequential(*decode)
    
    def forward(self, x):
        """X1~X5分别为encoder从最后一个sequential开始到第一个的输出"""
        temp = x[0]
        for block, fm in zip(self.decoder[:-1], x[1:]):
            temp = block(temp, fm)
        out = self.decoder[-1](temp)
        return out


class Unet(nn.Module):
    def __init__(self, backbone : str = "vgg16", num_class = 2):
        super().__init__()
        self.encoder = UnetEncoder(back_name=backbone)
        self.decoder = UnetDecoder(num_class=num_class, back_name=backbone)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


if __name__ == "__main__":
    u1 = Unet()
    x = torch.randn([8, 3, 224, 320])
    print(u1)
    y = u1(x)
    print(y.shape)

    # print("---" * 20)
    #
    # u2 = Unet(backbone="resnet34")
    # x = torch.randn([8, 3, 832, 1312])
    # y = u2(x)
    # print(y.shape)