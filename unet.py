# import torch
# from torch import nn
# from torchvision import models
#
#
# class UnetEncoder(nn.Module):
#     def __init__(self, back_name):
#         super(UnetEncoder, self).__init__()
#         backbone = []
#         backbone_raw = UnetEncoder.get_backbone(back_name=back_name)
#         if back_name == "vgg16":
#             for block in backbone_raw:
#                 backbone.append(UnetEncoder.add_bn(block))
#         else:
#             backbone = backbone_raw
#         self.backbone = nn.Sequential(*backbone)
#
#     def forward(self, x):
#         y = []
#         for block in self.backbone:
#             x = block(x)
#             y.append(x)
#         y.reverse()
#         return y
#
#     @staticmethod
#     def add_bn(sequential):
#         output = []
#         for layer in sequential:
#             if type(layer) == nn.Conv2d:
#                 output.extend([layer, nn.BatchNorm2d(layer.out_channels)])
#             else:
#                 output.append(layer)
#         return nn.Sequential(*output)
#
#     @staticmethod
#     def get_backbone(back_name):
#         backbone = []
#         if back_name == "vgg16":
#             vgg_model = models.vgg16(pretrained=True)
#             features = list(vgg_model.children())[0]
#             backbone = [features[0:4], features[4:9], features[9:16], features[16:23], features[23:-1]]
#         elif back_name == "resnet34":
#             res_model = models.resnet34(pretrained=True)
#             stage_0 = nn.Sequential(*list(res_model.children())[0:4])
#             backbone.append(stage_0)
#             for layer in list(res_model.children())[4:8]:
#                 backbone.append(layer)
#         else:
#             raise ValueError(f"backbone {back_name} doesn't exist")
#         return backbone
#
#
# class UnetDecoderBlock(nn.Module):
#     def __init__(self, channel, is_change=True, stride=2):
#         """channel代表该bolck输出向量的通道数, is_change代表该block的FCN是否改变通道数, is_up代表通过转置卷积feature map是否增大"""
#         super(UnetDecoderBlock, self).__init__()
#         in_channel = 2 * channel if is_change else channel
#
#         self.net1 = nn.Sequential(
#             nn.ConvTranspose2d(in_channel, channel, kernel_size=2, stride=stride),
#             nn.BatchNorm2d(channel),
#             nn.ReLU()
#         )
#         self.attn = Attention_block(F_g=channel, F_l=channel, F_int=channel // 2)
#         self.net2 = nn.Sequential(
#             nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channel),
#             nn.ReLU(),
#             nn.Conv2d(channel, channel, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channel),
#             nn.ReLU(),
#         )
#
#     def forward(self, x1, x2):
#         """X1是上一个decoderblock传来的tensor，X2是从backbone传入的特征向量"""
#         x = self.net1(x1)
#         x2 = self.attn(x2, x)
#         x = torch.cat([x2, x], dim=1)
#         x = self.net2(x)
#         return x
#
#
# class Attention_block(nn.Module):
#     def __init__(self, F_g, F_l, F_int):
#         super(Attention_block, self).__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#
#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#
#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#
#         self.relu = nn.ReLU()
#
#     def forward(self, g, x):
#         # g为encoder传入的fm，x为decoder每次转置卷积后的上采样结果
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#
#         return x * psi
#
# class UnetDecoder(nn.Module):
#     Trans_info = {
#         "vgg16": [
#             (512, False, 2), (256, True, 2), (128, True, 2), (64, True, 2), 64
#         ],
#         "resnet34": [
#             (256, True, 2), (128, True, 2), (64, True, 2), (64, False, 1), 32
#         ]
#     }
#
#     def __init__(self, num_class, back_name):
#         super().__init__()
#         decode = []
#         info, last_dim = UnetDecoder.Trans_info[back_name][:-1], UnetDecoder.Trans_info[back_name][-1]
#         for channel, is_change, is_up in info:
#             decode.append(UnetDecoderBlock(channel=channel, is_change=is_change, stride=is_up))
#
#         decode.append(nn.Conv2d(last_dim, num_class, kernel_size=1))
#         self.decoder = nn.Sequential(*decode)
#
#     def forward(self, x):
#         """X1~X5分别为encoder从最后一个sequential开始到第一个的输出"""
#         temp = x[0]
#         for block, fm in zip(self.decoder[:-1], x[1:]):
#             temp = block(temp, fm)
#         out = self.decoder[-1](temp)
#         return out
#
#
# class Unet(nn.Module):
#     def __init__(self, backbone : str = "vgg16", num_class = 2):
#         super().__init__()
#         self.encoder = UnetEncoder(back_name=backbone)
#         self.decoder = UnetDecoder(num_class=num_class, back_name=backbone)
#
#     def forward(self, x):
#         x = self.encoder(x)
#         return self.decoder(x)
#
#
# if __name__ == "__main__":
#     u1 = Unet()
#     x = torch.randn([8, 3, 224, 320])
#     print(u1)
#     y = u1(x)
#     print(y.shape)
#
#     # print("---" * 20)
#     #
#     # u2 = Unet(backbone="resnet34")
#     # x = torch.randn([8, 3, 832, 1312])
#     # y = u2(x)
#     # print(y.shape)

import torch
import torch.nn.functional as F
import torch.nn as nn
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
        # y.reverse()           # reverse后，y[0]为　E_5的结果，y[-1]为E_1的结果, 方便Decoder处理
        # y = [x1, x2, x3, x4, x5]    xn为Encoder第n个block处理后的结果
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


class Up_Sample(nn.Module):
    def __init__(self, scale_factor, mode="bilinear"):
        super(Up_Sample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        return x


class UnetDecoderBlock(nn.Module):
    def __init__(self, num):
        """channel代表该bolck输出向量的通道数, is_change代表该block的FCN是否改变通道数, is_up代表通过转置卷积feature map是否增大"""
        # num从4开始到1
        super(UnetDecoderBlock, self).__init__()
        self.in_block = nn.Sequential(
            UnetDecoderBlock.get_layer(block_num=1, num=num),
            UnetDecoderBlock.get_layer(block_num=2, num=num),
            UnetDecoderBlock.get_layer(block_num=3, num=num),
            UnetDecoderBlock.get_layer(block_num=4, num=num),
            UnetDecoderBlock.get_layer(block_num=5, num=num)
        )
        self.out_block = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU()
        )

    @staticmethod
    def get_layer(block_num, num):
        # blcok_num为当前处理的是第几个encoder的输出结果，num为当前在创建哪个decoderblock
        scale = 2 ** abs(block_num - num)
        blcok = nn.Sequential()
        if block_num < num:
            # 如果当前处理的encoder编号小于num，说明需要进行maxpool
            blcok.add_module("maxpool", nn.MaxPool2d(kernel_size=scale, stride=scale))
            blcok.add_module("Conv2d",
                             nn.Conv2d(in_channels=64 * (2 ** (block_num - 1)), out_channels=64, kernel_size=3,
                                       padding=1))
        elif block_num > num:
            # 如果当前处理的encoder编号小于num，说明需要进行bilinear差值
            blcok.add_module("bilinear unsample", Up_Sample(scale_factor=scale))
            if block_num == 5:
                blcok.add_module("Conv2d", nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3, padding=1))
            else:
                blcok.add_module("Conv2d", nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1))
        else:
            blcok.add_module("Conv2d",
                             nn.Conv2d(in_channels=64 * (2 ** (block_num - 1)), out_channels=64, kernel_size=3,
                                       padding=1))

        return blcok

    def forward(self, x):
        """x_encoder是从backbone传入的特征向量(x1-x4), x_decoder是decoderblock传来的tensor(), x = x_encoder + x_decoder"""
        assert len(x) == 5
        out = []
        for block, item in zip(self.in_block, x):
            out.append(block(item))
        out = torch.cat(out, dim=1)
        assert out.shape[1] == 320
        out = self.out_block(out)
        return out


class UnetDecoder(nn.Module):
    # Trans_info = {
    #     "vgg16": [
    #         (512, False, 2), (256, True, 2), (128, True, 2), (64, True, 2), 64
    #     ],
    #     "resnet34": [
    #         (256, True, 2), (128, True, 2), (64, True, 2), (64, False, 1), 32
    #     ]
    # }

    Trans_info = {
        "vgg16": [
            4, 3, 2, 1, 320
        ],
        "resnet34": [
            (256, True, 2), (128, True, 2), (64, True, 2), (64, False, 1), 32
        ]
    }

    def __init__(self, num_class, back_name):
        super().__init__()
        decode = []
        info, last_dim = UnetDecoder.Trans_info[back_name][:-1], UnetDecoder.Trans_info[back_name][-1]
        for num in info:
            decode.append(UnetDecoderBlock(num=num))

        decode.append(nn.Conv2d(last_dim, num_class, kernel_size=1))
        self.decoder = nn.Sequential(*decode)

    def forward(self, x):
        """X1~X5分别为encoder从最后一个sequential开始到第一个的输出"""
        x_encoder = x  # 分别为 E_1到E_5的输出，长度为5
        x_decoder = [x[-1]]  # x[-1]为E_5的输出
        # decoder[-1]为用来调整通道数的1x1卷积
        for num, block in enumerate(self.decoder[:-1], 1):
            temp = block(x_encoder[:-num] + list(reversed(x_decoder[:num])))
            x_decoder.append(temp)
        out = self.decoder[-1](temp)
        return out


class Unet(nn.Module):
    def __init__(self, backbone: str = "vgg16", num_class=2):
        super().__init__()
        self.encoder = UnetEncoder(back_name=backbone)
        self.decoder = UnetDecoder(num_class=num_class, back_name=backbone)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


if __name__ == "__main__":
    u1 = Unet()
    print(u1)
    x = torch.randn([8, 3, 224, 320])
    y = u1(x)
    print(y.shape)

    # up = Up_Sample(16)
    # x = torch.randn([8, 3, 14, 20])
    # y = up(x)
    # print(y.shape)

    # print("---" * 20)
    #
    # u2 = Unet(backbone="resnet34")
    # x = torch.randn([8, 3, 832, 1312])
    # y = u2(x)
    # print(y.shape)