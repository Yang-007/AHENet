import torch
from torch import nn
import torch.nn.functional as F


class GCBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_c, 1, 1, 1, 0)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_c, in_c // 2, 1, 1, 0),
            nn.LayerNorm(1),
            nn.ReLU(),
            nn.Conv2d(in_c // 2, in_c, 1, 1, 0),
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        weight = self.conv_1(x).view(b, h * w, 1)
        weight = torch.softmax(weight, 1)
        x_output = torch.matmul(x.view(b, c, h * w), weight).view(b, c, 1, 1)

        return self.conv_2(x_output) + x


class SEBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_c, in_c // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_c // 2, in_c, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, -1).mean(-1).view(b, c, 1, 1)
        return self.layers(x)


class RDB(nn.Module):
    def __init__(self, block_num=3, inter_channel=32, channel=64, has_GCB=False):
        super(RDB, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=channel + inter_channel * 0, out_channels=inter_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.1, inplace=True),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=channel + inter_channel * 1, out_channels=inter_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.1, inplace=True),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),

        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=channel + inter_channel * 2, out_channels=inter_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.1, inplace=True),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=channel + inter_channel * 3, out_channels=inter_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.1, inplace=True),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(channel + inter_channel * 4, channel, kernel_size=1, stride=1, padding=0),
            nn.Dropout2d(p=0.1, inplace=True),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )

        self.attention_2 = SEBlock(channel+inter_channel)
        self.attention_3 = SEBlock(channel+inter_channel * 2)
        self.attention_4 = SEBlock(channel+inter_channel * 3)
        self.attention_5 = SEBlock(channel+inter_channel * 4)

        if has_GCB:
            self.conv_1 = nn.Conv2d(channel, 1, 1, 1, 0)
            self.conv_2 = nn.Sequential(
                nn.Conv2d(channel, channel // 2, 1, 1, 0),
                nn.LayerNorm(1),
                nn.ReLU(),
                nn.Conv2d(channel // 2, channel, 1, 1, 0),
            )

    def forward(self, x, label=None):
        x1 = self.block_1(x)

        cat_2 = torch.cat([x, x1], 1)
        x2 = self.block_2(cat_2 * self.attention_2(cat_2))

        cat_3 = torch.cat([x, x1, x2], 1)
        x3 = self.block_3(cat_3 * self.attention_3(cat_3))

        cat_4 = torch.cat([x, x1, x2, x3], 1)
        x4 = self.block_4(cat_4 * self.attention_4(cat_4))

        cat_5 = torch.cat([x, x1, x2, x3, x4], dim=1)
        fusion_outputs = self.fusion(cat_5 * self.attention_5(cat_5))

        # -------------- GCB ----------------
        if label is not None:
            b, c, h, w = fusion_outputs.size()
            weight = self.conv_1(fusion_outputs).view(b, h * w, 1)
            weight = torch.softmax(weight, 1)
            fusion_outputs = torch.matmul(fusion_outputs.view(b, c, h * w), weight).view(b, c, 1, 1)

            weight = weight.view(b, 1, h, w)
            label = F.interpolate(label, size=(h, w), mode='nearest')

            label = label / (1e-7 + label.view(b, -1).sum(1).view(b, 1, 1, 1))
            weight = weight / (1e-7 + weight.view(b, -1).sum(1).view(b, 1, 1, 1))
            loss = (label * torch.log(1e-7 + label / (1e-7 + weight))).view(b, -1).mean(1)

            return fusion_outputs + x, loss
        return fusion_outputs + x

if __name__ == '__main__':


    net = RDB(3, 32, 64)


    x = torch.randn(1, 64, 64, 64)
    y = net(x)

    print('Input Size:', x.size())
    print('Output Size:', y.size())
