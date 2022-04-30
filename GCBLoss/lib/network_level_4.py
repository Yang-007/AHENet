import torch
from torch import nn
import sys
sys.path.append("..")
import config as cfg
from lib.RDB import RDB
import pdb


class DownBlock(nn.Module):
    def __init__(self, block_num=4, inter_channel=32, channel=64):
        super().__init__()

        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            RDB(block_num=block_num, inter_channel=inter_channel, channel=channel)
        )

    def forward(self, x):
        return self.layer(x)


class UpBlock(nn.Module):
    def __init__(self, block_num=4, inter_channel=32, channel=64):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            # nn.Conv2d(64 * 2, 64, 3, 1, 1),
            nn.Conv2d(channel * 2, channel, 1, 1, 0),
            nn.Dropout2d(p=0.1, inplace=True),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            RDB(block_num=block_num, inter_channel=inter_channel, channel=channel)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        return self.layer(torch.cat([x1, x2], 1))

class Network(nn.Module):
    def __init__(self, base_channel, inter_channel, rdb_block_num):
        super(Network, self).__init__()
        print('Using: Plain AHENet')
        print('BaseChannel:', base_channel)
        print('InterChannel:', inter_channel)
        print('RDB_block_num:', rdb_block_num)

        # self.cn_block = ColorNormBlock(in_c=3)
        self.head_conv = nn.Sequential(
            nn.Conv2d(3, base_channel, 3, 1, 1),
            nn.Dropout2d(p=0.1, inplace=True),
            nn.BatchNorm2d(base_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channel, base_channel, 3, 1, 1),
            nn.Dropout2d(p=0.1, inplace=True),
            nn.BatchNorm2d(base_channel),
            nn.ReLU(inplace=True),
        )

        self.down1 = DownBlock(block_num=rdb_block_num, inter_channel=inter_channel, channel=base_channel)
        self.down2 = DownBlock(block_num=rdb_block_num, inter_channel=inter_channel, channel=base_channel)
        self.down3 = DownBlock(block_num=rdb_block_num, inter_channel=inter_channel, channel=base_channel)

        self.mp4 = nn.MaxPool2d(2)
        self.down4 = RDB(block_num=rdb_block_num, inter_channel=inter_channel, channel=base_channel, has_GCB=True)

        self.up1 = UpBlock(block_num=rdb_block_num, inter_channel=inter_channel, channel=base_channel)
        self.up2 = UpBlock(block_num=rdb_block_num, inter_channel=inter_channel, channel=base_channel)
        self.up3 = UpBlock(block_num=rdb_block_num, inter_channel=inter_channel, channel=base_channel)
        self.up4 = UpBlock(block_num=rdb_block_num, inter_channel=inter_channel, channel=base_channel)
        self.tail_conv = nn.Conv2d(base_channel, 1, 1, 1, 0)
    
    def forward(self, x, label=None):
        # x = self.cn_block(x)
        x1 = self.head_conv(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5, loss_d4 = self.down4(self.mp4(x4), label)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.tail_conv(x), [loss_d4,]

if __name__ == '__main__':
    import time

    net = Network(64, 32, 4)
    net.eval()
    torch.set_grad_enabled(False)

    x = torch.randn(1, 3, 256, 256)
    y = net(x)

    print('Input Size:', x.size())
    print('Output Size:', y.size())


    