'''
这里提供了俩个示例用来演示如何写RevBlock
模块中 forward 和 invert 都必须要有定义，并且要有俩个输入和输出
invert 是 forward 的逆函数

输出时交换 y 和 x2 的位置是为了能让 y 也通过 F 进行处理，交换不是必须的，但建议交换位置

可逆原理
forward 时
输入 x1, x2
y = x1 + F(x2)
输出 y, x2

invert 时
输入 y，x2
x1 = y - F(x2)
输出 x1, x2

示例模块
SimpleRevBlock      最简单的可逆模块
SimpleRevBlock2     加入了下采样的可逆模块，注意，例如下采样倍数为2，则输出通道数至少输入通道的4倍

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .rev_layers import RevChannelPad2D, RevPS_Inverse, RevIdenity


class SimpleRevBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.func = nn.Sequential(
            nn.Conv2d(20, 20, 3, 1, 1),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(0.2),
            nn.Conv2d(20, 20, 3, 1, 1),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x1, x2):
        y = x1 + self.func(x2)
        return x2, y

    def invert(self, y1, y2):
        x2, y = y1, y2
        x1 = y - self.func(x2)
        return x1, x2


class SimpleRevBlock2(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act):
        super().__init__()
        assert stride in {1, 2}
        assert (out_ch >= in_ch) if stride == 1 else (out_ch >= in_ch * stride**2 and out_ch % stride**2 == 0)
        self.stride = stride

        if stride >= 2:
            self.ds = RevPS_Inverse(stride)
            self.pad = RevChannelPad2D(out_ch // stride ** 2 - in_ch)
        elif out_ch > in_ch:
            self.ds = RevIdenity()
            self.pad = RevChannelPad2D(out_ch - in_ch)
        else:
            self.ds = RevIdenity()
            self.pad = RevIdenity()

        self.func = nn.Sequential(
            nn.Conv2d(out_ch, out_ch//2, 3, 1, 1),
            nn.BatchNorm2d(out_ch//2),
            act,
            nn.Conv2d(out_ch//2, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            act
        )

    def forward(self, x1, x2):
        x1 = self.pad(x1)
        x1 = self.ds(x1)
        x2 = self.pad(x2)
        x2 = self.ds(x2)
        y = x1 + self.func(x2)
        return x2, y

    def invert(self, y1, y2):
        x2, y = y1, y2
        x1 = y - self.func(x2)
        x1 = self.ds.invert(x1)
        x1 = self.pad.invert(x1)
        x2 = self.ds.invert(x2)
        x2 = self.pad.invert(x2)
        return x1, x2


class RevSequential(nn.ModuleList):
    '''
    功能大部分与ModuleList重叠
    '''
    def __init__(self, modules=None):
        super().__init__(modules)

    def append(self, module):
        assert hasattr(module, 'invert') and callable(module.invert)
        super().append(module)

    def extend(self, modules):
        for m in modules:
            self.append(m)

    def forward(self, x1, x2):
        y1, y2 = x1, x2
        for m in self:
            y1, y2 = m(y1, y2)
        return y1, y2

    def invert(self, y1, y2):
        x1, x2 = y1, y2
        for m in list(self)[::-1]:
            x1, x2 = m.invert(x1, x2)
        return x1, x2


class RevGroupBlock(RevSequential):
    '''
    当前只支持输入通道等于输出通道，并且不允许下采样
    '''
    def __init__(self, in_ch, out_ch, stride, act, block_type, blocks, **kwargs):
        assert in_ch == out_ch
        assert stride == 1
        mods = []
        for _ in range(blocks):
            mods.append(block_type(in_ch=in_ch, out_ch=out_ch, stride=1, act=act, **kwargs))
        # self.extend(mods)
        super().__init__(mods)
