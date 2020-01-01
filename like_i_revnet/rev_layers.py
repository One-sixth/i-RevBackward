'''
定义了一些可逆层

pytorch v1.3.1 Bug 注意
你可以在 https://github.com/pytorch/pytorch/issues/31748 跟踪这个bug处理
输入参数 pad 全为0的 F.pad 会导致我的自定义反传函数报错，目前处理方法是预先判断 pad 是否全为0，若全为 0 则跳过执行 F.pad 函数
y = F.pad(x, pad=[0, 0, 0, 0, 0, 0], mode=self.mode, value=self.value)


'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, List, Tuple


def pixelshuffle(x: torch.Tensor, factor_hw: Tuple[int, int]):
    pH = factor_hw[0]
    pW = factor_hw[1]
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC//(pH*pW), iH*pH, iW*pW
    y = y.reshape(B, oC, pH, pW, iH, iW)
    y = y.permute(0, 1, 4, 2, 5, 3)     # B, oC, iH, pH, iW, pW
    y = y.reshape(B, oC, oH, oW)
    return y


def pixelshuffle_invert(x: torch.Tensor, factor_hw: Tuple[int, int]):
    pH = factor_hw[0]
    pW = factor_hw[1]
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC*(pH*pW), iH//pH, iW//pW
    y = y.reshape(B, iC, oH, pH, oW, pW)
    y = y.permute(0, 1, 3, 5, 2, 4)     # B, iC, pH, pW, oH, oW
    y = y.reshape(B, oC, oH, oW)
    return y


class RevIdenity(nn.Module):
    '''
    不干任何事情的可逆层
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def invert(self, x):
        return x


class RevChannelPad2D(nn.Module):
    '''
    只支持第一维度的通道填充层
    '''
    def __init__(self, pad_size: int, mode='constant', value=0):
        super().__init__()
        assert pad_size >= 0
        self.pad_size = pad_size
        self.mode = mode
        self.value = value

    def forward(self, x):
        # because pytorch v1.3.1 bug, cannot use this. we need to check pad_size
        # y = F.pad(x, pad=[0, 0, 0, 0, 0, self.pad_size], mode=self.mode, value=self.value)
        if self.pad_size != 0:
            y = F.pad(x, pad=[0, 0, 0, 0, 0, self.pad_size], mode=self.mode, value=self.value)
        else:
            y = x
        return y

    def invert(self, y):
        return y[:, :y.shape[1] - self.pad_size, :, :]


class RevPad2D(nn.Module):
    '''
    支持多个通道的Pad层，输入请参考 F.pad
    '''
    def __init__(self, pad: List, mode='constant', value=0):
        super().__init__()
        assert isinstance(pad, Iterable)
        assert len(pad) > 0 and len(pad) % 2 == 0
        pad = list(pad)
        pad += [0] * (8 - len(pad))
        unpad = np.array(pad).reshape([-1, 2])[::-1].flatten().tolist()

        self.pad = pad
        self.unpad = unpad
        self.mode = mode
        self.value = value

    def forward(self, x):
        # 注意，F.pad bug。Bug相关解释请看本文件的头部
        if sum(self.pad) != 0:
            y = F.pad(x, pad=self.pad, mode=self.mode, value=self.value)
        else:
            y = x
        return y

    def invert(self, y):
        return y[self.unpad[0]: y.shape[0]-self.unpad[1],
                 self.unpad[2]: y.shape[1]-self.unpad[3],
                 self.unpad[4]: y.shape[2]-self.unpad[5],
                 self.unpad[6]: y.shape[3]-self.unpad[7]]


class RevPS_Inverse(nn.Module):
    '''
    这里是逆向 PixelShuffle 层，一般用作 RevNet 的下采样层
    '''
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        return pixelshuffle_invert(x, (self.block_size, self.block_size))

    def invert(self, y):
        return pixelshuffle(y, (self.block_size, self.block_size))


class RevPS(nn.Module):
    '''
    这里是 PixelShuffle 层，一般用作 RevNet 的上采样层
    '''
    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = block_size

    def forward(self, x: torch.Tensor):
        return pixelshuffle(x, (self.block_size, self.block_size))

    def invert(self, y: torch.Tensor):
        return pixelshuffle_invert(y, (self.block_size, self.block_size))


if __name__ == '__main__':

    # check RevChannelPad
    pad = RevChannelPad2D(5)
    x = torch.rand(3, 5, 6, 6)
    y = pad(x)
    rx = pad.invert(y)
    assert torch.allclose(x, rx)

    # check RevPad2D
    pad = RevPad2D([1, 1, 2, 2, 3, 3])
    x = torch.rand(6, 6, 6, 6)
    y = pad(x)
    assert y.shape == (6, 12, 10, 8)
    rx = pad.invert(y)
    assert torch.allclose(x, rx)

    # check RevPS_Inverse
    psi = RevPS_Inverse(2)
    x = torch.arange(16).reshape(1, 1, 4, 4)
    y = psi(x)
    assert y.shape == (1, 4, 2, 2)
    assert np.all(y[0, :, 0, 0].numpy().astype(np.int) == np.array([0, 1, 4, 5]))
    rx = psi.invert(y)
    assert torch.allclose(x, rx)

    # check RevPS
    ps = RevPS(2)
    x = torch.arange(16).reshape(1, 4, 2, 2)
    y = ps(x)
    assert y.shape == (1, 1, 4, 4)
    # print(y[0, :, 0, 1])
    rx = ps.invert(y)
    assert torch.allclose(x, rx)
