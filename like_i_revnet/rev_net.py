'''
这是一个示例网络
演示如何使用 rev block 节省显存 和 如何在可逆块之间穿插不可逆模块

可逆模块堆叠的次数越多，节省的显存相比普通情况下越可观
'''

import torch
import torch.nn as nn
from .rev_blocks import RevSequential, SimpleRevBlock2
from .rev_utils import rev_sequential_backward_wrapper


class Like_IRevNet(nn.Module):
    def __init__(self, use_rev_bw):
        '''
        use_rev_bw 是否使用 rev_bw 反传模块
        '''
        super().__init__()
        self.use_rev_bw = use_rev_bw

        act = nn.LeakyReLU(0.02)
        self.seq1 = RevSequential([
            SimpleRevBlock2(3, 12, 1, act),          # 32
            SimpleRevBlock2(12, 12, 1, act),
            SimpleRevBlock2(12, 12, 1, act),
            SimpleRevBlock2(12, 12, 1, act),
            SimpleRevBlock2(12, 48, 2, act),         # 16
            SimpleRevBlock2(48, 48, 1, act),
            SimpleRevBlock2(48, 48, 1, act),
            SimpleRevBlock2(48, 48, 1, act),
            SimpleRevBlock2(48, 48, 1, act),
            SimpleRevBlock2(48, 192, 2, act),       # 8
            SimpleRevBlock2(192, 192, 1, act),
            SimpleRevBlock2(192, 192, 1, act),
            SimpleRevBlock2(192, 192, 1, act),
            SimpleRevBlock2(192, 192, 1, act),
            SimpleRevBlock2(192, 192, 1, act),
            SimpleRevBlock2(192, 192, 1, act),
            SimpleRevBlock2(192, 192, 1, act),
            SimpleRevBlock2(192, 192, 1, act),
            SimpleRevBlock2(192, 192, 1, act),
            SimpleRevBlock2(192, 192, 1, act),
            SimpleRevBlock2(192, 192, 1, act),
            SimpleRevBlock2(192, 192, 1, act),
            SimpleRevBlock2(192, 192, 1, act),
            SimpleRevBlock2(192, 192, 1, act),
            SimpleRevBlock2(192, 192, 1, act),
            SimpleRevBlock2(192, 192, 1, act),
            ])

        self.d_conv1 = nn.Conv2d(192, 128, 1, 1, 0)

        self.seq2 = RevSequential([
            SimpleRevBlock2(128, 512, 2, act),      # 4
            SimpleRevBlock2(512, 512, 1, act),
            SimpleRevBlock2(512, 512, 1, act),
            SimpleRevBlock2(512, 512, 1, act),
        ])
        self.gavg = nn.AdaptiveAvgPool2d(1)
        self.dense1 = nn.Linear(512, 10)            # 1

    def forward(self, x):
        x1 = x
        x2 = x

        if self.use_rev_bw:
            y1, y2 = rev_sequential_backward_wrapper(self.seq1, x1, x2, preserve_rng_state=False)
        else:
            y1, y2 = self.seq1(x1, x2)

        y = y1 + y2
        y = self.d_conv1(y)

        if self.use_rev_bw:
            y1, y2 = rev_sequential_backward_wrapper(self.seq2, y, y, preserve_rng_state=False)
        else:
            y1, y2 = self.seq2(y, y)

        y = y1 + y2
        y = self.gavg(y).flatten(1)
        y = self.dense1(y)
        return y


if __name__ == '__main__':
    net = Like_IRevNet(True)
    im = torch.rand(5, 3, 32, 32)
    im += torch.zeros(1, device=im.device, dtype=im.dtype, requires_grad=True)
    out = net(im)
    print(out.shape)
    out.sum().backward()
