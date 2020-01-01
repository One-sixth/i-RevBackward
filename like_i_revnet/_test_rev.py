'''
检查各个可逆模块是否正常
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.checkpoint import checkpoint
from .rev_blocks import RevSequential, SimpleRevBlock, SimpleRevBlock2
from .rev_utils import rev_sequential_backward_wrapper


if __name__ == '__main__':

    # check SimpleRevBlock
    rb1 = SimpleRevBlock()
    rb2 = SimpleRevBlock()
    rb3 = SimpleRevBlock()
    rb4 = SimpleRevBlock()

    x1 = torch.rand(3, 20, 32, 32)
    x2 = torch.rand(3, 20, 32, 32)
    xs = (x1, x2)

    y = rb1(*xs)
    y = rb2(*y)
    y = rb3(*y)
    y = rb4(*y)
    ry = rb4.invert(*y)
    ry = rb3.invert(*ry)
    ry = rb2.invert(*ry)
    rx1, rx2 = rb1.invert(*ry)
    assert torch.allclose(rx1, x1, rtol=1e-5, atol=1e-5) and torch.allclose(rx2, x2, rtol=1e-5, atol=1e-5)


    # check SimpleRevBlock2
    rb1 = SimpleRevBlock2(6, 32, 2, nn.LeakyReLU())
    rb2 = SimpleRevBlock2(32, 48, 1, nn.LeakyReLU())
    rb3 = SimpleRevBlock2(48, 48, 1, nn.LeakyReLU())
    rb4 = SimpleRevBlock2(48, 48, 1, nn.LeakyReLU())

    x1 = torch.rand(3, 6, 32, 32)
    x2 = torch.rand(3, 6, 32, 32)
    xs = (x1, x2)

    y = rb1(*xs)
    y = rb2(*y)
    y = rb3(*y)
    y = rb4(*y)
    ry = rb4.invert(*y)
    ry = rb3.invert(*ry)
    ry = rb2.invert(*ry)
    rx1, rx2 = rb1.invert(*ry)
    assert torch.allclose(rx1, x1, rtol=1e-5, atol=1e-5) and torch.allclose(rx2, x2, rtol=1e-5, atol=1e-5)


    # check RevSequential and RevSequentialBackwardFunction
    rb1 = SimpleRevBlock()
    rb2 = SimpleRevBlock()
    rb3 = SimpleRevBlock()
    rb4 = SimpleRevBlock()
    rb5 = SimpleRevBlock()
    rb6 = SimpleRevBlock()
    rb7 = SimpleRevBlock()
    rb8 = SimpleRevBlock()
    rb9 = SimpleRevBlock()
    rb10 = SimpleRevBlock()
    rb11 = SimpleRevBlock()
    rb12 = SimpleRevBlock()
    rb13 = SimpleRevBlock()

    rs = RevSequential()
    rs.append(rb1)
    rs.append(rb2)
    rs.append(rb3)
    rs.append(rb4)
    rs.append(rb5)
    rs.append(rb6)
    rs.append(rb7)
    rs.append(rb8)
    rs.append(rb9)
    rs.append(rb10)
    rs.append(rb11)
    rs.append(rb12)
    rs.append(rb13)

    x1 = torch.rand(3, 20, 32, 32)
    x2 = torch.rand(3, 20, 32, 32)
    x1.requires_grad_(True)
    x2.requires_grad_(True)

    xs = (x1, x2)
    ys = rs(*xs)
    # ys2 = checkpoint(rs, *xs)
    ys3 = rev_sequential_backward_wrapper(rs, *xs)

    assert torch.allclose(ys[0], ys3[0]) and torch.allclose(ys[1], ys3[1])

    y = (ys3[0] + ys3[1]).sum()
    y.backward()
    print(y)


    # check train
    optim = torch.optim.Adam(rs.parameters(), lr=1e-2)

    x1 = torch.rand(3, 20, 128, 128).cuda()
    x2 = torch.rand(3, 20, 128, 128).cuda()

    rs.cuda()

    for _ in range(20):
        in1 = x1 + torch.zeros(1, requires_grad=True).cuda()
        in2 = x2 + torch.zeros(1, requires_grad=True).cuda()

        # ou1, ou2 = rs(in1, in2)
        # ou1, ou2 = checkpoint(rs, in1, in2)
        ou1, ou2 = rev_sequential_backward_wrapper(rs, in1, in2, preserve_rng_state=False)

        loss = (ou1 + ou2).abs().sum()
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(loss.item())

        for p in rs.parameters():
            assert (not p.requires_grad) or p.grad is not None
