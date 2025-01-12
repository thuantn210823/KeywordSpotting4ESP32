from typing import Optional, Tuple

import torch
from torch import nn

class ConvT(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_groups: int,
                 kernel_size: int,
                 dilation: int = 1,
                 bias: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = num_groups
        padding = (kernel_size - 1)*dilation//2
        out_channels = in_channels*num_groups
        self.convt = nn.Conv2d(out_channels,
                              out_channels,
                              kernel_size = (1, kernel_size),
                              dilation = (1, dilation),
                              padding = (0, padding),
                              groups = out_channels,
                              bias = bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, F, T = x.shape
        x = x.reshape(N, C*self.num_groups, F//self.num_groups, T)
        x = self.convt(x)
        x = x.reshape(N, C, F, T)
        return x

class ConvF(nn.Module):
    def __init__(self,
                 in_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 #dilation: int = 1,
                 bias = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.convf = nn.Conv2d(in_channels,
                               in_channels,
                               kernel_size = (kernel_size, 1),
                               stride = (stride, 1),
                               padding = (padding, 0),
                               groups = in_channels,
                               bias = bias
                               )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convf(x)

class ConvTF(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 num_groups: Optional[int] = None,
                 bias: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride == 1:
            self.convt = ConvT(in_channels,
                               kernel_size = kernel_size[1],
                               num_groups = num_groups if num_groups else 1,
                               dilation = dilation,
                               bias = bias
                               )
        else:
            self.convt = nn.Sequential(nn.AdaptiveAvgPool2d((1, None)),
                                       nn.Conv2d(in_channels,
                                                 in_channels,
                                                 kernel_size = (1, kernel_size[1]),
                                                 padding = (0, (kernel_size[1]-1)*dilation//2),
                                                 dilation = dilation,
                                                 groups = in_channels,
                                                 bias = bias))
        self.convf = ConvF(in_channels,
                           kernel_size = kernel_size[0],
                           stride = stride,
                           padding = padding,
                           bias = bias
                           )
        self.pconv = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size = 1,
                               bias = bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = self.convt(x)
        xf = self.convf(x)
        return self.pconv(xt+xf)

class BaseBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 num_groups: Optional[int] = None,
                 bias: bool = False,
                 dropout:float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv = ConvTF(in_channels,
                           out_channels,
                           kernel_size = kernel_size,
                           stride = stride,
                           padding = padding,
                           dilation = dilation,
                           num_groups = num_groups,
                           bias = bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace = True)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.dropout(self.silu(self.bn(self.conv(x))))
        if (self.in_channels == self.out_channels):
            output += x
        return self.relu(output)

class StackBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: int = 1,
                 padding: int = 0,
                 num_groups: Optional[int] = None,
                 bias: bool = False,
                 dropout:float = 0.1,
                 num_blks:int = 1,
                 idx: int = 0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blks = num_blks

        blks = []
        for i in range(num_blks):
            blks.append(BaseBlock(in_channels if i==0 else out_channels,
                                  out_channels,
                                  kernel_size = kernel_size,
                                  stride = stride if i == 0 else 1,
                                  padding = padding,
                                  dilation = 2**idx,
                                  num_groups = num_groups,
                                  bias = bias,
                                  dropout = dropout
                                  ))
        self.blks = nn.Sequential(*blks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blks(x)

class TFResNet(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 bias: bool = False,
                 mul_factor: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.rest = nn.Sequential(nn.Conv2d(in_channels,
                                            int(mul_factor*16),
                                            kernel_size = 5,
                                            stride = (2, 1),
                                            padding = 2,
                                            dilation = 1,
                                            bias = bias),
                                  nn.BatchNorm2d(int(mul_factor*16)),
                                  nn.ReLU(),
                                  StackBlock(int(mul_factor*16),
                                             int(8*mul_factor),
                                             kernel_size = (3, 3),
                                             stride = 1,
                                             padding = 1,
                                             num_groups = 5,
                                             num_blks = 2,
                                             bias = bias), # 40
                                  StackBlock(int(8*mul_factor),
                                             int(12*mul_factor),
                                             kernel_size = (3, 3),
                                             stride = 2,
                                             padding = 1,
                                             num_groups = 5,
                                             num_blks = 2,
                                             idx = 1,
                                             bias = bias), # 20
                                  StackBlock(int(12*mul_factor),
                                             int(16*mul_factor),
                                             kernel_size = (3, 3),
                                             stride = 2,
                                             padding = 1,
                                             num_groups = 5,
                                             num_blks = 4,
                                             idx = 2,
                                             bias = bias), # 20
                                  StackBlock(int(16*mul_factor),
                                             int(20*mul_factor),
                                             kernel_size = (3, 3),
                                             stride = 1,
                                             padding = 1,
                                             num_groups = 5,
                                             num_blks = 4,
                                             idx = 3,
                                             bias = bias), # 10
                                  BaseBlock(int(20*mul_factor),
                                            int(32*mul_factor),
                                            kernel_size = (3, 3),
                                            stride = 1,
                                            padding = 1),
                                  nn.AdaptiveAvgPool2d((1, 1)),
                                  nn.Flatten(),
                                  nn.Dropout(0.1),
                                  nn.Linear(int(32*mul_factor), num_classes)
                                  )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = self.conv1d(x)
        #x = self.bn1(x)
        #x = self.relu1(x)
        #x = self.rest(x.unsqueeze(1))
        x = self.rest(x)
        return x

if __name__ == '__main__':
    tfres1 = TFResNet(1, 12, False, 1)
    tfres1p5 = TFResNet(1, 12, False, 1.5)
    tfres3 = TFResNet(1, 12, False, 3)
    tfres6 = TFResNet(1, 12, False, 6)
    tfres8 = TFResNet(1, 12, False, 8)