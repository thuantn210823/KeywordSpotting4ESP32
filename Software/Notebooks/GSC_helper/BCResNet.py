from typing import Tuple, List

import torch
from torch import nn, Tensor

from SSN import SubSpectralNorm

def get_padding(kernel_size: Tuple[int, int], 
                dilation: int):
    kh, kw = kernel_size
    ph = (kh-1)*dilation//2
    pw = (kw-1)*dilation//2
    return (ph, pw)

class BaseBlock(nn.Module):
    """
    Base block
    
    Args:
    kernel_size: Tuple
        kernel_size for both freq_conv and temporal_conv
    
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int],
                 dilation: Tuple[int, int],
                 bias: bool = True,
                 ssn_kwargs: dict = None,
                 dropout: float = 0.1 
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Set up arguments
        f_kernel = (kernel_size[0], 1)
        t_kernel = (1, kernel_size[1])
        f_stride = (stride[0], 1)
        t_stride = (1, stride[1])
        f_padding = get_padding(f_kernel, dilation[0])
        t_padding = get_padding(t_kernel, dilation[1])

        # Freq_conv
        f2 = []
        if in_channels != out_channels:
            f2.extend([nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size = 1,
                                bias = bias),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace = True)])


        self.f2 = nn.Sequential(*f2,
                                nn.Conv2d(out_channels, 
                                          out_channels,
                                          kernel_size = f_kernel,
                                          stride = f_stride,
                                          padding = f_padding,
                                          dilation = dilation[0],
                                          groups = out_channels,
                                          bias = bias
                                          ),
                                SubSpectralNorm(out_channels, **ssn_kwargs) 
                                if ssn_kwargs else nn.BatchNorm2d(out_channels))
        # Temporal_conv
        self.f1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, None)),
                                nn.Conv2d(out_channels,
                                          out_channels,
                                          kernel_size = t_kernel,
                                          stride = t_stride,
                                          dilation = dilation[1],
                                          padding = t_padding,
                                          groups = out_channels,
                                          bias = bias),
                                nn.BatchNorm2d(out_channels),
                                nn.SiLU(inplace = True),
                                nn.Conv2d(out_channels, 
                                          out_channels,
                                          kernel_size = 1,
                                          bias = bias),
                                nn.Dropout(dropout))
        self.relu = nn.ReLU(inplace = True)

    def forward(self, 
                input: Tensor) -> Tensor:
        auxiliary_x = self.f2(input)
        output = self.f1(auxiliary_x)
        if self.in_channels == self.out_channels:
            output = output + auxiliary_x + input
        else:
            output = output + auxiliary_x
        return self.relu(output)

class BCResBlock(nn.Module):
    """
    
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int],
                 bias: bool = True,
                 ssn_kwargs: dict = None,
                 dropout: float = 0.1,
                 num_blks: int = 1,
                 idx: int = 0
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blks = num_blks

        blks = []
        for i in range(num_blks):
            blks.append(BaseBlock(in_channels if i==0 else out_channels,
                                  out_channels,
                                  kernel_size,
                                  stride if i == 0 else (1, 1),
                                  dilation = (1, 2**idx),
                                  bias = False,
                                  ssn_kwargs = {
                                      'spec_groups': 5
                                  }))
        self.blks = nn.Sequential(*blks)

    def forward(self, 
                input: Tensor) -> Tensor:
        return self.blks(input)

class BCResNet(nn.Module):
    """
    BC-ResNet-tau

    Args:
    in_channels: int
        Number of input channels
    num_classes: int
        Number of classes
    bias: bool
        Whether using bias for Convolutional Layer
    num_factor: int
        Also called tau, determines the type of model [1, 1.5, 2, 3, 6, 8].
    """
    def __init__(self,
                 in_channels: int, 
                 num_classes: int,
                 bias: bool = False, 
                 mul_factor: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes 
        self.mul_factor = mul_factor
        self.net = nn.Sequential(nn.Conv2d(in_channels, 
                                           int(16*mul_factor),
                                           kernel_size = 5,
                                           padding = 2,
                                           stride = (2, 1),
                                           dilation = 1,
                                           bias = bias),
                                 nn.BatchNorm2d(int(16*mul_factor)),
                                 nn.ReLU(inplace = True),
                                 BCResBlock(int(16*mul_factor),
                                            int(8*mul_factor),
                                            kernel_size = (3, 3),
                                            stride = (1, 1),
                                            bias = bias, 
                                            ssn_kwargs = {
                                                'spec_groups': 5
                                            },
                                            num_blks = 2, 
                                            idx = 0),
                                 BCResBlock(int(8*mul_factor),
                                            int(12*mul_factor),
                                            kernel_size = (3, 3),
                                            stride = (2, 1),
                                            bias = bias, 
                                            ssn_kwargs = {
                                                'spec_groups': 5
                                            },
                                            num_blks = 2, 
                                            idx = 1),
                                 BCResBlock(int(12*mul_factor),
                                            int(16*mul_factor),
                                            kernel_size = (3, 3),
                                            stride = (2, 1),
                                            bias = bias, 
                                            ssn_kwargs = {
                                                'spec_groups': 5
                                            },
                                            num_blks = 4, 
                                            idx = 2),
                                 BCResBlock(int(16*mul_factor),
                                            int(20*mul_factor),
                                            kernel_size = (3, 3),
                                            stride = (1, 1),
                                            bias = False, 
                                            ssn_kwargs = {
                                                'spec_groups': 5
                                            },
                                            num_blks = 4, 
                                            idx = 3),
                                 nn.Conv2d(int(20*mul_factor),
                                           int(20*mul_factor),
                                           kernel_size = 5,
                                           padding = (0, 2),
                                           bias  = bias,
                                           groups = int(20*mul_factor)),
                                 nn.Conv2d(int(20*mul_factor),
                                           int(32*mul_factor),
                                           kernel_size = 1,
                                           bias = bias),
                                 nn.BatchNorm2d(int(32*mul_factor)),
                                 nn.ReLU(inplace = True),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Conv2d(int(32*mul_factor),
                                           num_classes,
                                           kernel_size = 1)                                 
                                 )
    def forward(self, input:Tensor) -> Tensor:
        """
        Args:
        input: Tensor
            Input tensor, should be in shape of (N, C, F, T)
        
        Return
        Tensor in shape (N, num_classes)
        """
        x = self.net(input)
        return x.reshape(-1, x.shape[1])

if __name__ == '__main__':
    # Assume that input has the shape (N, 1, 40, W)
    bcres1 = BCResNet(1, 40, False, 1)
    bcres1p5 = BCResNet(1, 40, False, 1.5)
    bcres3 = BCResNet(1, 40, False, 3)
    bcres6 = BCResNet(1, 40, False, 6)
    bcres8 = BCResNet(1, 40, False, 8)
