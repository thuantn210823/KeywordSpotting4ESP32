import torch
from torch import nn, Tensor
from torch.nn import functional as F

class BaseBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 bias: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        out_ahalf = out_channels//2
        in_ahalf = in_channels//2

        self.pconv1 = nn.Conv2d(in_ahalf,
                                out_ahalf,
                                kernel_size = 1,
                                bias = bias)
        self.bn1 = nn.BatchNorm2d(out_ahalf)
        self.relu1 = nn.ReLU(inplace = True)

        self.dwconv = nn.Conv2d(out_ahalf,
                                out_ahalf,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = padding,
                                dilation = dilation,
                                groups = out_ahalf,
                                bias = bias)
        self.bn2 = nn.BatchNorm2d(out_ahalf)

        self.pconv2 = nn.Conv2d(out_ahalf,
                                out_ahalf,
                                kernel_size = 1,
                                bias = bias)
        self.bn3 = nn.BatchNorm2d(out_ahalf)
        self.relu2 = nn.ReLU(inplace = True)

        self.relu3 = nn.ReLU(inplace = True)
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
        input: Input Tensor: (N, C, H, W)
        """
        output1 = torch.chunk(input, 2, 1)[0]
        output2 = torch.chunk(input, 2, 1)[1]
        output2 = self.relu1(self.bn1(self.pconv1(output2)))
        output2 = self.bn2(self.dwconv(output2))
        output2 = self.relu2(self.bn3(self.pconv2(output2)))
        #if self.in_channels == self.out_channels:
        #    output += input
        output = torch.concat([output1, output2], dim = 1)
        return self.relu3(output)

class EdgeCRNNBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 2,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = True) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size = kernel_size,
                      stride = stride,
                      padding = padding,
                      dilation = dilation,
                      groups = in_channels,
                      bias = bias),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size = 1,
                      bias = bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size = 1,
                      bias = bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size = kernel_size,
                      stride = stride,
                      padding = padding,
                      dilation = dilation,
                      groups = out_channels,
                      bias = bias
                      ),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size = 1,
                      bias = bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )
        self.relu = nn.ReLU(inplace = True)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
        input: Input Tensor: (N, C, H, W)
        """
        output1 = self.branch1(input)
        output2 = self.branch2(input)
        output = torch.concat([output1, output2], dim = 1)
        return self.relu(output)

class StageBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int = 1,
                 dilation: int = 1,
                 bias: bool = True,
                 num_base_blks: int = 1) -> None:
        super().__init__()
        stack = []

        out_ahalf = out_channels//2
        stack.append(EdgeCRNNBlock(in_channels,
                                      out_ahalf,
                                      kernel_size = kernel_size,
                                      stride = 2,
                                      padding = padding,
                                      bias = bias))
        for _ in range(num_base_blks):
            stack.append(BaseBlock(out_channels,
                                      out_channels,
                                      kernel_size = kernel_size,
                                      stride = 1,
                                      padding = padding,
                                      bias = bias))
        self.stack = nn.Sequential(*stack)
    def forward(self, input):
        return self.stack(input)

class EdgeCRNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_size: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 width_multiplier: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(in_channels, 
                               int(24*width_multiplier),
                               kernel_size = 3,
                               stride = 1,
                               padding = 1,
                               bias = False
                               )
        self.bn1 = nn.BatchNorm2d(int(24*width_multiplier))
        self.relu1 = nn.ReLU(inplace = True)

        self.maxpool = nn.MaxPool2d(kernel_size = 3,
                                    stride = 2,
                                    padding = 1
                                    )
        self.stage2 = StageBlock(int(24*width_multiplier),
                                 int(72*width_multiplier),
                                 kernel_size = 3,
                                 padding = 1,
                                 bias = False)
        self.stage3 = StageBlock(int(72*width_multiplier),
                                 int(144*width_multiplier),
                                 kernel_size = 3,
                                 padding = 1,
                                 num_base_blks = 2,
                                 bias = False)
        self.stage4 = StageBlock(int(144*width_multiplier),
                                 int(288*width_multiplier),
                                 kernel_size = 3,
                                 padding = 1,
                                 bias = False)
        
        self.conv5 = nn.Conv2d(int(288*width_multiplier),
                               int(512*width_multiplier),
                               kernel_size = 1,
                               bias = False)
        self.bn5 = nn.BatchNorm2d(int(512*width_multiplier))
        self.relu5 = nn.ReLU(inplace = True)

        self.globalpool = nn.AvgPool2d((3, 1), stride = (1, 1))

        self.lstm = nn.LSTM(input_size = int(512*width_multiplier),
                            hidden_size = hidden_size,
                            batch_first = True)
        
        self.fc = nn.Linear(hidden_size,
                            num_classes)
        
    def forward(self, input: Tensor) -> Tensor:
        output = self.relu1(self.bn1(self.conv1(input)))
        output = self.maxpool(output)
        output = self.stage2(output)
        output = self.stage3(output)
        output = self.stage4(output)
        output = self.relu5(self.bn5(self.conv5(output)))
        output = self.globalpool(output).squeeze(2)
        output, _ = self.lstm(output.transpose(1, 2)) # N, T, H
        output = output.transpose(1, 2).mean(dim = 2)
        return self.fc(output)