#######################################
#
#
# Date: 18th, March 2024
#######################################
import torch
from torch import Tensor, nn
from torch.nn import functional as F

class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride = 1,
                 dilation = 1,
                 groups = 1,
                 bias = True) -> None:
        self.__padding = (kernel_size - 1)*dilation

        super(CausalConv1d, self).__init__(in_channels,
                                  out_channels,
                                  kernel_size = kernel_size,
                                  stride = stride,
                                  padding = self.__padding,
                                  dilation = dilation,
                                  groups = groups,
                                  bias = bias
                                  )

    def forward(self, input: Tensor) -> Tensor:
        input = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return input[:, :, :-self.__padding]
        return input
    
class DTCBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.dilated_depth_tcn = CausalConv1d(in_channels,
                                              in_channels,
                                              kernel_size = kernel_size,
                                              dilation = dilation,
                                              groups = in_channels)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.point_conv1 = nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size = 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.point_conv2 = nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size = 1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: torch.Tensor: Input tensor (N, C, T)
        Returns
            torch.Tensor: Output tensor (N, C, T)
        """
        output = self.dilated_depth_tcn(input)
        output = self.bn1(output)
        output = self.point_conv1(output)
        output = self.relu1(self.bn2(output))
        output = self.point_conv2(output)
        output = self.bn3(output)
        if self.in_channels == self.out_channels:
            output = input + output
        return self.relu2(output)

class DTCStack(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stack_size: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stack_size = stack_size

        dilations = [2**i for i in range(stack_size)]
        stack = []
        stack.append(DTCBlock(in_channels,
                              out_channels,
                              kernel_size = kernel_size,
                              dilation = dilations[0]))
        for i in range(1, stack_size):
            stack.append(DTCBlock(out_channels,
                                  out_channels,
                                  kernel_size = kernel_size,
                                  dilation = dilations[i]))
        self.stack = nn.Sequential(*stack)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input: torch.Tensor: Input tensor (N, C, T)
        Returns
            torch.Tensor: Output tensor (N, C, T)
        """
        return self.stack(input)

class MDTC(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stack_num: int,
                 stack_size: int,
                 classification: bool = None,
                 hidden_size: int = None,
                 num_classes: int = None,
                 dropout: float = 0.1) -> None:
        """
        This model is built base on the original paper .
        'Two-stage streaming keyword detection and localization with
        multi-scale depthwise temporal convolution'- 2021.

        Multi-scale depthwise temporal convolution model (MDTC).
        MDTC here provide 2 mode of using which is defined through classification argument.

        Args:
            in_channels: int
                Number of input channels.
            out_channels: int
                Number of output channels.
            kernel_size: int
                Size of filters used for convolutions.
            stack_num: int
                (M argument in the paper). The number of DTCStack block.
            stack_size: int
                The number of DTCBlock in each DTCStack.
            classification: bool
                If True, allow the model can directly be used for classification problems. 
            hidden_size: int
                Number of nodes in hidden layers, needed when classification is True.
            num_classes: int
                Number of classes for classification problems, needed when classification is True
        
        """
        super().__init__()
        self.stack_num = stack_num
        self.classification = classification
        self.preprocessing_tdc = DTCBlock(in_channels,
                                          out_channels,
                                          kernel_size = kernel_size,
                                          dilation = 1)
        self.stack = nn.ModuleList()
        for i in range(stack_num):
            self.stack.append(DTCStack(out_channels,
                                  out_channels,
                                  kernel_size = kernel_size,
                                  stack_size = stack_size))

        if classification:
            assert hidden_size and num_classes, \
            "In classification mode you should give the model hidden_size and num_classes"
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(out_channels, hidden_size)
            self.relu1 = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        input = self.preprocessing_tdc(input)
        output = 0

        for i in range(self.stack_num):
            input = self.stack[i](input)
            output += input

        if self.classification:
            output = self.avgpool(output).squeeze(1)
            output = self.dropout(self.relu1(self.fc1(output)))
            output = self.fc2(output)

        return output

if __name__ == '__main__':
    x = torch.Tensor(128, 40, 81)
    mdtc = MDTC(40, 64, 5, 4, 4, True, 64, 12)
    print(mdtc(x).shape)