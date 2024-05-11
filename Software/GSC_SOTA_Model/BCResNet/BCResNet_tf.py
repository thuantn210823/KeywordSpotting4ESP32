import tensorflow as tf
from tensorflow import keras
import numpy as np

class SubSpectralNorm(keras.Model):
    def __init__(self, normalized_shape, spec_groups = 16, affine = 'Sub', batch = True):
        super().__init__()
        self.spec_groups = spec_groups
        self.affine_all = False
        if affine == 'Sub':
            affine_norm = True
        else:
            assert affine == 'Sub', "Haven't supported yet"

        if batch:
            self.ssnorm = keras.layers.BatchNormalization(axis = -1)
        else:
            self.ssnorm = keras.layers.InstanceNormalization(axis = [0, 1])
        h, w, c = normalized_shape
        self.permute1 = keras.layers.Permute((3, 1, 2))
        self.reshape1 = keras.layers.Reshape((c*spec_groups, h//spec_groups, w))
        self.reshape2 = keras.layers.Reshape((c, h, w))    
        self.permute2 = keras.layers.Permute((2, 3, 1))    

    def call(self, x):
        """
        x: (N, F, T, C)
        """
        x = self.reshape1(self.permute1(x))
        x = self.ssnorm(x)
        x = self.permute2(self.reshape2(x))
        return x

class AdaptiveAvgPool2d(keras.Model):
    def __init__(self,
                 axis: list,
                 keepdims:bool = True) -> None:
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def call(self, input):
        return tf.reduce_mean(input, axis = self.axis, keepdims = self.keepdims)
    
from typing import Tuple, List

class BaseBlock(keras.Model):
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
        f_kernel = (kernel_size[0], 1)
        t_kernel = (1, kernel_size[1])
        f_stride = (stride[0], 1)
        t_stride = (1, stride[1])
        f_dilation = (dilation[0], 1)
        t_dilation = (1, dilation[1])

        # Freq_conv
        f2 = []
        if in_channels != out_channels:
            f2.extend([keras.layers.Conv2D(filters = out_channels,
                                           kernel_size = 1,
                                           use_bias = bias),
                        keras.layers.BatchNormalization(axis = -1),
                        keras.layers.ReLU()])


        self.f2 = keras.Sequential([*f2,
                                keras.layers.Conv2D(filters = out_channels,
                                                    kernel_size = f_kernel,
                                                    strides = f_stride,
                                                    padding = 'same',
                                                    dilation_rate = f_dilation,
                                                    groups = out_channels,
                                                    use_bias = bias
                                          ),
                                SubSpectralNorm(**ssn_kwargs)
                                if ssn_kwargs else keras.layers.BatchNormalization(axis = -1)])

        # Temporal_conv
        self.f1 = keras.Sequential([AdaptiveAvgPool2d(axis = 1, keepdims = True),
                                keras.layers.Conv2D(filters = out_channels,
                                                    kernel_size = t_kernel,
                                                    strides = t_stride,
                                                    dilation_rate = t_dilation,
                                                    padding = 'same',
                                                    groups = out_channels,
                                                    use_bias = bias,
                                                    activation = 'swish'),
                                keras.layers.BatchNormalization(axis = -1),
                                keras.layers.Activation("swish"),
                                keras.layers.Conv2D(filters = out_channels,
                                                    kernel_size = 1,
                                                    use_bias = bias),
                                keras.layers.Dropout(dropout)])
        self.relu = keras.layers.ReLU()

    def call(self,
             input: tf.Tensor) -> tf.Tensor:
        auxiliary_x = self.f2(input)
        output = self.f1(auxiliary_x)
        if self.in_channels == self.out_channels:
            output = output + auxiliary_x + input
        else:
            output = output + auxiliary_x
        return self.relu(output)

class BCResBlock(keras.Model):
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
                                  # dilation = (1, 1),
                                  bias = False,
                                  ssn_kwargs = ssn_kwargs))
        self.blks = keras.Sequential(blks)

    def call(self,
                input: tf.Tensor) -> tf.Tensor:
        return self.blks(input)

class BCResNet(keras.Model):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 bias: bool = False,
                 mul_factor: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mul_factor = mul_factor
        self.net = keras.Sequential([keras.layers.Conv2D(filters = int(16*mul_factor),
                                                     kernel_size = 5,
                                                     padding = 'same',
                                                     strides = (2, 1),
                                                     use_bias = bias),
                                 keras.layers.BatchNormalization(axis = -1),
                                 keras.layers.ReLU(),
                                 BCResBlock(int(16*mul_factor),
                                            int(8*mul_factor),
                                            kernel_size = (3, 3),
                                            stride = (1, 1),
                                            bias = bias,
                                            ssn_kwargs = {
                                                'normalized_shape': (20, 100, int(8*mul_factor)),
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
                                                'normalized_shape': (10, 100, int(12*mul_factor)),
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
                                                'normalized_shape': (5, 100, int(16*mul_factor)),
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
                                                'normalized_shape': (5, 100, int(20*mul_factor)),
                                                'spec_groups': 5
                                            },
                                            num_blks = 4,
                                          idx = 3),
                                 keras.layers.Conv2D(filters = int(20*mul_factor),
                                                     kernel_size = 5,
                                                     padding = 'same',
                                                     use_bias  = bias,
                                                     groups = int(20*mul_factor)),
                                 keras.layers.Conv2D(filters = int(32*mul_factor),
                                                     kernel_size = 1,
                                                     use_bias = bias),
                                 keras.layers.BatchNormalization(axis = -1),
                                 keras.layers.ReLU(),
                                 AdaptiveAvgPool2d([1, 2]),
                                 keras.layers.Conv2D(filters = num_classes,
                                           kernel_size = 1),
                                 keras.layers.Flatten()]
                                 )
    def call(self, input: tf.Tensor) -> tf.Tensor:
        x = self.net(input)
        return x

if __name__ == '__main__':
    bcres1 = BCResNet(1, 12, False, 1)
    bcres1p5 = BCResNet(1, 12, False, 1.5)
    bcres3 = BCResNet(1, 12, False, 3)
    bcres6 = BCResNet(1, 12, False, 6)
    bcres8 = BCResNet(1, 12, False, 8)