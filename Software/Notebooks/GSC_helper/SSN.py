import torch
from torch import nn, Tensor

class SubSpectralNorm(nn.Module):
    """
    
    Args:
    num_features: int
        Number of features/ channels.
    affine: str
        Type of SSN, while 'Sub': gammas, betas are specific for each sub-band, 'All': gammas, betas are used for all.
    dim: int
        Provide the index dimension of frequency axis.
    """
    def __init__(self, 
                 num_features: int, 
                 spec_groups: int = 16, 
                 affine: str = 'Sub', 
                 batch: bool = True, 
                 dim: int = 2) -> None:
        super().__init__()
        self.spec_groups = spec_groups
        self.affine_all = False
        affine_norm = False
        if affine == 'Sub':
            # affine transform for each sub group. use affine of torch implementation
            affine_norm = True
        elif affine == 'All':
            self.affine_all = True
            self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1)) # gamma
            self.bias = nn.Parameter(torch.ones(1, num_features, 1, 1)) # beta

        if batch:
            self.ssnorm = nn.BatchNorm2d(num_features*spec_groups, affine = affine_norm)
        else:
            self.ssnorm = nn.InstanceNorm2d(num_features*spec_groups, affine = affine_norm)
        self.sub_dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        x: Tensor
            Input: Should be in the shape (N, C, F, T), or (N, C, T, F)
        """
        if self.sub_dim in (3, -1):
            x = x.transpose(2, 3)
            x = x.contiguous()
        b, c, h, w = x.size()
        assert h%self.spec_groups == 0
        x = x.view(b, c*self.spec_groups, h//self.spec_groups, w)
        x = self.ssnorm(x)
        x = x.view(b, c, h, w)
        if self.affine_all:
            x = x*self.weight + self.bias
        if self.sub_dim in (3, -1):
            x = x.transpose(2, 3)
            x = x.contiguous()
        return x