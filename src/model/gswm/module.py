import torch
from torch.nn import functional as F
import math
from torch import nn
import numpy as np
from utils import transform_tensors

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class MultiLayerConv(nn.Module):

    def __init__(self, channels, kernel_sizes, strides, act):
        """

        Args:
            channels:
            kernel_sizes:
            strides:
            act:
            norm:
            nchannels:
        """
        nn.Module.__init__(self)
        assert len(channels) == len(strides) + 1
        assert isinstance(kernel_sizes, int) or len(kernel_sizes) == len(channels) - 1
        assert isinstance(act, nn.Module)
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(strides)
    
        self.net = []
    
        for i in range(len(channels) - 1):
            padding = kernel_sizes[i] // 2
            self.net.append(nn.Conv2d(channels[i], channels[i + 1], kernel_sizes[i], strides[i], padding))
            if i != len(channels) - 2:
                self.net.append(act)
                self.net.append(nn.GroupNorm(channels[i + 1] // 16, channels[i + 1]))
    
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
    
        return self.net(x)

class MultiLayerSubpixel(nn.Module):
    def __init__(self, channels, kernel_sizes, upscale_factors, act):
        """

        Args:
            channels:
            kernel_sizes:
            strides:
            act:
            norm:
            nchannels:
        """
        nn.Module.__init__(self)
        assert len(channels) == len(upscale_factors) + 1
        assert isinstance(kernel_sizes, int) or len(kernel_sizes) == len(channels) - 1
        assert isinstance(act, nn.Module)
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(upscale_factors)
    
        self.net = []
        for i in range(len(channels) - 1):
            padding = kernel_sizes[i] // 2
            self.net.append(
                nn.Conv2d(channels[i], channels[i + 1] * upscale_factors[i] ** 2, kernel_sizes[i], 1, padding))
            if i != len(channels) - 2:
                self.net.append(act)
            self.net.append(nn.PixelShuffle(upscale_factors[i]))
            if i != len(channels) - 2:
                self.net.append(nn.GroupNorm(channels[i + 1] // 16, channels[i + 1]))
    
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
    
        return self.net(x)
class MLP(nn.Module):
    
    def __init__(self, sizes, act, output_act=None):
        """
        
        Args:
            sizes: a list of ints
            act: activation function. nn.Module
            output_act: activation function. nn.Module
        """
        nn.Module.__init__(self)
        assert len(sizes) >= 2
        assert isinstance(act, nn.Module)
        assert isinstance(output_act, (nn.Module, type(None)))
        
        self.layers = nn.ModuleList()
        self.act = act
        self.output_act = output_act
        self.sizes = sizes
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
    
    def forward(self, x):
        """
        
        Args:
            x: (*, D)

        Returns:
            x: (*, D)
        """
        *ORI, D = x.size()
        x = x.view(np.prod(ORI), D)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.act(x)
                # x = F.group_norm(x, self.sizes[i+1] // 16)
            elif self.output_act:
                # output layer
                x = self.output_act(x)
        
        x = x.view(*ORI, self.sizes[-1])
        
        return x


def gaussian_kernel_2d(xy, sigma, dim):
    """
    Compute weights using a gaussian kernel
    Args:
        xy: (*A, 2, *B)
        sigma: float
        dim: the dimension for which to compute the weight
    Returns:
        (*A, *B)

    """
    # (*A, *B)
    square_sum = (xy ** 2).sum(dim=dim)
    # Kernel, (*A, *B)
    weights = 1 / (2. * math.pi * sigma ** 2) * torch.exp(
        -square_sum / (2 * sigma ** 2)
    )
    return weights

def kl_divergence_bern_bern(z_pres_probs, prior_pres_prob, eps=1e-15):
    """
    Compute kl divergence
    :param z_pres_logits: (B, ...)
    :param prior_pres_prob: float
    :return: kl divergence, (B, ...)
    """
    # z_pres_probs = torch.sigmoid(z_pres_logits)
    kl = z_pres_probs * (torch.log(z_pres_probs + eps) - torch.log(prior_pres_prob + eps)) + \
         (1 - z_pres_probs) * (torch.log(1 - z_pres_probs + eps) - torch.log(1 - prior_pres_prob + eps))
    
    return kl

class BatchApply(nn.Module):
    def __init__(self, module):
        nn.Module.__init__(self)
        self.mod = module
    
    def forward(self, *args):
        return batch_apply(self.mod)(*args)

def batch_apply(func):
    def factory(*args):
        
        assert isinstance(args[0], torch.Tensor), 'For BatchApply, first input must be Tensor'
        *OTHER, D = args[0].size()
        args = transform_tensors(args, func=lambda x: x.view(int(np.prod(OTHER)), -1))
    
        out = func(*args)
        out = transform_tensors(out, func=lambda x: x.view(*OTHER, -1))
        return out
    return factory


def anneal(step, start_step, end_step, start_value, end_value, type):
    assert type in ['linear', 'exp']
    if type == 'exp':
        start_value = math.log(start_value)
        end_value = math.log(end_value)
    
    if step <= start_step:
        x = start_value
    elif start_step < step < end_step:
        slope = (end_value - start_value) / (end_step - start_step)
        x = start_value + slope * (step - start_step)
    else:
        x = end_value
    
    if type == 'exp':
        x = math.exp(x)
    return x



