from __future__ import absolute_import
from __future__ import print_function

import numpy as  np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .endpoint_error import downsample2d_as
from .endpoint_error import elementwise_epe
from .resample2d_package.resample2d import Resample2d


# Charbonnier penalty
def penalty(x):
    return torch.sqrt(x + 1e-5)


class Elbo(nn.Module):
    def __init__(self, args, alpha=1.0, beta=1.0, Nsamples=1):

        super(Elbo, self).__init__()
        self._args = args
        self._Normal = torch.distributions.Normal(0, 1)
        self._Normal.loc = self._Normal.loc.cuda() # hack to get sampling on the GPU
        self._Normal.scale = self._Normal.scale.cuda()
        self._Nsamples = Nsamples
        self._alpha = alpha
        self._beta = beta
        self._resample2d = Resample2d()
        # Convolution kernels for horizontal and vertical derivatives
        self._kernel_dx = torch.tensor([[[[-1, 1]], [[0, 0]]], [[[0, 0]], [[-1, 1]]]], dtype=torch.float32)
        self._kernel_dy = torch.tensor([[[[-1], [1]], [[0], [0]]], [[[0], [0]], [[-1], [1]]]], dtype=torch.float32)
        if args.cuda:
            self._kernel_dx = self._kernel_dx.cuda()
            self._kernel_dy = self._kernel_dy.cuda()

    # Reparametrization trick
    def reparam(self, mean, log_var):
        """
        Generates normal distributed samples with a given mean and variance.
        mean: mean - tensor of size (batch, 2, height, width)
        log_var: log(variance) - tensor of size (batch, 2, height, width)
        returns: samples - tensor of size (Nsamples*batch, 2, height, width)
        """
        mean = mean.repeat(self._Nsamples, 1, 1, 1)
        log_var = log_var.repeat(self._Nsamples, 1, 1, 1)
        z = mean + torch.exp(log_var / 2.0) * self._Normal.sample(mean.size())
        return z

    # Energy function
    def energy(self, flow, img1, img2):
        """
        flow: tensor of size (batch, 2, height, width)
        img1: tensor of size (batch, 3, height, width)
        img2: tensor of size (batch, 3, height, width)
        """

        # Warp img2 according to flow
        img2_warp = self._resample2d(img2, flow.contiguous())
        A = torch.sum((img1.repeat(self._Nsamples, 1, 1, 1) - img2_warp)**2, dim=1)
        data_term = torch.sum(penalty(A), dim=(1, 2))

        B = F.pad(torch.sum(F.conv2d(flow, self._kernel_dx)**2, dim=1), (0,1,0,0)) \
            + F.pad(torch.sum(F.conv2d(flow, self._kernel_dy)**2, dim=1), (0,0,0,1))
        smooth_term = torch.sum(penalty(B), dim=(1, 2))

        return self._alpha * data_term + self._beta * smooth_term

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        target = target_dict["target1"]
        img1 = target_dict["input1"]
        img2 = target_dict["input2"]
        mean, log_var = output_dict["flow1"]

        # Evaluate ELBO
        flow_sample = self.reparam(mean, log_var)
        energy = self.energy(flow_sample, img1, img2)
        entropy = torch.sum(log_var, dim=(1,2,3))/2
        mean_elbo = energy.mean() - entropy.mean()
        loss_dict["elbo"] = mean_elbo

        # Calculate epe for validation
        epe = elementwise_epe(mean, target)
        mean_epe = epe.mean()
        loss_dict["epe"] = mean_epe

        return loss_dict
