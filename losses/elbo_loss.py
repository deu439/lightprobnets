from __future__ import absolute_import
from __future__ import print_function

import numpy as np
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
        kernel_dx = torch.tensor([[[[-1, 1]], [[0, 0]]], [[[0, 0]], [[-1, 1]]]], dtype=torch.float32)
        kernel_dy = torch.tensor([[[[-1], [1]], [[0], [0]]], [[[0], [0]], [[-1], [1]]]], dtype=torch.float32)
        # Buffers are moved to GPU together with the parent module
        self.register_buffer('_kernel_dx', kernel_dx)
        self.register_buffer('_kernel_dy', kernel_dy)

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

        # Warp img2 according to flow - on cpu-located tensors warping module fails without warning!
        assert(img2.is_cuda and flow.is_cuda)
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


class MultiScaleElbo(Elbo):
    def __init__(self,
                 args,
                 num_scales=5,
                 num_highres_scales=2,
                 coarsest_resolution_loss_weight=0.32,
                 alpha=1.0,
                 beta=1.0,
                 Nsamples=1):

        super(MultiScaleElbo, self).__init__(args=args, alpha=alpha, beta=beta, Nsamples=Nsamples)
        self._num_scales = num_scales

        # ---------------------------------------------------------------------
        # start with initial scale
        # for "low-resolution" scales we apply a scale factor of 4
        # for "high-resolution" scales we apply a scale factor of 2
        #
        # e.g. for FlyingChairs  weights=[0.005, 0.01, 0.02, 0.08, 0.32]
        # ---------------------------------------------------------------------
        self._weights = [coarsest_resolution_loss_weight]
        num_lowres_scales = num_scales - num_highres_scales
        for k in range(num_lowres_scales - 1):
            self._weights += [self._weights[-1] / 4]
        for k in range(num_highres_scales):
            self._weights += [self._weights[-1] / 2]
        self._weights.reverse()
        assert (len(self._weights) == num_scales)  # sanity check

    # Override base class forward() method !
    def forward(self, output_dict, target_dict):
        loss_dict = {}
        target = target_dict["target1"]
        img1 = target_dict["input1"]
        img2 = target_dict["input2"]

        if self.training:
            outputs = [output_dict[key] for key in ["flow2", "flow3", "flow4", "flow5", "flow6"]]

            total_loss = 0
            for i, output_i in enumerate(outputs):
                yscale = output_i[0].size(2) / img1.size(2)
                xscale = output_i[0].size(3) / img1.size(3)
                img1_i = downsample2d_as(img1, output_i[0])
                img2_i = downsample2d_as(img2, output_i[0])
                mean_i, log_var_i = output_i

                # Scale flow and variance
                scale = torch.tensor([xscale, yscale]).cuda()
                mean_i = mean_i * scale[None, :, None, None]
                log_var_i = log_var_i + 2*torch.log(scale)[None, :, None, None]

                # Evaluate ELBO for a given scale
                flow_sample = self.reparam(mean_i, log_var_i)
                energy_i = self.energy(flow_sample, img1_i, img2_i)
                entropy_i = torch.sum(log_var_i, dim=(1, 2, 3)) / 2
                mean_elbo_i = energy_i.mean() - entropy_i.mean()

                # Cummulate
                total_loss += self._weights[i] * mean_elbo_i
                loss_dict["elbo%i" % (i + 2)] = mean_elbo_i

            loss_dict["total_loss"] = total_loss
        else:
            mean, log_var = output_dict["flow1"]

            # Evaluate -ELBO
            flow_sample = self.reparam(mean, log_var)
            energy = self.energy(flow_sample, img1, img2)
            entropy = torch.sum(log_var, dim=(1, 2, 3)) / 2
            mean_elbo = energy.mean() - entropy.mean()
            loss_dict["elbo"] = mean_elbo

            # Evaluate EPE
            epe = elementwise_epe(mean, target)
            mean_epe = epe.mean()
            loss_dict["epe"] = mean_epe

            # Evaluate log-likelihood -2 log q(w=w_true|F, G, theta)
            err = target - mean
            llh = np.log(2*np.pi) + log_var + (err**2) / torch.exp(log_var)
            loss_dict["llh"] = llh.mean()

        return loss_dict

