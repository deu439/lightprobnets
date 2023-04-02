from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .endpoint_error import downsample2d_as
from .endpoint_error import elementwise_epe
from .resample2d_package.resample2d import Resample2d


def upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return F.interpolate(inputs, [h, w], mode=mode, align_corners=True)


# Charbonnier penalty
def penalty(x):
    return torch.sqrt(x + 1e-5)


def border_mask(flow):
    """
    Generates a mask that is True for pixels whose correspondence is inside the image borders.
    flow: optical flow tensor (batch, 2, height, width)
    returns: mask (batch, height, width)
    """
    b, _, h, w = flow.size()
    x = torch.arange(w).cuda()
    y = torch.arange(h).cuda()
    X, Y = torch.meshgrid(x, y, indexing='xy')
    Xp = X.view(1, h, w).repeat(b, 1, 1) + flow[:, 0, :, :]
    Yp = Y.view(1, h, w).repeat(b, 1, 1) + flow[:, 1, :, :]
    mask_x = (Xp > -0.5) & (Xp < w-0.5)
    mask_y = (Yp > -0.5) & (Yp < h-0.5)
    return mask_x & mask_y


class Elbo(nn.Module):
    def __init__(self, args, alpha=1.0, beta=1.0, gamma=1.0, Nsamples=1, entropy_weight=1.0, mask_cost=0.0):

        super(Elbo, self).__init__()
        self._args = args
        self._Normal = torch.distributions.Normal(0, 1)
        self._Normal.loc = self._Normal.loc.cuda() # hack to get sampling on the GPU
        self._Normal.scale = self._Normal.scale.cuda()
        self._Nsamples = Nsamples
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._entropy_weight = entropy_weight
        self._mask_cost = mask_cost
        self._resample2d = Resample2d()
        # Convolution kernels for horizontal and vertical derivatives
        kernel_dx = torch.tensor([[[[-1, 1]], [[0, 0]]], [[[0, 0]], [[-1, 1]]]], dtype=torch.float32)
        kernel_dy = torch.tensor([[[[-1], [1]], [[0], [0]]], [[[0], [0]], [[-1], [1]]]], dtype=torch.float32)
        # Buffers are moved to GPU together with the parent module
        self.register_buffer('_kernel_dx', kernel_dx)
        self.register_buffer('_kernel_dy', kernel_dy)
        # Convolution kernels for derivatives of the images
        kernel_gx = torch.tensor([
            [[[-1 / 12, 2 / 3, 0, -2 / 3, 1 / 12]], [[0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0]]],
            [[[0, 0, 0, 0, 0]], [[-1 / 12, 2 / 3, 0, -2 / 3, 1 / 12]], [[0, 0, 0, 0, 0]]],
            [[[0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0]], [[-1 / 12, 2 / 3, 0, -2 / 3, 1 / 12]]]
        ], dtype=torch.float32)
        kernel_gy = torch.tensor([
            [[[-1 / 12], [2 / 3], [0], [-2 / 3], [1 / 12]], [[0], [0], [0], [0], [0]], [[0], [0], [0], [0], [0]]],
            [[[0], [0], [0], [0], [0]], [[-1 / 12], [2 / 3], [0], [-2 / 3], [1 / 12]], [[0], [0], [0], [0], [0]]],
            [[[0], [0], [0], [0], [0]], [[0], [0], [0], [0], [0]], [[-1 / 12], [2 / 3], [0], [-2 / 3], [1 / 12]]]
        ], dtype=torch.float32)
        self.register_buffer('_kernel_gx', kernel_gx)
        self.register_buffer('_kernel_gy', kernel_gy)

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

        # Calculate border mask
        mask = border_mask(flow)
        mask_term = torch.sum(mask, dim=(1, 2))

        # Warp img2 according to flow - on cpu-located tensors warping module fails without warning!
        assert(img2.is_cuda and flow.is_cuda)
        img2_warp = self._resample2d(img2, flow.contiguous())
        A = torch.sum((img1.repeat(self._Nsamples, 1, 1, 1) - img2_warp)**2, dim=1)
        data_term = torch.sum(penalty(A) * mask, dim=(1, 2))    # Sum masked penalty over x, y

        B = F.pad(torch.sum(F.conv2d(flow, self._kernel_dx)**2, dim=1), (0,1,0,0)) \
            + F.pad(torch.sum(F.conv2d(flow, self._kernel_dy)**2, dim=1), (0,0,0,1))
        smooth_term = torch.sum(penalty(B), dim=(1, 2))

        img1_gx = F.conv2d(img1, self._kernel_gx, padding=(0, 2))
        img1_gy = F.conv2d(img1, self._kernel_gy, padding=(2, 0))
        img2_warp_gx = F.conv2d(img2_warp, self._kernel_gx, padding=(0, 2))
        img2_warp_gy = F.conv2d(img2_warp, self._kernel_gy, padding=(2, 0))
        C = torch.sum((img1_gx.repeat(self._Nsamples, 1, 1, 1) - img2_warp_gx)**2, dim=1) \
            + torch.sum((img1_gy.repeat(self._Nsamples, 1, 1, 1) - img2_warp_gy)**2, dim=1)
        gradient_term = torch.sum(penalty(C) * mask, dim=(1, 2))    # Sum masked penalty over x, y

        return self._alpha * data_term + self._beta * smooth_term + self._gamma * gradient_term + self._mask_cost * mask_term

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
        mean_elbo = energy.mean() - self._entropy_weight * entropy.mean()
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
                 gamma=1.0,
                 Nsamples=1,
                 scale_var=True,
                 entropy_weight=1.0,
                 mask_cost=0.0):

        super(MultiScaleElbo, self).__init__(args=args, alpha=alpha, beta=beta, gamma=gamma, Nsamples=Nsamples, entropy_weight=entropy_weight, mask_cost=mask_cost)
        self._num_scales = num_scales
        self._scale_var = scale_var

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
                if self._scale_var:
                    log_var_i = log_var_i + 2*torch.log(scale)[None, :, None, None]

                # Evaluate ELBO for a given scale
                flow_sample = self.reparam(mean_i, log_var_i)
                energy_i = self.energy(flow_sample, img1_i, img2_i)
                entropy_i = torch.sum(log_var_i, dim=(1, 2, 3)) / 2
                mean_elbo_i = energy_i.mean() - self._entropy_weight*entropy_i.mean()

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


class MultiScaleElboUpflow(Elbo):
    def __init__(self,
                 args,
                 num_scales=5,
                 num_highres_scales=2,
                 coarsest_resolution_loss_weight=0.32,
                 alpha=1.0,
                 beta=1.0,
                 gamma=1.0,
                 Nsamples=1,
                 entropy_weight=1.0,
                 mask_cost=0.0):

        super(MultiScaleElboUpflow, self).__init__(args=args, alpha=alpha, beta=beta, gamma=gamma, Nsamples=Nsamples, entropy_weight=entropy_weight, mask_cost=mask_cost)
        self._num_scales = num_scales

        # ---------------------------------------------------------------------
        # start with initial scale
        # for "low-resolution" scales we apply a scale factor of 4
        # for "high-resolution" scales we apply a scale factor of 2
        #
        # e.g. for FlyingChairs  weights=[0.005, 0.01, 0.02, 0.08, 0.32]
        # ---------------------------------------------------------------------
        #self._weights = [coarsest_resolution_loss_weight]
        #num_lowres_scales = num_scales - num_highres_scales
        #for k in range(num_lowres_scales - 1):
        #    self._weights += [self._weights[-1] / 4]
        #for k in range(num_highres_scales):
        #    self._weights += [self._weights[-1] / 2]
        #self._weights.reverse()
        self._weights = [1.0, 1.0, 1.0, 1.0, 1.0]
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
                mean_i, log_var_i = output_i
                mean_i = upsample2d_as(mean_i, img1, mode="bilinear")
                log_var_i = upsample2d_as(log_var_i, img1, mode="bilinear")

                # Evaluate ELBO for a given scale
                flow_sample = self.reparam(mean_i, log_var_i)
                energy_i = self.energy(flow_sample, img1, img2)
                entropy_i = torch.sum(log_var_i, dim=(1, 2, 3)) / 2
                mean_elbo_i = energy_i.mean() - self._entropy_weight*entropy_i.mean()

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
