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


def sigmoid(x, t, mu=1.0):
    return torch.special.expit(mu*(x-t))


def border_mask(flow, soft_threshold=True):
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
    if soft_threshold:
        mask_x = sigmoid(Xp, -0.5) * (1.0 - sigmoid(Xp, w - 0.5))
        mask_y = sigmoid(Yp, -0.5) * (1.0 - sigmoid(Yp, h - 0.5))
    else:
        mask_x = (Xp > -0.5) & (Xp < w-0.5)
        mask_y = (Yp > -0.5) & (Yp < h-0.5)
    return mask_x * mask_y


class ElboFB(nn.Module):
    def __init__(self, args, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, Nsamples=1, entropy_weight=1.0, mask_cost=1.0, soft_threshold=True):

        super(ElboFB, self).__init__()
        self._args = args
        self._Normal = torch.distributions.Normal(0, 1)
        self._Normal.loc = self._Normal.loc.cuda() # hack to get sampling on the GPU
        self._Normal.scale = self._Normal.scale.cuda()
        self._Nsamples = Nsamples
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta
        self._entropy_weight = entropy_weight
        self._mask_cost = mask_cost
        self._soft_threshold = soft_threshold
        if not soft_threshold:
            assert(mask_cost == 0)
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

    def energy_fb(self, flowf, flowb, img1, img2):
        """
        flowf: forward flow - tensor of size (batch, 2, height, width)
        flowb: backward flow - tensor of size (batch, 2, height, width)
        img1: tensor of size (batch, 3, height, width)
        img2: tensor of size (batch, 3, height, width)
        """
        flowf = flowf.contiguous()
        flowb = flowb.contiguous()
        assert(flowf.is_cuda and flowb.is_cuda)
        assert(img1.is_cuda and img2.is_cuda)

        energy_dict = dict()

        # Forward part ======
        # Calculate forward border mask
        maskf = border_mask(flowf, soft_threshold=self._soft_threshold)

        # Calculate forward occlusion mask
        flowb_warp = self._resample2d(flowb, flowf)
        magf = torch.sum(flowf**2 + flowb_warp**2, dim=1)
        flowf_diff = flowf + flowb_warp
        occf_thresh = 0.01*magf + 0.5
        if self._soft_threshold:
            occf = 1.0 - sigmoid(torch.sum(flowf_diff ** 2, dim=1), occf_thresh)
        else:
            occf = torch.sum(flowf_diff**2, dim=1) <= occf_thresh   # True if there is no occlusion

        maskf = maskf * occf    # Combine to get only valid pixels

        # Penalize occluded pixels to prevent trivial solutions
        energy_dict["mask_termf"] = mask_termf = self._mask_cost * torch.sum(1.0 - maskf, dim=(1, 2)).mean()

        # Data term
        img2_warp = self._resample2d(img2, flowf)
        Af = torch.sum((img1.repeat(self._Nsamples, 1, 1, 1) - img2_warp)**2, dim=1)
        energy_dict["data_termf"] = data_termf = self._alpha * torch.sum(penalty(Af) * maskf, dim=(1, 2)).mean()

        # Smoothness term
        Bf = F.pad(torch.sum(F.conv2d(flowf, self._kernel_dx)**2, dim=1), (0,1,0,0)) \
            + F.pad(torch.sum(F.conv2d(flowf, self._kernel_dy)**2, dim=1), (0,0,0,1))
        energy_dict["smooth_termf"] = smooth_termf = self._beta * torch.sum(penalty(Bf), dim=(1, 2)).mean()

        # Gradient term
        img1_gx = F.conv2d(img1, self._kernel_gx, padding=(0, 2))
        img1_gy = F.conv2d(img1, self._kernel_gy, padding=(2, 0))
        img2_warp_gx = F.conv2d(img2_warp, self._kernel_gx, padding=(0, 2))
        img2_warp_gy = F.conv2d(img2_warp, self._kernel_gy, padding=(2, 0))
        Cf = torch.sum((img1_gx.repeat(self._Nsamples, 1, 1, 1) - img2_warp_gx)**2, dim=1) \
            + torch.sum((img1_gy.repeat(self._Nsamples, 1, 1, 1) - img2_warp_gy)**2, dim=1)
        energy_dict["gradient_termf"] = gradient_termf = self._gamma * torch.sum(penalty(Cf) * maskf, dim=(1, 2)).mean()

        # Backward part =====
        # Calculate forward border mask
        maskb = border_mask(flowb, soft_threshold=self._soft_threshold)

        # Calculate forward occlusion mask
        flowf_warp = self._resample2d(flowf, flowb)
        magb = torch.sum(flowb**2 + flowf_warp**2, dim=1)
        flowb_diff = flowb + flowf_warp
        occb_thresh = 0.01*magb + 0.5
        if self._soft_threshold:
            occb = 1.0 - sigmoid(torch.sum(flowb_diff ** 2, dim=1), occb_thresh)
        else:
            occb = torch.sum(flowb_diff**2, dim=1) <= occb_thresh   # True if there is no occlusion

        maskb = maskb * occb    # Combine to get only valid pixels

        # Penalize occluded pixels to prevent trivial solutions
        energy_dict["mask_termb"] = mask_termb = self._mask_cost * torch.sum(1.0 - maskb, dim=(1, 2)).mean()

        # Data term
        img1_warp = self._resample2d(img1, flowb)
        Ab = torch.sum((img2.repeat(self._Nsamples, 1, 1, 1) - img1_warp)**2, dim=1)
        energy_dict["data_termb"] = data_termb = self._alpha * torch.sum(penalty(Ab) * maskb, dim=(1, 2)).mean()

        # Smoothness term
        Bb = F.pad(torch.sum(F.conv2d(flowb, self._kernel_dx)**2, dim=1), (0,1,0,0)) \
            + F.pad(torch.sum(F.conv2d(flowb, self._kernel_dy)**2, dim=1), (0,0,0,1))
        energy_dict["smooth_termb"] = smooth_termb = self._beta * torch.sum(penalty(Bb), dim=(1, 2)).mean()

        # Gradient term
        img2_gx = F.conv2d(img2, self._kernel_gx, padding=(0, 2))
        img2_gy = F.conv2d(img2, self._kernel_gy, padding=(2, 0))
        img1_warp_gx = F.conv2d(img1_warp, self._kernel_gx, padding=(0, 2))
        img1_warp_gy = F.conv2d(img1_warp, self._kernel_gy, padding=(2, 0))
        Cb = torch.sum((img2_gx.repeat(self._Nsamples, 1, 1, 1) - img1_warp_gx)**2, dim=1) \
            + torch.sum((img2_gy.repeat(self._Nsamples, 1, 1, 1) - img1_warp_gy)**2, dim=1)
        energy_dict["gradient_termb"] = gradient_termb = self._gamma * torch.sum(penalty(Cb) * maskb, dim=(1, 2)).mean()

        # Forward-backward consistency =====
        Df = torch.sum(flowf_diff ** 2, dim=1)
        Db = torch.sum(flowb_diff ** 2, dim=1)
        energy_dict["fb_term"] = fb_term = self._delta * torch.sum(penalty(Df) * maskf + penalty(Db) * maskb, dim=(1, 2)).mean()

        energy = data_termf + data_termb + smooth_termf + smooth_termb + gradient_termf + gradient_termb \
                 + mask_termf + mask_termb + fb_term

        return energy, energy_dict, maskf, maskb

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        target = target_dict["target1"]
        img1 = target_dict["input1"]
        img2 = target_dict["input2"]
        meanf, log_varf = output_dict["flow1f"]
        meanb, log_varb = output_dict["flow1b"]

        # Evaluate ELBO
        flowf_sample = self.reparam(meanf, log_varf)
        flowb_sample = self.reparam(meanb, log_varb)
        energy, energy_dict, maskf, maskb = self.energy_fb(flowf_sample, flowb_sample, img1, img2)

        entropy = torch.sum(log_varf, dim=(1,2,3))/2 + torch.sum(log_varb, dim=(1,2,3))/2
        entropy = entropy.mean()
        loss_dict["elbo"] = energy - self._entropy_weight * entropy

        # Calculate epe for validation
        epe = elementwise_epe(meanf, target)
        mean_epe = epe.mean()
        loss_dict["epe"] = mean_epe

        # Return also the masks and energies if in validation
        if not self.training:
            loss_dict = {**loss_dict, **energy_dict, "energy": energy, "entropy": entropy}
            output_dict["maskf"] = maskf
            output_dict["maskb"] = maskb

        return loss_dict

