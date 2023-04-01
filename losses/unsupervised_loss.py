from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .endpoint_error import elementwise_epe
from .resample2d_package.resample2d import Resample2d


def upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return F.interpolate(inputs, [h, w], mode=mode, align_corners=True)


# Charbonnier penalty
def penalty(x):
    return torch.sqrt(x + 1e-5)


class Unsupervised(nn.Module):
    def __init__(self, args, alpha=1.0, beta=1.0, gamma=1.0):
        super(Unsupervised, self).__init__()
        self._args = args
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
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
        A = torch.sum((img1 - img2_warp)**2, dim=1)
        data_term = torch.sum(penalty(A), dim=(1, 2))

        B = F.pad(torch.sum(F.conv2d(flow, self._kernel_dx)**2, dim=1), (0,1,0,0)) \
            + F.pad(torch.sum(F.conv2d(flow, self._kernel_dy)**2, dim=1), (0,0,0,1))
        smooth_term = torch.sum(penalty(B), dim=(1, 2))

        img1_gx = F.conv2d(img1, self._kernel_gx, padding=(0, 2))
        img1_gy = F.conv2d(img1, self._kernel_gy, padding=(2, 0))
        img2_warp_gx = F.conv2d(img2_warp, self._kernel_gx, padding=(0, 2))
        img2_warp_gy = F.conv2d(img2_warp, self._kernel_gy, padding=(2, 0))
        C = torch.sum((img1_gx - img2_warp_gx)**2, dim=1) \
            + torch.sum((img1_gy - img2_warp_gy)**2, dim=1)
        gradient_term = torch.sum(penalty(C), dim=(1, 2))

        return self._alpha * data_term + self._beta * smooth_term + self._gamma * gradient_term

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        target = target_dict["target1"]
        img1 = target_dict["input1"]
        img2 = target_dict["input2"]
        flow1 = output_dict["flow1"]

        # Evaluate ELBO
        energy = self.energy(flow1, img1, img2)
        loss_dict["energy"] = energy.mean()

        # Calculate epe for validation
        epe = elementwise_epe(flow1, target)
        mean_epe = epe.mean()
        loss_dict["epe"] = mean_epe

        return loss_dict
