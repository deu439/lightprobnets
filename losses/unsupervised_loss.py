from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .endpoint_error import elementwise_epe
from .resample2d_package.resample2d import Resample2d
from .aux import border_mask, color_loss, gradient_loss, census_loss, robust_l1, abs_robust_loss, smooth_grad_1st, smooth_grad_2nd


def upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return F.interpolate(inputs, [h, w], mode=mode, align_corners=True)


class Unsupervised(nn.Module):
    def __init__(self, args, color_weight=0.0, gradient_weight=0.0, census_weight=1.0, census_radius=3,
                 smooth_1st_weight=1.0, smooth_2nd_weight=0.0, edge_weight=4.0):
        super(Unsupervised, self).__init__()
        self._args = args
        self._color_weight = color_weight
        self._gradient_weight = gradient_weight
        self._census_weight = census_weight
        self._census_radius = census_radius
        self._smooth_1st_weight = smooth_1st_weight
        self._smooth_2nd_weight = smooth_2nd_weight
        self._edge_weight = edge_weight
        self._resample2d = Resample2d()

    # Energy function
    def energy(self, flow, img1, img2):
        """
        flow: tensor of size (batch, 2, height, width)
        img1: tensor of size (batch, 3, height, width)
        img2: tensor of size (batch, 3, height, width)
        """

        # Calculate border mask
        mask = border_mask(flow)

        # Warp img2 according to flow - on cpu-located tensors warping module fails without warning!
        assert(img2.is_cuda and flow.is_cuda)
        img2_warp = self._resample2d(img2, flow.contiguous())

        # Data losses
        energy_dict = {}
        if self._color_weight > 0:
            energy_dict['color'] = self._color_weight * color_loss(img1, img2_warp, mask)
        if self._gradient_weight > 0:
            energy_dict['gradient'] = self._gradient_weight * gradient_loss(img1, img2_warp, mask)
        if self._census_weight > 0:
            energy_dict['census'] = self._census_weight * census_loss(img1, img2_warp, mask, radius=self._census_radius)

        # Smoothness losses
        if self._smooth_1st_weight > 0:
            energy_dict['smooth_1st'] = self._smooth_1st_weight * smooth_grad_1st(flow, img1, edge_weight=self._edge_weight)
        if self._smooth_2nd_weight > 0:
            energy_dict['smooth_2nd'] = self._smooth_2nd_weight * smooth_grad_2nd(flow, img1, edge_weight=self._edge_weight)

        # Total loss
        energy = 0
        for key, value in energy_dict.items():
            energy += value

        return energy, energy_dict

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        target = target_dict["target1"]
        img1 = target_dict["input1"]
        img2 = target_dict["input2"]
        flow1 = output_dict["flow1"]

        # Evaluate loss
        energy, energy_dict = self.energy(flow1, img1, img2)
        loss_dict["energy"] = energy

        # Calculate epe for validation
        epe = elementwise_epe(flow1, target)
        mean_epe = epe.mean()
        loss_dict["epe"] = mean_epe

        # Return everything if in validation
        if not self.training:
            loss_dict = {**loss_dict, **energy_dict}

        return loss_dict
