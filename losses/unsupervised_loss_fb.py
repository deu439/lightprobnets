from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .endpoint_error import elementwise_epe
from .aux import flow_warp, border_mask, color_loss, gradient_loss, census_loss, smooth_grad_1st, smooth_grad_2nd, get_occu_mask_backward


def upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return F.interpolate(inputs, [h, w], mode=mode, align_corners=True)


class UnsupervisedFB(nn.Module):
    def __init__(self, args, color_weight=0.0, gradient_weight=0.0, census_weight=1.0, census_radius=3,
                 smooth_1st_weight=1.0, smooth_2nd_weight=0.0, edge_weight=150.0):
        super(UnsupervisedFB, self).__init__()
        self._args = args
        self._color_weight = color_weight
        self._gradient_weight = gradient_weight
        self._census_weight = census_weight
        self._census_radius = census_radius
        self._smooth_1st_weight = smooth_1st_weight
        self._smooth_2nd_weight = smooth_2nd_weight
        self._edge_weight = edge_weight

    # Energy function
    def energy(self, flowf, flowb, img1, img2):
        """
        flow: tensor of size (batch, 2, height, width)
        img1: tensor of size (batch, 3, height, width)
        img2: tensor of size (batch, 3, height, width)
        """

        # Calculate border mask
        mask = border_mask(flowf).detach()

        # Calculate occlusion mask - do not propagate gradient into occlusion mask calculation
        occ_mask = get_occu_mask_backward(flowb).detach()
        mask = mask * (1.0 - occ_mask)

        # Warp img2 according to flow
        img2_warp = flow_warp(img2, flowf)

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
            energy_dict['smooth_1st'] = self._smooth_1st_weight * smooth_grad_1st(flowf, img1, edge_weight=self._edge_weight)
        if self._smooth_2nd_weight > 0:
            energy_dict['smooth_2nd'] = self._smooth_2nd_weight * smooth_grad_2nd(flowf, img1, edge_weight=self._edge_weight)

        # Total loss
        energy = 0
        for key, value in energy_dict.items():
            energy += value

        return energy, energy_dict

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        target = target_dict["target1"]
        # Use original images for unsupervised loss
        img1 = target_dict["input1o"]
        img2 = target_dict["input2o"]
        flow1f = output_dict["flow1f"]
        flow1b = output_dict["flow1b"]

        # Evaluate energy
        energy, energy_dict = self.energy(flow1f, flow1b, img1, img2)
        loss_dict["energy"] = energy

        # Calculate epe
        epe = elementwise_epe(flow1f, target)
        mean_epe = epe.mean()
        loss_dict["epe"] = mean_epe

        # Return everything if in validation
        if not self.training:
            loss_dict = {**loss_dict, **energy_dict}

        return loss_dict


class UnsupervisedSequenceFB(UnsupervisedFB):
    def __init__(self, args, color_weight=0.0, gradient_weight=0.0, census_weight=1.0, census_radius=3,
                 smooth_1st_weight=1.0, smooth_2nd_weight=0.0, edge_weight=150.0, decay=0.8):

        self._decay = decay
        super(UnsupervisedSequenceFB, self).__init__(args, color_weight, gradient_weight, census_weight, census_radius,
              smooth_1st_weight, smooth_2nd_weight, edge_weight)

    def forward(self, output_dict, target_dict):
        loss_dict = {}
        target = target_dict["target1"]
        img1 = target_dict["input1o"]
        img2 = target_dict["input2o"]

        # Count the number of RAFT iterations
        niter = len([key for key in output_dict if key.startswith('flow')]) // 2

        sequence_energy = 0
        sequence_energy_dict = {}
        for iter in range(niter):
            flowf = output_dict[f'flow{iter+1}f']
            flowb = output_dict[f'flow{iter+1}b']
            energy, energy_dict = self.energy(flowf, flowb, img1, img2)

            # Sum energies over all iterations with lower weight for early iterations
            sequence_energy += (self._decay ** iter) * energy

            # Sum also all energy components if in validation
            if not self.training:
                # Initialize
                if iter == 0:
                    sequence_energy_dict = energy_dict.copy()
                    continue
                # Sum with decay
                for key in energy_dict:
                    sequence_energy_dict[key] += (self._decay ** iter) * energy_dict[key]

        # Store into loss dictionary
        loss_dict["energy"] = sequence_energy

        # Calculate epe
        flow1 = output_dict['flow1f']
        epe = elementwise_epe(flow1, target)
        mean_epe = epe.mean()
        loss_dict["epe"] = mean_epe

        # Return everything if in validation
        if not self.training:
            loss_dict = {**loss_dict, **sequence_energy_dict}

        return loss_dict
