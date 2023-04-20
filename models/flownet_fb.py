from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .flownet1s import FlowNetS
from .flownet_helpers import upsample2d_as


class FlowNetFB(nn.Module):
    def __init__(self, args):
        super(FlowNetFB, self).__init__()
        self._flownets = FlowNetS(args)

    def forward(self, input_dict):
        im1 = input_dict['input1']
        im2 = input_dict['input2']
        inputf = torch.cat((im1, im2), dim=1)
        inputb = torch.cat((im2, im1), dim=1)

        output_dict = {}
        if self.training:
            # Forward flow =====
            flow2f, flow3f, flow4f, flow5f, flow6f = self._flownets(inputf)

            # Upsample to get flow at full resolution
            output_dict['flow1f'] = upsample2d_as(flow2f, im1, mode="bilinear")
            output_dict['flow2f'] = flow2f
            output_dict['flow3f'] = flow3f
            output_dict['flow4f'] = flow4f
            output_dict['flow5f'] = flow5f
            output_dict['flow6f'] = flow6f

            # Backward flow =====
            flow2b, flow3b, flow4b, flow5b, flow6b = self._flownets(inputb)

            # Upsample to get flow at full resolution
            output_dict['flow1b'] = upsample2d_as(flow2b, im2, mode="bilinear")
            output_dict['flow2b'] = flow2b
            output_dict['flow3b'] = flow3b
            output_dict['flow4b'] = flow4b
            output_dict['flow5b'] = flow5b
            output_dict['flow6b'] = flow6b

        else:
            # Forward flow =====
            flow2f = self._flownets(inputf)
            output_dict['flow1f'] = upsample2d_as(flow2f, im1, mode="bilinear")

            # Backward flow =====
            flow2b = self._flownets(inputb)
            output_dict['flow1b'] = upsample2d_as(flow2b, im2, mode="bilinear")

        return output_dict
