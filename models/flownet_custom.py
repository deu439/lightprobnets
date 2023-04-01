from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .flownet1s import FlowNetS
from .flownet_helpers import upsample2d_as


class FlowNetCustom(nn.Module):
    def __init__(self, args, div_flow=0.05):
        super(FlowNetCustom, self).__init__()
        self._flownets = FlowNetS(args)

    def forward(self, input_dict):
        im1 = input_dict['input1']
        im2 = input_dict['input2']
        inputs = torch.cat((im1, im2), dim=1)

        output_dict = {}
        if self.training:
            flow2, flow3, flow4, flow5, flow6 = self._flownets(inputs)
            output_dict['flow1'] = upsample2d_as(flow2, im1, mode="bilinear")
            output_dict['flow2'] = flow2
            output_dict['flow3'] = flow3
            output_dict['flow4'] = flow4
            output_dict['flow5'] = flow5
            output_dict['flow6'] = flow6
        else:
            flow2 = self._flownets(inputs)
            output_dict['flow1'] = upsample2d_as(flow2, im1, mode="bilinear")

        return output_dict
