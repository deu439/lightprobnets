from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .flownet1s import FlowNetS
from .flownet_helpers import upsample2d_as


class FlowNetProbOutCustom(nn.Module):
    def __init__(self, args, min_variance=1e-3, log_variance=True):
        super(FlowNetProbOutCustom, self).__init__()
        self._flownets = FlowNetS(args, num_pred=4)

    def forward(self, input_dict):
        im1 = input_dict['input1']
        im2 = input_dict['input2']
        inputs = torch.cat((im1, im2), dim=1)

        output_dict = {}
        if self.training:
            flow2, flow3, flow4, flow5, flow6 = self._flownets(inputs)

            # Upsample to get flow at full resolution
            flow2_mean, flow2_log_variance = flow2.chunk(chunks=2, dim=1)
            flow1_mean = upsample2d_as(flow2_mean, im1, mode="bilinear")
            flow1_log_variance = upsample2d_as(flow2_log_variance, im1, mode="bilinear")

            output_dict['flow1'] = flow1_mean, flow1_log_variance
            output_dict['flow2'] = flow2.chunk(chunks=2, dim=1)
            output_dict['flow3'] = flow3.chunk(chunks=2, dim=1)
            output_dict['flow4'] = flow4.chunk(chunks=2, dim=1)
            output_dict['flow5'] = flow5.chunk(chunks=2, dim=1)
            output_dict['flow6'] = flow6.chunk(chunks=2, dim=1)

        else:
            flow2 = self._flownets(inputs)
            flow2_mean, flow2_log_variance = flow2.chunk(chunks=2, dim=1)

            flow1_mean = upsample2d_as(flow2_mean, im1, mode="bilinear")
            flow1_log_variance = upsample2d_as(flow2_log_variance, im1, mode="bilinear")

            output_dict['flow1'] = flow1_mean, flow1_log_variance

        return output_dict
