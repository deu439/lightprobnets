from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .flownet1s import FlowNetS
from .flownet_helpers import upsample2d_as


class FlowNetProbFB(nn.Module):
    def __init__(self, args):
        super(FlowNetProbFB, self).__init__()
        self._flownets = FlowNetS(args, num_pred=4)

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
            flow2f_mean, flow2f_log_variance = flow2f.chunk(chunks=2, dim=1)
            flow1f_mean = upsample2d_as(flow2f_mean, im1, mode="bilinear")
            flow1f_log_variance = upsample2d_as(flow2f_log_variance, im1, mode="bilinear")

            output_dict['flow1f'] = flow1f_mean, flow1f_log_variance
            output_dict['flow2f'] = flow2f.chunk(chunks=2, dim=1)
            output_dict['flow3f'] = flow3f.chunk(chunks=2, dim=1)
            output_dict['flow4f'] = flow4f.chunk(chunks=2, dim=1)
            output_dict['flow5f'] = flow5f.chunk(chunks=2, dim=1)
            output_dict['flow6f'] = flow6f.chunk(chunks=2, dim=1)

            # Backward flow =====
            flow2b, flow3b, flow4b, flow5b, flow6b = self._flownets(inputf)

            # Upsample to get flow at full resolution
            flow2b_mean, flow2b_log_variance = flow2b.chunk(chunks=2, dim=1)
            flow1b_mean = upsample2d_as(flow2b_mean, im2, mode="bilinear")
            flow1b_log_variance = upsample2d_as(flow2b_log_variance, im2, mode="bilinear")

            output_dict['flow1b'] = flow1b_mean, flow1b_log_variance
            output_dict['flow2b'] = flow2b.chunk(chunks=2, dim=1)
            output_dict['flow3b'] = flow3b.chunk(chunks=2, dim=1)
            output_dict['flow4b'] = flow4b.chunk(chunks=2, dim=1)
            output_dict['flow5b'] = flow5b.chunk(chunks=2, dim=1)
            output_dict['flow6b'] = flow6b.chunk(chunks=2, dim=1)

        else:
            # Forward flow =====
            flow2f = self._flownets(inputf)
            flow2f_mean, flow2f_log_variance = flow2f.chunk(chunks=2, dim=1)
            flow1f_mean = upsample2d_as(flow2f_mean, im1, mode="bilinear")
            flow1f_log_variance = upsample2d_as(flow2f_log_variance, im1, mode="bilinear")
            output_dict['flow1f'] = flow1f_mean, flow1f_log_variance

            # Backward flow =====
            flow2b = self._flownets(inputb)
            flow2b_mean, flow2b_log_variance = flow2b.chunk(chunks=2, dim=1)
            flow1b_mean = upsample2d_as(flow2b_mean, im2, mode="bilinear")
            flow1b_log_variance = upsample2d_as(flow2b_log_variance, im2, mode="bilinear")
            output_dict['flow1b'] = flow1b_mean, flow1b_log_variance

        return output_dict
