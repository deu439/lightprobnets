from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .tinyflownet1s import TinyFlowNetS
from .flownet_helpers import upsample2d_as


class TinyFlowNetProbout(nn.Module):
    def __init__(self, args):
        super(TinyFlowNetProbout, self).__init__()
        self._flownets = TinyFlowNetS(args, num_pred=4)

    def forward(self, input_dict):
        im1 = input_dict['input1']
        im2 = input_dict['input2']
        inputs = torch.cat((im1, im2), dim=1)

        output_dict = {}
        flow1 = self._flownets(inputs)
        mean, log_var = flow1.chunk(chunks=2, dim=1)

        mean = upsample2d_as(mean, im1, mode="bilinear")
        log_var = upsample2d_as(log_var, im1, mode="bilinear")
        output_dict["flow1"] = mean, log_var

        return output_dict
