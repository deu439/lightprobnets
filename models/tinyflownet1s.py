from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .flownet_helpers import conv, deconv
from .flownet_helpers import concatenate_as, upsample2d_as
from .flownet_helpers import initialize_msra


class TinyFlowNetS(nn.Module):
    def __init__(self, args, num_pred=2):
        super(TinyFlowNetS, self).__init__()

        self._num_pred = num_pred

        def make_conv(in_planes, out_planes, kernel_size, stride):
            return conv(in_planes, out_planes, kernel_size=kernel_size,
                        stride=stride, pad=kernel_size//2, nonlinear=True, bias=True)

        self._conv1 = make_conv(  6,  64, kernel_size=3, stride=2)
        self._conv2 = make_conv( 64, 128, kernel_size=3, stride=2)
        self._conv2_1 = make_conv(128, 128, kernel_size=3, stride=1)
        self._conv3 = make_conv(128, 256, kernel_size=3, stride=2)
        self._conv3_1 = make_conv(256, 256, kernel_size=3, stride=1)

        def make_deconv(in_planes, out_planes):
            return deconv(in_planes, out_planes, kernel_size=3, stride=2, pad=1,
                          nonlinear=True, bias=False)

        self._deconv3 = make_deconv(256, 128)
        self._deconv2 = make_deconv(256, 64)
        self._deconv1 = make_deconv(128, 32)

        def make_predict(in_planes, out_planes):
            return conv(in_planes, out_planes, kernel_size=3, stride=1, pad=1,
                        nonlinear=False, bias=True)

        self._predict_flow1 = make_predict(32, num_pred)

        initialize_msra(self.modules())

    def forward(self, inputs):
        conv1 = self._conv1(inputs)
        conv2_1 = self._conv2_1(self._conv2(conv1))
        conv3_1 = self._conv3_1(self._conv3(conv2_1))

        deconv3 = self._deconv3(conv3_1)
        concat3 = concatenate_as((conv2_1, deconv3), conv2_1, dim=1)
        deconv2 = self._deconv2(concat3)
        concat2 = concatenate_as((conv1, deconv2), conv1, dim=1)
        deconv1 = self._deconv1(concat2)

        predict_flow1 = self._predict_flow1(deconv1)

        return predict_flow1


class TinyFlowNet1S(nn.Module):
    def __init__(self, args, div_flow=0.05):
        super(TinyFlowNet1S, self).__init__()
        self._flownets = TinyFlowNetS(args)
        self._div_flow = div_flow

    def forward(self, input_dict):
        im1 = input_dict['input1']
        im2 = input_dict['input2']
        inputs = torch.cat((im1, im2), dim=1)

        output_dict = {}
        flow1 = self._flownets(inputs)
        #output_dict['flow1'] = flow1
        if self.training:
            output_dict['flow1'] = upsample2d_as(flow1, im1, mode="bilinear")
        else:
            # div_flow trick
            output_dict['flow1'] = (1.0 / self._div_flow) * upsample2d_as(flow1, im1, mode="bilinear")

        return output_dict
