import torch
import torch.nn as nn

import numpy as np
import random


class ContrastiveLoss(nn.Module):
    def __init__(self, eta=0.1):
        super().__init__()
        self.eta = eta  # Regularization weight

    def forward(self, output_dict, target_dict):
        pos_output = output_dict['pos_energy']
        neg_output = output_dict['neg_energy']

        # Evaluate the loss
        reg_loss = self.eta * (pos_output ** 2 + neg_output ** 2).mean()
        cdiv_loss = pos_output.mean() - neg_output.mean()
        loss = reg_loss + cdiv_loss

        loss_dict = dict()
        loss_dict['total_loss'] = loss
        loss_dict['reg_loss'] = reg_loss
        loss_dict['cdiv_loss'] = cdiv_loss

        return loss_dict
