import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.aux import flow_warp

import numpy as np
import random


# Charbonnier penalty
def penalty(x):
    return torch.sqrt(x + 1e-5)


class EnergyModel(nn.Module):
    def __init__(self, max_flow=80.0, max_len=1000, steps=200, step_size=10):
        super().__init__()

        # Initialize model parameters
        #self.log_weights = nn.Parameter(torch.randn(2)+10.0)
        self.log_weights = nn.Parameter(torch.tensor([1.0, 1.0]))

        # Convolution kernels for horizontal and vertical derivatives
        kernel_dx = torch.tensor([[[[-1, 1]], [[0, 0]]], [[[0, 0]], [[-1, 1]]]], dtype=torch.float32)
        kernel_dy = torch.tensor([[[[-1], [1]], [[0], [0]]], [[[0], [0]], [[-1], [1]]]], dtype=torch.float32)
        self.register_buffer('_kernel_dx', kernel_dx)
        self.register_buffer('_kernel_dy', kernel_dy)

        # Sample buffer
        self.buffer = None

        # Parameters
        self.max_flow = max_flow
        self.max_len = max_len
        self.steps = steps
        self.step_size = step_size

    # Energy function
    def energy(self, flow, img1, img2):
        """
        flow: tensor of size (batch, 2, height, width)
        img1: tensor of size (batch, 3, height, width)
        img2: tensor of size (batch, 3, height, width)
        """
        _, _, height, width = img1.size()
        Nsamples = int(flow.size(0) / img1.size(0))

        # Adapt flow and img domain
        flow = flow * self.max_flow
        img1 = (img1 + 1.0) / 2.0
        img2 = (img2 + 1.0) / 2.0

        # Warp img2 according to flow
        #img2_warp = self._resample2d(img2, flow.contiguous())
        img2_warp = img2
        A = torch.sum((img1.repeat(Nsamples, 1, 1, 1) - img2_warp) ** 2, dim=1)
        data_term = torch.sum(penalty(A), dim=(1, 2))

        B = F.pad(torch.sum(F.conv2d(flow, self._kernel_dx) ** 2, dim=1), (0, 1, 0, 0)) \
            + F.pad(torch.sum(F.conv2d(flow, self._kernel_dy) ** 2, dim=1), (0, 0, 0, 1))
        smooth_term = torch.sum(penalty(B), dim=(1, 2))

        out = torch.exp(self.log_weights[0]) * data_term + torch.exp(self.log_weights[1]) * smooth_term
        return out / (height * width)

    def langevin_sample(self, init, img2, steps, step_size):
        """
        Function for sampling images from a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            init - Initialization of the Markov chain
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        for p in self.parameters():
            p.requires_grad = False
        init.requires_grad = True

        # Enable gradient calculation
        with torch.enable_grad():

            # We use a buffer tensor in which we generate noise each loop iteration.
            # More efficient than creating a new tensor every iteration.
            noise = torch.randn(init.shape, device=init.device)

            # Loop over K (steps)
            for _ in range(steps):
                # Part 1: Add noise to the input.
                noise.normal_(0, 0.05)
                init.data.add_(noise.data)
                init.data.clamp_(min=-1.0, max=1.0)

                # Part 2: calculate gradients for the current input.
                flow, img1 = init.split([2, 3], dim=1)
                energy = self.energy(flow, img1, img2)
                energy.sum().backward()
                init.grad.data.clamp_(-0.03, 0.03)  # For stabilizing and preventing too high gradients

                # Apply gradients to our current samples
                init.data.add_(-step_size * init.grad.data)
                init.grad.detach_()
                init.grad.zero_()
                init.data.clamp_(min=-1.0, max=1.0)

        # Reactivate gradients for parameters for training
        for p in self.parameters():
            p.requires_grad = True

        return init.detach()

    def sample_new_exmps(self, sample_shape, img2):
        batch_size, channels, height, width = sample_shape

        # If there are no samples in the buffer, initialize it randomly
        if self.buffer is None:
            self.buffer = torch.rand(batch_size, 2+3, height, width) * 2.0 - 1.0

        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(batch_size, 0.05)
        new_samples = torch.rand(n_new, 2+3, height, width) * 2.0 - 1.0
        i_old = np.random.choice(self.buffer.size(0), batch_size-n_new)
        old_samples = self.buffer[i_old]
        init = torch.cat([new_samples, old_samples], dim=0).detach().to('cuda')

        # Perform MCMC sampling
        init = self.langevin_sample(init, img2, self.steps, self.step_size)

        # Add new images to the buffer and remove old ones if needed
        self.buffer = torch.cat([init.to(torch.device("cpu")), self.buffer], dim=0)
        self.buffer = self.buffer[:self.max_len]

        return init

    def forward(self, input_dict):
        # Get individual elements of positive example
        flow_pos = input_dict['target1'] / self.max_flow
        img1_pos = input_dict['input1'] * 2.0 - 1.0
        img2 = input_dict['input2'] * 2.0 - 1.0

        # Generate negative example
        sample_shape = img1_pos.size()
        neg_example = self.sample_new_exmps(sample_shape, img2)
        flow_neg, img1_neg = neg_example.split([2, 3], dim=1)

        output_dict = dict()
        output_dict['pos_energy'] = self.energy(flow_pos, img1_pos, img2)
        output_dict['neg_energy'] = self.energy(flow_neg, img1_neg, img2)
        output_dict['flow_neg'] = flow_neg * self.max_flow
        output_dict['img1_neg'] = (img1_neg + 1.0) / 2.0
        output_dict['img2_neg'] = (img2 + 1.0) / 2.0
        output_dict['weights'] = {'alpha': torch.exp(self.log_weights[0]), 'beta': torch.exp(self.log_weights[1])}
        return output_dict
