import numpy as np
import torch
from datasets.tinyflyingchairs import TinyFlyingChairsTrain
from models import EnergyModel
import flowpy
import matplotlib.pyplot as plt

device = 'cuda'
batch_size = 1
max_flow = 80.0


def langevin_sample(model, init, img2, steps, step_size):
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
    for p in model.parameters():
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
            noise.normal_(0, 0.01)
            init.data.add_(noise.data)
            init.data.clamp_(min=-1.0, max=1.0)

            # Part 2: calculate gradients for the current input.
            flow, img1 = init.split([2, 3], dim=1)
            energy = model.energy(flow, img1, img2)
            energy.sum().backward()
            init.grad.data.clamp_(-0.03, 0.03)  # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            init.data.add_(-step_size * init.grad.data)
            init.grad.detach_()
            init.grad.zero_()
            init.data.clamp_(min=-1.0, max=1.0)

    # Reactivate gradients for parameters for training
    for p in model.parameters():
        p.requires_grad = True

    return init.detach()


def main():
    train_data = TinyFlyingChairsTrain(None, '/home/deu/FlyingChairs_release/downscaled_data',photometric_augmentations=False)

    # Determine data size
    _, h, w = train_data[100]['input1'].size()
    model = EnergyModel().to(device)

    dict = train_data[100]
    # Construct real data tensor & normalize
    flow = dict['target1'].to(device).view((batch_size, 2, h, w))
    img1 = dict['input1'].to(device).view((batch_size, 3, h, w))
    img2 = dict['input2'].to(device).view((batch_size, 3, h, w))
    # To -1, 1 range
    img1 = 2 * img1 - 1.0
    img2 = 2 * img2 - 1.0
    flow = flow / max_flow

    # Construct random data tensor
    ran_sample = torch.rand((batch_size, 5, h, w), device=device) * 2 - 1
    flow_ran, img1_ran = ran_sample.split([2, 3], dim=1)
    neg_sample = langevin_sample(model, ran_sample.clone(), img2, steps=1000, step_size=1.0)
    flow_neg, img1_neg = neg_sample.split([2, 3], dim=1)

    with torch.no_grad():
        print('True sample:', model.energy(flow, img1, img2).item())
        print('Random sample:', model.energy(flow_ran, img1_ran, img2).item())
        print('Fake sample:', model.energy(flow_neg, img1_neg, img2).item())

    flow_neg = flow_neg[0].cpu().numpy().transpose((1, 2, 0)) * max_flow
    img1_neg = (img1_neg[0].cpu().numpy().transpose((1, 2, 0)) + 1.0) * 128.0
    img2 = (img2[0].cpu().numpy().transpose((1, 2, 0)) + 1.0) * 128.0
    fig, ax = plt.subplots(3)
    ax[0].imshow(flowpy.flow_to_rgb(flow_neg))
    ax[1].imshow(np.uint8(img1_neg))
    ax[2].imshow(np.uint8(img2))
    plt.show()


if __name__ == '__main__':
    main()