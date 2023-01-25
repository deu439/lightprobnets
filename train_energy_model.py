import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from datasets.tinyflyingchairs import TinyFlyingChairsTrain
from datasets.tinyflyingchairs import TinyFlyingChairsValid

import random
from tqdm import tqdm
import numpy as np

from losses.resample2d_package.resample2d import Resample2d


# User options
learning_rate = 1.0
batch_size = 20
epochs = 20
max_flow = 300.0
device = 'cuda'
# End of user option


# Charbonnier penalty
def penalty(x):
    return torch.sqrt(x + 1e-5)


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Warping module
        self._resample2d = Resample2d()

        # Initialize model parameters
        self.log_weights = nn.Parameter(torch.randn(2)-1.0)

        # Convolution kernels for horizontal and vertical derivatives
        kernel_dx = torch.tensor([[[[-1, 1]], [[0, 0]]], [[[0, 0]], [[-1, 1]]]], dtype=torch.float32)
        kernel_dy = torch.tensor([[[[-1], [1]], [[0], [0]]], [[[0], [0]], [[-1], [1]]]], dtype=torch.float32)
        self.register_buffer('_kernel_dx', kernel_dx)
        self.register_buffer('_kernel_dy', kernel_dy)

    # Energy function
    def energy(self, flow, img1, img2):
        """
        flow: tensor of size (batch, 2, height, width)
        img1: tensor of size (batch, 3, height, width)
        img2: tensor of size (batch, 3, height, width)
        """
        _, _, height, width = img1.size()
        Nsamples = int(flow.size(0) / img1.size(0))

        # Warp img2 according to flow
        img2_warp = self._resample2d(img2, flow.contiguous())
        A = torch.sum((img1.repeat(Nsamples, 1, 1, 1) - img2_warp) ** 2, dim=1)
        data_term = torch.sum(penalty(A), dim=(1, 2))

        B = F.pad(torch.sum(F.conv2d(flow, self._kernel_dx) ** 2, dim=1), (0, 1, 0, 0)) \
            + F.pad(torch.sum(F.conv2d(flow, self._kernel_dy) ** 2, dim=1), (0, 0, 0, 1))
        smooth_term = torch.sum(penalty(B), dim=(1, 2))

        out = torch.exp(self.log_weights[0]) * data_term + torch.exp(self.log_weights[1]) * smooth_term
        return out / (height * width)

    def forward(self, x):
        flow, img1, img2 = x.split([2,3,3], dim=1)
        return self.energy(flow*max_flow, img1, img2)


class ContrastiveLoss(nn.Module):
    def __init__(self, model, img_shape, sample_size, max_len=8192, steps=60, step_size=10, eta=0.1):
        super().__init__()
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = max_len
        self.steps = steps
        self.step_size = step_size
        self.eta = eta  # Regularization weight
        self.examples = [(torch.rand((1,)+img_shape)*2-1) for _ in range(self.sample_size)]

    def sample_new_exmps(self):
        """
        Function for getting a new batch of "fake" images.
        Inputs:
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(self.sample_size, 0.05)
        rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=self.sample_size - n_new), dim=0)
        # inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(device)
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to('cuda')

        # Perform MCMC sampling
        inp_imgs = self.generate_samples(inp_imgs)

        # Add new images to the buffer and remove old ones if needed
        self.examples = list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
        self.examples = self.examples[:self.max_len]
        return inp_imgs

    def generate_samples(self, inp_imgs, return_img_per_step=False):
        """
        Function for sampling images for a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        is_training = self.model.training
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        inp_imgs.requires_grad = True

        # Enable gradient calculation if not already the case
        #had_gradients_enabled = torch.is_grad_enabled()
        #torch.set_grad_enabled(True)
        with torch.enable_grad():

            # We use a buffer tensor in which we generate noise each loop iteration.
            # More efficient than creating a new tensor every iteration.
            noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)

            # List for storing generations at each step (for later analysis)
            imgs_per_step = []

            # Loop over K (steps)
            for _ in range(self.steps):
                # Part 1: Add noise to the input.
                noise.normal_(0, 0.05)
                inp_imgs.data.add_(noise.data)
                inp_imgs.data.clamp_(min=-1.0, max=1.0)

                # Part 2: calculate gradients for the current input.
                out_imgs = -self.model(inp_imgs)
                out_imgs.sum().backward()
                #inp_imgs.grad.data.clamp_(-0.03, 0.03)  # For stabilizing and preventing too high gradients

                # Apply gradients to our current samples
                inp_imgs.data.add_(-self.step_size * inp_imgs.grad.data)
                inp_imgs.grad.detach_()
                inp_imgs.grad.zero_()
                inp_imgs.data.clamp_(min=-1.0, max=1.0)

                if return_img_per_step:
                    imgs_per_step.append(inp_imgs.clone().detach())

            # Reactivate gradients for parameters for training
            for p in self.model.parameters():
                p.requires_grad = True
            self.model.train(is_training)

        # Reset gradient calculation to setting before this function
        #torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs.detach()

    def forward(self, pos_output):
        neg_input = self.sample_new_exmps()
        neg_output = self.model(neg_input)

        # Evaluate the loss
        reg_loss = self.eta * (pos_output ** 2 + neg_output ** 2).mean()
        cdiv_loss = pos_output.mean() - neg_output.mean()
        loss = reg_loss + cdiv_loss

        if self.model.training:
            return loss
        else:
            return loss, reg_loss, cdiv_loss


def train_loop(epoch, dataloader, model, optimizer, loss_fn):
    model.train()
    num_batches = len(dataloader)
    pbar = tqdm(enumerate(dataloader), total=num_batches)
    for batch, dict in pbar:
        # Construct real data tensor & normalize
        flow = dict['target1'].to(device)
        img1 = dict['input1'].to(device)
        img2 = dict['input2'].to(device)
        # To -1, 1 range
        img1 = 2*img1 - 1.0
        img2 = 2*img2 - 1.0
        flow = flow / max_flow
        pos_input = torch.cat([flow, img1, img2], dim=1)

        # Forward
        pos_output = model(pos_input)
        loss = loss_fn(pos_output)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Update
        optimizer.step()

        # Update progress bar
        log_weights = model.log_weights.detach().cpu().numpy()
        pbar.set_description("epoch: %d, batch: %d, loss: %e, alpha: %e, beta: %e"
            % (epoch+1, batch, loss.item(), np.exp(log_weights[0]), np.exp(log_weights[1])))


def test_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    avg_loss = 0
    avg_reg_loss = 0
    avg_cdiv_loss = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=num_batches)
        for batch, dict in pbar:
            # Construct real data tensor & normalize
            flow = dict['target1'].to(device)
            img1 = dict['input1'].to(device)
            img2 = dict['input2'].to(device)
            # To -1, 1 range
            img1 = 2 * img1 - 1.0
            img2 = 2 * img2 - 1.0
            flow = flow / max_flow
            pos_input = torch.cat([flow, img1, img2], dim=1)

            # Forward
            pos_output = model(pos_input)
            loss, reg_loss, cdiv_loss = loss_fn(pos_output)
            avg_loss += loss
            avg_reg_loss += reg_loss
            avg_cdiv_loss += cdiv_loss

            # Update progress bar
            pbar.set_description("Validation: batch: %d" % batch)

    avg_loss /= num_batches
    avg_reg_loss /= num_batches
    avg_cdiv_loss /= num_batches
    print("Loss: %.3f, Reg. loss: %.3f, Cdiv. loss: %.3f" % (avg_loss, avg_reg_loss, avg_cdiv_loss))


def main():
    train_data = TinyFlyingChairsTrain(None, '/home/deu/FlyingChairs_release/downscaled_data',photometric_augmentations=False)
    valid_data = TinyFlyingChairsValid(None, '/home/deu/FlyingChairs_release/downscaled_data',photometric_augmentations=False)
    train_loader = DataLoader(train_data, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    # Determine data size
    image_size = train_data.__getitem__(0)['input1'].size()
    data_size = (8, image_size[1], image_size[2])

    model = EnergyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = MultiStepLR(optimizer, milestones=[1, 5, 10], gamma=0.1)
    loss = ContrastiveLoss(model, data_size, batch_size).to(device)

    for t in range(epochs):
        train_loop(t, train_loader, model, optimizer, loss)
        test_loop(valid_loader, model, loss)

        with torch.no_grad():
            rand_imgs = torch.rand(data_size) * 2 - 1
            sample = loss.generate_samples(rand_imgs)
            sample_flow = sample[0:2, :, :]*max_flow
            sample_img1 = (sample[2:5, :, :] + 1.0) / 2.0
            sample_img2 = (sample[5:8, :, :] + 1.0) / 2.0

    #    scheduler.step()


if __name__ == '__main__':
    main()