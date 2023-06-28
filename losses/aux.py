from __future__ import absolute_import
from __future__ import print_function

import torch
from torchvision.transforms.functional import rgb_to_grayscale
import torch.nn.functional as F


def robust_l1(x):
  return torch.sqrt(x**2 + 0.001**2)


def abs_robust_loss(diff, eps=0.01, q=0.4):
  """The so-called robust loss used by DDFlow."""
  return torch.pow((torch.abs(diff) + eps), q)


def flow_gradient(flow):
    kernel_dx = torch.tensor([[[[-1, 1]], [[0, 0]]], [[[0, 0]], [[-1, 1]]]]).type_as(flow)
    kernel_dy = torch.tensor([[[[-1], [1]], [[0], [0]]], [[[0], [0]], [[-1], [1]]]]).type_as(flow)
    return F.conv2d(flow, kernel_dx, padding='same'), F.conv2d(flow, kernel_dy, padding='same')


def gradient(img, stride=1):
    """
    image: (batch, channels, height, width) tensor
    """
    img_dx = img[:, :, :, stride:] - img[:, :, :, :-stride]
    img_dy = img[:, :, stride:, :] - img[:, :, :-stride, :]
    return img_dx, img_dy


def image_gradient1(img):
    kernel_dx = torch.tensor([
        [[[-1, 1]], [[0, 0]], [[0, 0]]],
        [[[0, 0]], [[-1, 1]], [[0, 0]]],
        [[[0, 0]], [[0, 0]], [[-1, 1]]]
    ]).type_as(img)
    kernel_dy = torch.tensor([
        [[[-1], [1]], [[0], [0]], [[0], [0]]],
        [[[0], [0]], [[-1], [1]], [[0], [0]]],
        [[[0], [0]], [[0], [0]], [[-1], [1]]]
    ]).type_as(img)
    return F.conv2d(img, kernel_dx, padding='same'), F.conv2d(img, kernel_dy, padding='same')


def image_gradient2(img):
    kernel_gx = torch.tensor([
        [[[-1 / 12, 2 / 3, 0, -2 / 3, 1 / 12]], [[0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0]]],
        [[[0, 0, 0, 0, 0]], [[-1 / 12, 2 / 3, 0, -2 / 3, 1 / 12]], [[0, 0, 0, 0, 0]]],
        [[[0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0]], [[-1 / 12, 2 / 3, 0, -2 / 3, 1 / 12]]]
    ]).type_as(img)
    kernel_gy = torch.tensor([
        [[[-1 / 12], [2 / 3], [0], [-2 / 3], [1 / 12]], [[0], [0], [0], [0], [0]], [[0], [0], [0], [0], [0]]],
        [[[0], [0], [0], [0], [0]], [[-1 / 12], [2 / 3], [0], [-2 / 3], [1 / 12]], [[0], [0], [0], [0], [0]]],
        [[[0], [0], [0], [0], [0]], [[0], [0], [0], [0], [0]], [[-1 / 12], [2 / 3], [0], [-2 / 3], [1 / 12]]]
    ]).type_as(img)
    return F.conv2d(img, kernel_gx, padding='same'), F.conv2d(img, kernel_gy, padding='same')


def border_mask(flow):
    """
    Generates a mask that is True for pixels whose correspondence is inside the image borders.
    flow: optical flow tensor (batch, 2, height, width)
    returns: mask (batch, height, width)
    """
    b, _, h, w = flow.size()
    x = torch.arange(w).type_as(flow)
    y = torch.arange(h).type_as(flow)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    Xp = X.view(1, h, w).repeat(b, 1, 1) + flow[:, 0, :, :]
    Yp = Y.view(1, h, w).repeat(b, 1, 1) + flow[:, 1, :, :]
    mask_x = (Xp > -0.5) & (Xp < w-0.5)
    mask_y = (Yp > -0.5) & (Yp < h-0.5)
    return mask_x & mask_y


def ternary_census_transform(image, patch_size):
    intensities = rgb_to_grayscale(image) * 255
    out_channels = patch_size * patch_size
    w = torch.eye(out_channels).view((out_channels, 1, patch_size, patch_size))
    weights = w.type_as(image)
    patches = F.conv2d(intensities, weights, padding='same')
    transf = patches - intensities
    transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))
    return transf_norm


def census_loss(img1, img2_warp, radius=3):
    patch_size = 2 * radius + 1

    def _hamming_distance(t1, t2):
        dist = torch.pow(t1 - t2, 2)
        dist_norm = dist / (0.1 + dist)
        dist_mean = torch.mean(dist_norm, 1, keepdim=True)  # instead of sum
        return dist_mean

    def _valid_mask(t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    t1 = ternary_census_transform(img1, patch_size)
    t2 = ternary_census_transform(img2_warp, patch_size)
    dist = _hamming_distance(t1, t2)
    valid_mask = _valid_mask(img1, radius)

    return dist, valid_mask


def color_loss(img1, img2_warp):
    return torch.sqrt(torch.mean((img1 - img2_warp)**2, dim=1))


def gradient_loss(img1, img2_warp):
    img1_gx, img1_gy = image_gradient2(img1)
    img2_warp_gx, img2_warp_gy = image_gradient2(img2_warp)
    return torch.sqrt(torch.mean((img1_gx - img2_warp_gx)**2 + (img1_gy - img2_warp_gy)**2, dim=1))


def smooth_grad_1st(flow, image, alpha):
    """
    flow: (batch, 2, height, width) tensor
    image: (batch, 3, height, width) tensor
    alpha: edge weighting parameter
    """
    # Calculate edge-weighting factor
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    # Calculate flow gradients
    flow_dx, flow_dy = gradient(flow)
    loss_x = weights_x * robust_l1(flow_dx) / 2.0
    loss_y = weights_y * robust_l1(flow_dy) / 2.0

    return loss_x.mean() + loss_y.mean()


def smooth_grad_2nd(flow, image, alpha):
    """
    flow: (batch, 2, height, width) tensor
    image: (batch, 3, height, width) tensor
    alpha: edge weighting parameter
    """
    img_dx, img_dy = gradient(image, stride=2)
    weights_xx = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_yy = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    flow_dx, flow_dy = gradient(flow)
    flow_dx2, _ = gradient(flow_dx)
    _, flow_dy2 = gradient(flow_dy)

    loss_x = weights_xx * robust_l1(flow_dx2) / 2.0
    loss_y = weights_yy * robust_l1(flow_dy2) / 2.0

    return loss_x.mean() + loss_y.mean()
