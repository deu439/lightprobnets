from __future__ import absolute_import
from __future__ import print_function

import torch
from torchvision.transforms.functional import rgb_to_grayscale
import torch.nn.functional as F
import inspect


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


# Credit: https://github.com/lliuz/ARFlow/blob/e92a8bbe66f0ced244267f43e3e55ad0fe46ff3e/utils/warp_utils.py#L83
def flow_warp(x, flow12, pad='border', mode='bilinear'):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    if 'align_corners' in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = F.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    else:
        im1_recons = F.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons


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
    mask_x = (Xp > 0.0) & (Xp < w-1.0)
    mask_y = (Yp > 0.0) & (Yp < h-1.0)
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


def census_loss(img1, img2_warp, mask, radius=3):
    """
    img1: first image (batch, 3, height, width) tensor
    img2_warp: warped second image (batch, 3, height, width) tensor
    mask: binary occlusion mask (batch, height, width) tensor
    radius: radius of the neighborhood used in census transform
    """
    patch_size = 2 * radius + 1

    def _hamming_distance(t1, t2):
        dist = torch.pow(t1 - t2, 2)
        dist_norm = dist / (0.1 + dist)
        dist_mean = torch.sum(dist_norm, 1, keepdim=True)
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

    return torch.sum(abs_robust_loss(dist) * mask * valid_mask) / (torch.sum(mask * valid_mask) + 1e-6)


def color_loss(img1, img2_warp, mask):
    """
    img1: first image (batch, 3, height, width) tensor
    img2_warp: warped second image (batch, 3, height, width) tensor
    mask: binary occlusion mask (batch, height, width) tensor
    """
    per_pixel = torch.sqrt(torch.mean((img1 - img2_warp)**2, dim=1))    # Average over channels
    return torch.mean(robust_l1(per_pixel) * mask)  # Average over pixels and batch


def gradient_loss(img1, img2_warp, mask):
    """
    img1: first image (batch, 3, height, width) tensor
    img2_warp: warped second image (batch, 3, height, width) tensor
    mask: binary occlusion mask (batch, height, width) tensor
    """
    img1_gx, img1_gy = image_gradient2(img1)
    img2_warp_gx, img2_warp_gy = image_gradient2(img2_warp)
    per_pixel = torch.sqrt(torch.mean((img1_gx - img2_warp_gx)**2 + (img1_gy - img2_warp_gy)**2, dim=1))    # Average over channels
    return torch.mean(robust_l1(per_pixel) * mask)  # Average over pixels and batch


def smooth_grad_1st(flow, image, edge_weight=4.0):
    """
    flow: optical flow (batch, 2, height, width) tensor
    image: image used to calculate edge-based weights (batch, 3, height, width) tensor
    alpha: edge weighting parameter
    """
    # Calculate edge-weighting factor
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * edge_weight)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * edge_weight)

    # Calculate flow gradients
    flow_dx, flow_dy = gradient(flow)
    loss_x = weights_x * robust_l1(flow_dx) / 2.0
    loss_y = weights_y * robust_l1(flow_dy) / 2.0

    return loss_x.mean() + loss_y.mean()    # Average over pixels, channels and batch


def smooth_grad_2nd(flow, image, edge_weight=4.0):
    """
    flow: optical flow (batch, 2, height, width) tensor
    image: image used to calculate edge-based weights (batch, 3, height, width) tensor
    alpha: edge weighting parameter
    """
    img_dx, img_dy = gradient(image, stride=2)
    weights_xx = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * edge_weight)
    weights_yy = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * edge_weight)

    flow_dx, flow_dy = gradient(flow)
    flow_dx2, _ = gradient(flow_dx)
    _, flow_dy2 = gradient(flow_dy)

    loss_x = weights_xx * robust_l1(flow_dx2) / 2.0
    loss_y = weights_yy * robust_l1(flow_dy2) / 2.0

    return loss_x.mean() + loss_y.mean()    # Average over pixels channels and batch


# Credit: https://github.com/lliuz/ARFlow/blob/e92a8bbe66f0ced244267f43e3e55ad0fe46ff3e/utils/warp_utils.py#L26
def get_corresponding_map(data):
    """
    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()

    # x = data[:, 0, :, :].view(B, -1).clamp(0, W - 1)  # BxN (N=H*W)
    # y = data[:, 1, :, :].view(B, -1).clamp(0, H - 1)

    x = data[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(B, -1)

    # invalid = (x < 0) | (x > W - 1) | (y < 0) | (y > H - 1)   # BxN
    # invalid = invalid.repeat([1, 4])

    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)

    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out,
                         x_ceil_out | y_floor_out,
                         x_floor_out | y_ceil_out,
                         x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W,
                         x_ceil + y_floor * W,
                         x_floor + y_ceil * W,
                         x_floor + y_floor * W], 1).long()  # BxN   (N=4*H*W)
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))],
                       1)
    # values = torch.ones_like(values)

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(B, H, W)

    return corresponding_map.unsqueeze(1)


# Credit: https://github.com/lliuz/ARFlow/blob/e92a8bbe66f0ced244267f43e3e55ad0fe46ff3e/utils/warp_utils.py#L106
def get_occu_mask_backward(flow21, th=0.2):
    B, _, H, W = flow21.size()
    base_grid = mesh_grid(B, H, W).type_as(flow21)  # B2HW

    corr_map = get_corresponding_map(base_grid + flow21)  # BHW
    occu_mask = corr_map.clamp(min=0., max=1.) < th
    return occu_mask.float()