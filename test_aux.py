from losses.resample2d_package.resample2d import Resample2d
from losses.aux import border_mask, ternary_census_transform, smooth_grad_1st, smooth_grad_2nd
from datasets.tinyflyingchairs import TinyFlyingChairsTrain
import matplotlib.pyplot as plt
import torch


# Load a data sample
train_data = TinyFlyingChairsTrain(None, '/home/deu/FlyingChairs_release/downscaled_data',
                                   photometric_augmentations=False)

image = train_data.__getitem__(10)['input1']
flow = train_data.__getitem__(10)['target1']
c, h, w = image.size()
image = image.cuda().view(1,c,h,w)
flow = flow.cuda().view(1,2,h,w).contiguous()

# Test census transform
cens = ternary_census_transform(image, 7)
cens = cens.cpu().squeeze().numpy().transpose(1,2,0)
plt.imshow(cens.mean(axis=2))
plt.show()

# Test 1st order smoothness
dx, dy, wx, wy = smooth_grad_1st(flow, image, 4.0)
dx = dx.cpu().squeeze().numpy()
dy = dy.cpu().squeeze().numpy()
wx = wx.cpu().squeeze().numpy()
wy = wy.cpu().squeeze().numpy()
fig, ax = plt.subplots(3,2)
ax[0,0].imshow(dx[0])
ax[0,1].imshow(dx[1])
ax[1,0].imshow(dy[0])
ax[1,1].imshow(dy[1])
ax[2,0].imshow(wx)
ax[2,1].imshow(wy)
plt.show()

# Test warping
resample2d = Resample2d()
warped_image = resample2d(image, flow)
warped_image = warped_image.cpu().squeeze()
warped_image = warped_image.numpy().transpose(1,2,0)
plt.imshow(warped_image)
plt.show()

# Test border-mask
mask = border_mask(flow)
mask = mask.cpu().squeeze().numpy()
plt.imshow(mask, cmap='gray')
plt.show()



