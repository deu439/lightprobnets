from losses.resample2d_package.resample2d import Resample2d
from losses.aux import border_mask, ternary_census_transform, smooth_grad_1st, smooth_grad_2nd, warp, flow_warp
from datasets.tinyflyingchairs import TinyFlyingChairsTrain
import matplotlib.pyplot as plt
import torch


# Load a data sample
train_data = TinyFlyingChairsTrain(None, '/home/deu/FlyingChairs_release/downscaled_data',
                                   photometric_augmentations=False)

img1 = train_data.__getitem__(100)['input1']
img2 = train_data.__getitem__(100)['input2']
flow = train_data.__getitem__(100)['target1']
c, h, w = img1.size()
img1 = img1.cuda().view(1,c,h,w)
img2 = img2.cuda().view(1,c,h,w)
flow = flow.cuda().view(1,2,h,w).contiguous()

# Test census transform
cens = ternary_census_transform(img1, 7)
cens = cens.cpu().squeeze().numpy().transpose(1,2,0)
plt.imshow(cens.mean(axis=2))
plt.show()

# Test warping
#resample2d = Resample2d()
#img2_warp = resample2d(img1, flow)
img2_warp = warp(img2, flow)
#img2_warp = flow_warp(img2, flow)
img2_warp = img2_warp.cpu().squeeze().numpy().transpose(1,2,0)
img1 = img1.cpu().squeeze().numpy().transpose(1,2,0)
img2 = img2.cpu().squeeze().numpy().transpose(1,2,0)
fig, ax = plt.subplots(2)
ax[0].imshow(img1 - img2)
ax[1].imshow(img1 - img2_warp)
plt.show()

# Test border-mask
mask = border_mask(flow)
mask = mask.cpu().squeeze().numpy()
plt.imshow(mask, cmap='gray')
plt.show()



