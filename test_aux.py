from losses.aux import border_mask, ternary_census_transform, smooth_grad_1st, smooth_grad_2nd, flow_warp, census_loss, color_loss, warp
from datasets.tinyflyingchairs import TinyFlyingChairsTrain
import matplotlib.pyplot as plt
import torch


# Load a data sample
train_data = TinyFlyingChairsTrain(None, '/home/deu/FlyingChairs_release/downscaled_data',
                                   photometric_augmentations=False)

img1 = train_data.__getitem__(400)['input1']
img2 = train_data.__getitem__(400)['input2']
flow = train_data.__getitem__(400)['target1']
c, h, w = img1.size()
img1 = img1.cuda().view(1,c,h,w)
img2 = img2.cuda().view(1,c,h,w)
flow = flow.cuda().view(1,2,h,w).contiguous()

# Test border-mask
mask = border_mask(flow)
mask = mask.cpu().squeeze().numpy()
plt.imshow(mask, cmap='gray')
plt.show()

# Test census transform
cens = ternary_census_transform(img1, 7)
cens = cens.cpu().squeeze().numpy().transpose(1,2,0)
plt.imshow(cens.mean(axis=2))
plt.show()

# Test warping
img2_warp = warp(img2, flow)
#img2_warp = flow_warp(img2, flow)
mask = border_mask(flow)
print(color_loss(img1, img2, mask).item())
print(color_loss(img1, img2_warp, mask).item())
img2_warp = img2_warp.cpu().squeeze().numpy().transpose(1,2,0)
img1 = img1.cpu().squeeze().numpy().transpose(1,2,0)
img2 = img2.cpu().squeeze().numpy().transpose(1,2,0)
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(img1)
ax[0,1].imshow(img2)
ax[1,0].imshow(img1)
ax[1,1].imshow(img2_warp)
plt.show()




