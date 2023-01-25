from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import torch.utils.data as data
from glob import glob
import torch
import torch.nn.functional as tf

from torchvision import transforms as vision_transforms

from . import transforms
from . import common


VALIDATE_INDICES = np.arange(5000, 5500)


class TinyFlyingChairs(data.Dataset):
    def __init__(self,
                 args,
                 root,
                 photometric_augmentations=False,
                 resize_targets=[-1,-1],
                 num_examples=-1,
                 dstype="train"):

        self._args = args
        self._resize_targets = resize_targets

        # -------------------------------------------------------------
        # filenames for all input images and target flows
        # -------------------------------------------------------------
        image_filenames = sorted( glob( os.path.join(root, "*.ppm")) )
        flow_filenames = sorted( glob( os.path.join(root, "*.flo")) )
        assert (len(image_filenames)/2 == len(flow_filenames))
        num_flows = len(flow_filenames)

        # -------------------------------------------------------------
        # Remove invalid validation indices
        # -------------------------------------------------------------
        validate_indices = [x for x in VALIDATE_INDICES if x in range(num_flows)]

        # ----------------------------------------------------------
        # Construct list of indices for training/validation
        # ----------------------------------------------------------
        list_of_indices = None
        if dstype == "train":
            list_of_indices = [x for x in range(num_flows) if x not in validate_indices]
        elif dstype == "valid":
            list_of_indices = validate_indices
        elif dstype == "full":
            list_of_indices = range(num_flows)
        else:
            raise ValueError("FlyingChairs: dstype '%s' unknown!", dstype)

        # ----------------------------------------------------------
        # Restrict dataset indices if num_examples is given
        # ----------------------------------------------------------
        if num_examples > 0:
            restricted_indices = common.deterministic_indices(
                seed=0, k=num_examples, n=len(list_of_indices))
            list_of_indices = [list_of_indices[i] for i in restricted_indices]

        # ----------------------------------------------------------
        # Save list of actual filenames for inputs and flows
        # ----------------------------------------------------------
        self._image_list = []
        self._flow_list = []
        for i in list_of_indices:
            flo = flow_filenames[i]
            im1 = image_filenames[2*i]
            im2 = image_filenames[2*i + 1]
            self._image_list += [ [ im1, im2 ] ]
            self._flow_list += [ flo ]
        self._size = len(self._image_list)
        assert len(self._image_list) == len(self._flow_list)

        # ----------------------------------------------------------
        # photometric_augmentations
        # ----------------------------------------------------------
        if photometric_augmentations:
            self._photometric_transform = transforms.ConcatTransformSplitChainer([
                # uint8 -> PIL
                vision_transforms.ToPILImage(),
                # PIL -> PIL : random hsv and contrast
                vision_transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                # PIL -> FloatTensor
                vision_transforms.transforms.ToTensor(),
                transforms.RandomGamma(min_gamma=0.7, max_gamma=1.5, clip_image=True),
            ], from_numpy=True, to_numpy=False)

        else:
            self._photometric_transform = transforms.ConcatTransformSplitChainer([
                # uint8 -> FloatTensor
                vision_transforms.transforms.ToTensor(),
            ], from_numpy=True, to_numpy=False)

    def __getitem__(self, index):
        index = index % self._size

        im1_filename = self._image_list[index][0]
        im2_filename = self._image_list[index][1]
        flo_filename = self._flow_list[index]

        # read float32 images and flow
        im1_np0 = common.read_image_as_byte(im1_filename)
        im2_np0 = common.read_image_as_byte(im2_filename)
        flo_np0 = common.read_flo_as_float32(flo_filename)

        # possibly apply photometric transformations
        im1, im2 = self._photometric_transform(im1_np0, im2_np0)

        # convert flow to FloatTensor
        flo = common.numpy2torch(flo_np0)

        #import numpy as np
        #from matplotlib import pyplot as plt
        #import numpy as np
        #plt.figure()
        #im1_np = im1.numpy().transpose([1,2,0])
        #im2_np = im2.numpy().transpose([1,2,0])
        ##plt.imshow(np.concatenate((im1_np0.astype(np.float32)/255.0, im2_np0.astype(np.float32)/255.0, im1_np, im2_np), 1))
        #plt.imshow(np.concatenate((im1_np, im2_np), 1))
        #plt.show(block=True)

        # example filename
        basename = os.path.basename(im1_filename)[:5]

        example_dict = {
            "input1": im1,
            "input2": im2,
            "target1": flo,
            "index": index,
            "basename": basename
        }

        return example_dict

    def __len__(self):
        return self._size


class TinyFlyingChairsTrain(TinyFlyingChairs):
    def __init__(self,
                 args,
                 root,
                 photometric_augmentations=True,
                 num_examples=-1):
        super(TinyFlyingChairsTrain, self).__init__(
            args,
            root=root,
            photometric_augmentations=photometric_augmentations,
            dstype="train",
            num_examples=num_examples)


class TinyFlyingChairsValid(TinyFlyingChairs):
    def __init__(self,
                 args,
                 root,
                 photometric_augmentations=False,
                 num_examples=-1):
        super(TinyFlyingChairsValid, self).__init__(
            args,
            root=root,
            photometric_augmentations=photometric_augmentations,
            dstype="valid",
            num_examples=num_examples)


class TinyFlyingChairsFull(TinyFlyingChairs):
    def __init__(self,
                 args,
                 root,
                 photometric_augmentations=False,
                 num_examples=-1):
        super(TinyFlyingChairsValid, self).__init__(
            args,
            root=root,
            photometric_augmentations=photometric_augmentations,
            dstype="full",
            num_examples=num_examples)


