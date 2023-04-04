#%% This class is dataloader for the MRI data

import os
import random

import numpy as np
import numpy.ma as ma
import scipy

from scipy import ndimage
from skimage import measure


# The mean and variance normalization
def whitening_transformation(image, mask):
    # make sure image is a monomodal volume
    masked_img = ma.masked_array(image, np.logical_not(mask))
    image = (image - masked_img.mean()) / max(masked_img.std(), 1e-5)
    return image

def randomErase3d(image_list, EPSILON = 0.5, sl = 0.002, sh = 0.2, r1 = 0.3):

    if random.uniform(0, 1) > EPSILON:
        return image_list

    if len(image_list) <= 2:
        size0 = image_list[0].shape[0]
        size1 = image_list[0].shape[1]
        size2 = image_list[0].shape[2]
    else:

        size0 = min(image_list[0].shape[0], image_list[-1].shape[0])
        size1 = min(image_list[0].shape[1], image_list[-1].shape[1])
        size2 = min(image_list[0].shape[2], image_list[-1].shape[2])

    volume = size0 * size1 * size2

    for attempt in range(100):

        target_volume = random.uniform(sl, sh) * volume
        aspect_ratio1 = random.uniform(r1, 1 / r1)
        aspect_ratio2 = random.uniform(r1, 1 / r1)

        h = int((target_volume * aspect_ratio1 * aspect_ratio2) ** (1./3) )
        w = int((target_volume * aspect_ratio1 / (aspect_ratio2 * aspect_ratio2) ) ** (1./3) )
        l = int((target_volume * aspect_ratio2 / (aspect_ratio1 * aspect_ratio1) ) ** (1./3) )

        if h < size0 and w < size1 and l < size2:
            x1 = random.randint(0, size0 - h)
            y1 = random.randint(0, size1 - w)
            z1 = random.randint(0, size2 - l)

            for i in range(len(image_list)):
                image_list[i][x1:x1 + h, y1:y1 + w, z1:z1 + l] = np.random.randn(h, w, l) # Gaussian Distribution
                # image_list[0, x1:x1 + h, y1:y1 + w] = self.mean[1]
                # image_list[0, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(1, h, w))

            return image_list

    return image_list


def randomFlip3d(image_list):
    flip = random.randint(0, 5)  # generate a <= N <= b

    for i in range(len(image_list)):
        if flip == 0:
            image_list[i] = image_list[i][:, :, ::-1]
        elif flip == 1:
            image_list[i] = image_list[i][:, ::-1, :]
        elif flip == 2:
            image_list[i] = image_list[i][::-1, :, :]

    return image_list


def fixedFlip3d(image_list):
    # what dimension is flip left-right?

    for i in range(len(image_list)):
        image_list[i] = image_list[i][:, :, ::-1] # [:, ::-1, :] or [::-1, :, :]

    return image_list


def Normalize(image_list):

    for i in range(len(image_list)):
        image_list[i] = (image_list[i] - image_list[i].mean()) / max(image_list[i].std(), 1e-5)

    return image_list


def fixedCrop3d(image_list, output_size):
    # 3D crop ensuring most area of the mask is inside the cropped image.
    # if dimension of mask is less than required dimension, then random crop outside the mask;
    # if greater, then random crop inside the mask.

    # the last image is the mask

    # print(image_list[0].shape)
    # image_list[2] = ndimage.grey_dilation(image_list[2], size=(5, 5, 5))  # (dilated mask)
    # image_list[2] = ndimage.gaussian_filter(image_list[2], sigma=2)

    a = b = c = []
    if len(a) == 0:  # all labels are zeros
        mask_f = 0
        mask_b = image_list[0].shape[0]  # the same size as the input image
        mask_u = 0
        mask_d = image_list[0].shape[1]
        mask_l = 0
        mask_r = image_list[0].shape[2]

    else:
        mask_f = min(a)  # front
        mask_b = max(a)  # back
        mask_u = min(b)  # up
        mask_d = max(b)  # down
        mask_l = min(c)  # left
        mask_r = max(c)  # right

    mask_t = mask_b - mask_f
    mask_h = mask_d - mask_u
    mask_w = mask_r - mask_l

    t, h, w = image_list[0].shape

    # thickness, height, width: desired output size
    # [48, 80, 64]
    thickness, height, width = output_size

    if t < thickness:
        thick_diff = int(np.floor((thickness - t) / 2))
        for i in range(len(image_list)):
            temp = np.zeros([thickness, h, w])
            temp[thick_diff: thick_diff + t, :, :] = image_list[i]
            image_list[i] = temp
        t = thickness
    if h < height:
        height_diff = int(np.floor((height - h) / 2))
        for i in range(len(image_list)):
            temp = np.zeros([t, height, w])
            temp[:, height_diff: height_diff + h, :] = image_list[i]
            image_list[i] = temp
        h = height
    if w < width:
        width_diff = int(np.floor((width - w) / 2))
        for i in range(len(image_list)):
            temp = np.zeros([t, h, width])
            temp[:, :, width_diff: width_diff + w] = image_list[i]
            image_list[i] = temp
        w = width

    # output_image_list = []
    im1_new1 = [None] * len(image_list)
    im1_new2 = [None] * len(image_list)
    output_image_list = [None] * len(image_list)

    if mask_h > height:
        # top = np.random.randint(mask_u, mask_d - height)
        top = int((mask_u + mask_d - height) / 2)
        for i in range(len(image_list)):
            im1_new1[i] = image_list[i][:, top: top + height, :]
    elif mask_h <= height:
        if h < height:
            # pad_h = np.random.randint(0, height - h)
            pad_h = int((height - h)/2)
            for i in range(len(image_list)):
                im1_new1[i] = np.zeros([t, height, w])
                im1_new1[i][:, pad_h: pad_h + h, :] = image_list[i]
        elif h > height:
            # top = np.random.randint(0, min([mask_u, h - height]) + 1)
            top = int(min([mask_u, h - height])/2 + 1)
            for i in range(len(image_list)):
                im1_new1[i] = image_list[i][:, top: top + height, :]
        else:
            for i in range(len(image_list)):
                im1_new1[i] = image_list[i]

    if mask_w > width:
        # left = np.random.randint(mask_l, mask_r - width)
        left = int((mask_l + mask_r - width)/2)
        for i in range(len(image_list)):
            im1_new2[i] = im1_new1[i][:, :, left: left + width]
    elif mask_w <= width:
        if w < width:
            # pad_w = np.random.randint(0, width - w)
            pad_w = int((width - w)/2)
            for i in range(len(image_list)):
                im1_new2[i] = np.zeros([t, height, width])
                im1_new2[i][:, :, pad_w: pad_w + w] = im1_new1[i]
        elif w > width:
            # left = np.random.randint(0, min([mask_l, w - width]) + 1)
            left = int(min([mask_l, w - width])/2 + 1)
            for i in range(len(image_list)):
                im1_new2[i] = im1_new1[i][:, :, left: left + width]
        else:
            for i in range(len(image_list)):
                im1_new2[i] = im1_new1[i]

    if mask_t > thickness:
        # front = np.random.randint(mask_f, mask_b - thickness)
        front = int((mask_f + mask_b - thickness)/2)
        for i in range(len(image_list)):
            output_image_list[i] = im1_new2[i][front: front + thickness, :, :]
    elif mask_t <= thickness:
        if t < thickness:
            # pad_t = np.random.randint(0, thickness - t)
            pad_t = int((thickness - t)/2)
            for i in range(len(image_list)):
                output_image_list[i] = np.zeros([thickness, height, width])
                output_image_list[i][pad_t: pad_t + t, :, :] = im1_new2[i]
        elif t > thickness:
            # front = np.random.randint(0, min([mask_f, t - thickness]) + 1)
            front = int(min([mask_f, t - thickness])/2 + 1)
            for i in range(len(image_list)):
                output_image_list[i] = im1_new2[i][front: front + thickness, :, :]
        else:
            for i in range(len(image_list)):
                output_image_list[i] = im1_new2[i]

    for i in range(len(image_list)):
        if list(output_image_list[i].shape) != output_size:
            print("output_image_list[i].shape")
        # assert output_image_list[i].shape == output_size
    return output_image_list

def randomCrop3d(image_list, output_size):
    # 3D crop ensuring most area of the mask is inside the cropped image.
    # if dimension of mask is less than required dimension, then random crop outside the mask;
    # if greater, then random crop inside the mask.

    # the last image is the mask

    # print(image_list[0].shape)
    # image_list[2] = ndimage.grey_dilation(image_list[2], size=(5, 5, 5))  # (dilated mask)
    # image_list[2] = ndimage.gaussian_filter(image_list[2], sigma=2)

    a = b = c = []
    if len(a) == 0:  # all labels are zeros
        mask_f = 0
        mask_b = image_list[0].shape[0]  # the same size as the input image
        mask_u = 0
        mask_d = image_list[0].shape[1]
        mask_l = 0
        mask_r = image_list[0].shape[2]

    else:
        mask_f = min(a)  # front
        mask_b = max(a)  # back
        mask_u = min(b)  # up
        mask_d = max(b)  # down
        mask_l = min(c)  # left
        mask_r = max(c)  # right

    mask_t = mask_b - mask_f
    mask_h = mask_d - mask_u
    mask_w = mask_r - mask_l

    t, h, w = image_list[0].shape

    # thickness, height, width: desired output size
    # [48, 80, 64]
    thickness, height, width = output_size
    crop_pos_t, crop_pos_h, crop_pos_w = 0, 0, 0

    if t < thickness:
        thick_diff = int(np.floor((thickness - t) / 2))
        for i in range(len(image_list)):
            temp = np.zeros([thickness, h, w])
            temp[thick_diff: thick_diff + t, :, :] = image_list[i]
            image_list[i] = temp
        t = thickness
        crop_pos_t = -thick_diff  #####
    if h < height:
        height_diff = int(np.floor((height - h) / 2))
        for i in range(len(image_list)):
            temp = np.zeros([t, height, w])
            temp[:, height_diff: height_diff + h, :] = image_list[i]
            image_list[i] = temp
        h = height
        crop_pos_h = -height_diff  #####
    if w < width:
        width_diff = int(np.floor((width - w) / 2))
        for i in range(len(image_list)):
            temp = np.zeros([t, h, width])
            temp[:, :, width_diff: width_diff + w] = image_list[i]
            image_list[i] = temp
        w = width
        crop_pos_w = -width_diff  #####

    if mask_t > thickness:
        front = np.random.randint(mask_f, mask_b - thickness)
        crop_pos_t = front  #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][front: front + thickness, :, :]
    elif mask_t <= thickness and t > thickness:
        front = np.random.randint(0, min([mask_f, t - thickness]) + 1)
        crop_pos_t = front  #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][front: front + thickness, :, :]

    if mask_h > height:
        top = np.random.randint(mask_u, mask_d - height)
        crop_pos_h = top  #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, top: top + height, :]
    elif mask_h <= height and h > height:
        top = np.random.randint(0, min([mask_u, h - height]) + 1)
        crop_pos_h = top  #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, top: top + height, :]

    if mask_w > width:
        left = np.random.randint(mask_l, mask_r - width)
        crop_pos_w = left  #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, :, left: left + width]
    elif mask_w <= width and w > width:
        left = np.random.randint(0, min([mask_l, w - width]) + 1)
        crop_pos_w = left  #####
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, :, left: left + width]

    for i in range(len(image_list)):
        if tuple(image_list[i].shape) != output_size:
            print("output_image_list[i].shape")
        # assert output_image_list[i].shape == output_size

    return image_list, [crop_pos_t, crop_pos_h, crop_pos_w]


def randomCropBySeg3d(image_list, output_size):
    # 3D crop ensuring most area of the mask is inside the cropped image.
    # if dimension of mask is less than required dimension, then random crop outside the mask;
    # if greater, then random crop inside the mask.

    # the last image is the mask
    mask = image_list[2]

    a, b, c = np.nonzero(mask)
    if len(a) == 0:  # all labels are zeros
        mask_f = 0
        mask_b = mask.shape[0]  # the same size as the input image
        mask_u = 0
        mask_d = mask.shape[1]
        mask_l = 0
        mask_r = mask.shape[2]

    else:
        mask_f = min(a)  # front
        mask_b = max(a)  # back
        mask_u = min(b)  # up
        mask_d = max(b)  # down
        mask_l = min(c)  # left
        mask_r = max(c)  # right

    mask_t = mask_b - mask_f
    mask_h = mask_d - mask_u
    mask_w = mask_r - mask_l

    t, h, w = image_list[0].shape

    # thickness, height, width: desired output size
    # [48, 80, 64]
    thickness, height, width = output_size
    crop_pos_t, crop_pos_h, crop_pos_w  = 0, 0, 0

    if t < thickness:
        thick_diff = int(np.floor((thickness - t) / 2))
        for i in range(len(image_list)):
            temp = np.zeros([thickness, h, w])
            temp[thick_diff: thick_diff + t, :, :] = image_list[i]
            image_list[i] = temp
        t = thickness
        crop_pos_t = -thick_diff 
    if h < height:
        height_diff = int(np.floor((height - h) / 2))
        for i in range(len(image_list)):
            temp = np.zeros([t, height, w])
            temp[:, height_diff: height_diff + h, :] = image_list[i]
            image_list[i] = temp
        h = height
        crop_pos_h = -height_diff
    if w < width:
        width_diff = int(np.floor((width - w) / 2))
        for i in range(len(image_list)):
            temp = np.zeros([t, h, width])
            temp[:, :, width_diff: width_diff + w] = image_list[i]
            image_list[i] = temp
        w = width
        crop_pos_w = -width_diff 

    if mask_t > thickness:
        front = np.random.randint(mask_f, mask_b - thickness)
        crop_pos_t = front
        for i in range(len(image_list)):
            image_list[i] = image_list[i][front: front + thickness, :, :]
    elif mask_t <= thickness and t > thickness:
        front = np.random.randint(0, min([mask_f, t - thickness]) + 1)
        crop_pos_t = front  
        for i in range(len(image_list)):
            image_list[i] = image_list[i][front: front + thickness, :, :]

    if mask_h > height:
        top = np.random.randint(mask_u, mask_d - height)
        crop_pos_h = top 
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, top: top + height, :]
    elif mask_h <= height and h > height:
        top = np.random.randint(0, min([mask_u, h - height]) + 1)
        crop_pos_h = top 
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, top: top + height, :]

    if mask_w > width:
        left = np.random.randint(mask_l, mask_r - width)
        crop_pos_w = left 
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, :, left: left + width]
    elif mask_w <= width and w > width:
        left = np.random.randint(0, min([mask_l, w - width]) + 1)
        crop_pos_w = left 
        for i in range(len(image_list)):
            image_list[i] = image_list[i][:, :, left: left + width]

    for i in range(len(image_list)):
        if tuple(image_list[i].shape) != output_size:
            print("output_image_list[i].shape")
        # assert output_image_list[i].shape == output_size

    return image_list, [crop_pos_t, crop_pos_h, crop_pos_w]


# Function fetched and adapted from this thread
# https://stackoverflow.com/questions/43922198/how-to-rotate-a-3d-image-by-a-random-angle-in-python
def randomRotation3d(image_list, max_angle, prob, segs=True):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).

    Arguments:
    max_angle: `float`. The maximum rotation angle.

    Returns:
    A randomly rotated 3D image and the mask
    """
    # Consider this function being used in multithreading in pytorch's dataloader,
    # if one don't reseed each time this thing is run, the couple worker in pytorch's
    # data worker will produce exactly the same random number and that's no good.


    rotate = np.random.random() > prob
    if rotate > 0:
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        # image_raw = image1.copy()
        # rotate along z-axis
        angle = np.random.uniform(-max_angle, max_angle)
        if segs:
            for i in range(len(image_list)):
                if i % 3 == 2: # for mask
                    image_list[i] = scipy.ndimage.interpolation.rotate(image_list[i],
                                                                       angle,
                                                                       mode='constant',
                                                                       axes=(0, 1),
                                                                       reshape=False,
                                                                       order=0)
                else:
                    image_list[i] = scipy.ndimage.interpolation.rotate(image_list[i],
                                                                       angle,
                                                                       mode='constant',
                                                                       axes=(0, 1),
                                                                       reshape=False,
                                                                       order = 3)
        else:
            for i in range(len(image_list)):
                image_list[i] = scipy.ndimage.interpolation.rotate(image_list[i],
                                                                   angle,
                                                                   mode='constant',
                                                                   axes=(0, 1),
                                                                   reshape=False,
                                                                   order=3)

    rotate = np.random.random() > prob
    if rotate > 0:
        # rotate along y-axis
        angle = np.random.uniform(-max_angle, max_angle)
        if segs:
            for i in range(len(image_list)):
                if i % 3 == 2: # for mask
                    image_list[i] = scipy.ndimage.interpolation.rotate(image_list[i],
                                                                       angle,
                                                                       mode='constant',
                                                                       axes=(0, 2),
                                                                       reshape=False,
                                                                       order=0)
                else:
                    image_list[i] = scipy.ndimage.interpolation.rotate(image_list[i],
                                                                       angle,
                                                                       mode='constant',
                                                                       axes=(0, 2),
                                                                       reshape=False,
                                                                       order = 3)
        else:
            for i in range(len(image_list)):
                image_list[i] = scipy.ndimage.interpolation.rotate(image_list[i],
                                                                   angle,
                                                                   mode='constant',
                                                                   axes=(0, 2),
                                                                   reshape=False,
                                                                   order=3)

    rotate = np.random.random() > prob
    if rotate > 0:
        # rotate along x-axis
        angle = np.random.uniform(-max_angle, max_angle)
        if segs:
            for i in range(len(image_list)):
                if i % 3 == 2: # for mask
                    image_list[i] = scipy.ndimage.interpolation.rotate(image_list[i],
                                                                       angle,
                                                                       mode='constant',
                                                                       axes=(1, 2),
                                                                       reshape=False,
                                                                       order=0)
                else:
                    image_list[i] = scipy.ndimage.interpolation.rotate(image_list[i],
                                                                       angle,
                                                                       mode='constant',
                                                                       axes=(1, 2),
                                                                       reshape=False,
                                                                       order = 3)
        else:
            for i in range(len(image_list)):
                image_list[i] = scipy.ndimage.interpolation.rotate(image_list[i],
                                                                   angle,
                                                                   mode='constant',
                                                                   axes=(1, 2),
                                                                   reshape=False,
                                                                   order=3)

    return image_list


def alignCenterCrop3d(image_list2, crop_pos1, output_size, segs=True, vol_seg1=None):

    """ align two image pairs by centroid of two segmentations
    crop the second pair of images at the same center as image pair 1
    return cropped second images, and crop_pos2

    Arguments:
    vol_seg1: segmentation of the first image pair before cropping
    crop_pos1: crop position (upper left corner) of the first image pair
    image_list2: vol_bl2, vol_fu2, and vol_seg2 of the second image pair

    Returns:
    cropped second images, and crop_pos2
    """

    if segs:
        vol_seg1 = (vol_seg1 > 0).astype(int)
        properties1 = measure.regionprops(vol_seg1)
        center1 = [int(x) for x in properties1[0].centroid]

        vol_seg2 = (image_list2[-1] > 0).astype(int)
        properties2 = measure.regionprops(vol_seg2)
        center2 = [int(x) for x in properties2[0].centroid]

        crop_pos2 = [crop_pos1[i] - center1[i] + center2[i] for i in range(len(center1))]

    else:
        crop_pos2 = crop_pos1

    orig_shape = vol_seg2.shape

    big_shape = np.max([output_size, orig_shape], axis=0)
    # a big shape to maintain all data, and then crop
    for i in range(len(image_list2)):

        arr1 = np.zeros(big_shape)
        start10 = max(crop_pos2[0], 0)
        start11 = max(crop_pos2[1], 0)
        start12 = max(crop_pos2[2], 0)
        arr1[:orig_shape[0] - start10,
             :orig_shape[1] - start11,
             :orig_shape[2] - start12,] = image_list2[i][start10:,
                                                         start11:,
                                                         start12:]

        # This part might be wrong because there's no case crop_pos is positive.
        # So no testing data.
        start20 = -min(crop_pos2[0], 0)
        start21 = -min(crop_pos2[1], 0)
        start22 = -min(crop_pos2[2], 0)
        arr2 = arr1[start20:start20 + output_size[0],
                    start21:start21 + output_size[1],
                    start22:start22 + output_size[2]]

        if arr2.shape != tuple(output_size):
            print("didn't map to output size")

        image_list2[i] = arr2


    return image_list2, crop_pos2
