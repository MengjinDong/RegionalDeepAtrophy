#%% This class is dataloader for the MRI data

import os
import random

import numpy as np
import numpy.ma as ma
import scipy

from scipy import ndimage


# The mean and variance normalization
def whitening_transformation(image, mask):
    # make sure image is a monomodal volume
    masked_img = ma.masked_array(image, np.logical_not(mask))
    image = (image - masked_img.mean()) / max(masked_img.std(), 1e-5)
    return image


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
        top = np.random.randint(mask_u, mask_d - height)
        for i in range(len(image_list)):
            im1_new1[i] = image_list[i][:, top: top + height, :]
    elif mask_h <= height:
        if h < height:
            pad_h = np.random.randint(0, height - h)
            for i in range(len(image_list)):
                im1_new1[i] = np.zeros([t, height, w])
                im1_new1[i][:, pad_h: pad_h + h, :] = image_list[i]
        elif h > height:
            top = np.random.randint(0, min([mask_u, h - height]) + 1)
            for i in range(len(image_list)):
                im1_new1[i] = image_list[i][:, top: top + height, :]
        else:
            for i in range(len(image_list)):
                im1_new1[i] = image_list[i]

    if mask_w > width:
        left = np.random.randint(mask_l, mask_r - width)
        for i in range(len(image_list)):
            im1_new2[i] = im1_new1[i][:, :, left: left + width]
    elif mask_w <= width:
        if w < width:
            pad_w = np.random.randint(0, width - w)
            for i in range(len(image_list)):
                im1_new2[i] = np.zeros([t, height, width])
                im1_new2[i][:, :, pad_w: pad_w + w] = im1_new1[i]
        elif w > width:
            left = np.random.randint(0, min([mask_l, w - width]) + 1)
            for i in range(len(image_list)):
                im1_new2[i] = im1_new1[i][:, :, left: left + width]
        else:
            for i in range(len(image_list)):
                im1_new2[i] = im1_new1[i]

    if mask_t > thickness:
        front = np.random.randint(mask_f, mask_b - thickness)
        for i in range(len(image_list)):
            output_image_list[i] = im1_new2[i][front: front + thickness, :, :]
    elif mask_t <= thickness:
        if t < thickness:
            pad_t = np.random.randint(0, thickness - t)
            for i in range(len(image_list)):
                output_image_list[i] = np.zeros([thickness, height, width])
                output_image_list[i][pad_t: pad_t + t, :, :] = im1_new2[i]
        elif t > thickness:
            front = np.random.randint(0, min([mask_f, t - thickness]) + 1)
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

def randomCropBySeg3d(image_list, output_size):
    
    # 3D crop ensuring most area of the mask is inside the cropped image.
    # if dimension of mask is less than required dimension, then random crop outside the mask;
    # if greater, then random crop inside the mask.

    # the last image is the mask

    # print(image_list[0].shape)
    image_list[2] = ndimage.grey_dilation(image_list[2], size=(5, 5, 5))  # (dilated mask)
    image_list[2] = ndimage.gaussian_filter(image_list[2], sigma=2)

    a, b, c = np.nonzero(image_list[2])
    if len(a) == 0:  # all labels are zeros
        mask_f = 0
        mask_b = image_list[2].shape[0]  # the same size as the input image
        mask_u = 0
        mask_d = image_list[2].shape[1]
        mask_l = 0
        mask_r = image_list[2].shape[2]

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
            temp[thick_diff : thick_diff + t, :, :] = image_list[i]
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
    im1_new1 = [None]* len(image_list)
    im1_new2 = [None]* len(image_list)
    output_image_list = [None]* len(image_list)

    if mask_h > height:
        top = np.random.randint(mask_u, mask_d - height)
        for i in range(len(image_list)):
            im1_new1[i] = image_list[i][:, top: top + height, :]
    elif mask_h <= height:
        if h < height:
            pad_h = np.random.randint(0, height - h)
            for i in range(len(image_list)):
                im1_new1[i] = np.zeros([t, height, w])
                im1_new1[i][:, pad_h: pad_h + h, :] = image_list[i]
        elif h > height:
            top = np.random.randint(0, min([mask_u, h - height]) + 1)
            for i in range(len(image_list)):
                im1_new1[i] = image_list[i][:, top: top + height, :]
        else:
            for i in range(len(image_list)):
                im1_new1[i] = image_list[i]


    if mask_w > width:
        left = np.random.randint(mask_l, mask_r - width)
        for i in range(len(image_list)):
            im1_new2[i] = im1_new1[i][:, :, left: left + width]
    elif mask_w <= width:
        if w < width:
            pad_w = np.random.randint(0, width - w)
            for i in range(len(image_list)):
                im1_new2[i] = np.zeros([t, height, width])
                im1_new2[i][:, :, pad_w: pad_w + w] = im1_new1[i]
        elif w > width:
            left = np.random.randint(0, min([mask_l, w - width]) + 1)
            for i in range(len(image_list)):
                im1_new2[i] = im1_new1[i][:, :, left: left + width]
        else:
            for i in range(len(image_list)):
                im1_new2[i] = im1_new1[i]

    if mask_t > thickness:
        front = np.random.randint(mask_f, mask_b - thickness)
        for i in range(len(image_list)):
            output_image_list[i] = im1_new2[i][front: front + thickness, :, :]
    elif mask_t <= thickness:
        if t < thickness:
            pad_t = np.random.randint(0, thickness - t)
            for i in range(len(image_list)):
                output_image_list[i] = np.zeros([thickness, height, width])
                output_image_list[i][pad_t: pad_t + t, :, :] = im1_new2[i]
        elif t > thickness:
            front = np.random.randint(0, min([mask_f, t - thickness]) + 1)
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


# Function fetched and adapted from this thread
# https://stackoverflow.com/questions/43922198/how-to-rotate-a-3d-image-by-a-random-angle-in-python
def randomRotation3d(image_list, max_angle, prob):
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
        for i in range(len(image_list) - 1):
            image_list[i] = scipy.ndimage.interpolation.rotate(image_list[i], angle, mode='constant', axes=(0, 1),
                                                               reshape=False, order = 3)

        image_list[-1] = scipy.ndimage.interpolation.rotate(image_list[-1], angle, mode='constant', axes=(0, 1),
                                                           reshape=False, order=0) # for mask


    rotate = np.random.random() > prob
    if rotate > 0:
        # rotate along y-axis
        angle = np.random.uniform(-max_angle, max_angle)
        for i in range(len(image_list) - 1):
            image_list[i] = scipy.ndimage.interpolation.rotate(image_list[i], angle, mode='constant', axes=(0, 2), reshape=False, order = 3)

        image_list[-1] = scipy.ndimage.interpolation.rotate(image_list[-1], angle, mode='constant', axes=(0, 2),
                                                           reshape=False, order=0)

    rotate = np.random.random() > prob
    if rotate > 0:
        # rotate along x-axis
        angle = np.random.uniform(-max_angle, max_angle)
        for i in range(len(image_list) - 1):
            image_list[i] = scipy.ndimage.interpolation.rotate(image_list[i], angle, mode='constant', axes=(1, 2), reshape=False, order = 3)

        image_list[-1] = scipy.ndimage.interpolation.rotate(image_list[-1], angle, mode='constant', axes=(1, 2),
                                                           reshape=False, order=0)

    return image_list
