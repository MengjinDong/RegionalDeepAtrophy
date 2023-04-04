# internal python imports
import os
import csv
import functools
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import sys
from itertools import permutations, combinations

# third party imports
import numpy as np
import scipy
from skimage import measure
import glob
from datetime import datetime
# local/our imports
import pystrum.pynd.ndutils as nd
from pathlib import Path
from vxm_model.py import data_aug_cpu
import logging
import matplotlib.pyplot as plt
import vxm_model as vxm


date_format = "%Y-%m-%d"

def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features


def get_backend():
    """
    Returns the currently used backend. Default is tensorflow unless the
    VXM_BACKEND environment variable is set to 'pytorch'.
    """
    return 'pytorch' if os.environ.get('VXM_BACKEND') == 'pytorch' else 'tensorflow'


def get_file_list_intra_subject(root_dir, data_dir, groups, csv_out, stages):

    date_format = "%Y-%m-%d"

    print("stages = ", stages)

    if os.path.exists(csv_out):
        os.remove(csv_out)

    with open(csv_out, 'w') as filename:
        for group in groups:
            for stage in stages:
                print(group, stage)
                wr = csv.writer(filename, lineterminator='\n')

                subject_list = root_dir + "/" + data_dir + "/subject_list_" + group + stage + ".csv"
                if os.path.exists(subject_list):
                    with open(subject_list) as f:
                        for subjectID in f:
                            subjectID = subjectID.strip('\n')
                            for side in ["left", "right"]:
                                # All filenames are useful. Randomly take two baseline images and expand them into two pairs.
                                scan_list = glob.glob(
                                    root_dir + "/T1_Input_3d" + "/*/" + stage + "/" + subjectID + "*blmptrim_" + side + "_to_hw.nii.gz")
                                for bl_item1 in list(scan_list):
                                    # print(bl_item1, bl_item2)

                                    fu_item1 = Path(
                                        bl_item1.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1))
                                    if fu_item1.exists():
                                        fname1 = bl_item1.split("/")[-1]
                                        bl_time1 = datetime.strptime(fname1.split("_")[3], date_format)
                                        fu_time1 = datetime.strptime(fname1.split("_")[4], date_format)
                                        date_diff1 = (fu_time1 - bl_time1).days

                                    else:
                                        continue
                                    wr.writerow(
                                        [bl_item1, bl_time1, fu_time1, stage,
                                         date_diff1, subjectID, side])

    with open(csv_out, 'r') as f:
        reader = csv.reader(f)
        image_frame = list(reader)

    print("Intra-subject registration, stages", stages, "groups", groups, "number of images in total:", len(image_frame))


def get_file_list_intra_subject_RISI(root_dir, data_dir, groups, csv_out, stages):

    date_format = "%Y-%m-%d"

    print("stages = ", stages)

    if os.path.exists(csv_out):
        os.remove(csv_out)

    with open(csv_out, 'w') as filename:
        wr = csv.writer(filename, lineterminator='\n')

        for group in groups:
            for stage in stages:
                print(group, stage)
                subject_list = root_dir + "/" + data_dir + "/subject_list_" + group + stage + ".csv"
                if os.path.exists(subject_list):
                    with open(subject_list) as f:
                        for subjectID in f:
                            subjectID = subjectID.strip('\n')
                            for side in ["left", "right"]:
                                # All filenames are useful. Randomly take two baseline images and expand them into two pairs.
                                scan_list = glob.glob(
                                    root_dir + "/T1_Input_3d" + "/*/" + stage + "/" + subjectID + "*blmptrim_" + side + "_to_hw.nii.gz")
                                perm = permutations(range(0, len(scan_list)), 2)
                                for bl_item1, bl_item2 in list(perm):
                                    bl_item1 = scan_list[bl_item1]
                                    bl_item2 = scan_list[bl_item2]
                                    # print(bl_item1, bl_item2)

                                    fu_item1 = Path(
                                        bl_item1.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1))
                                    mask_item1 = Path(
                                        bl_item1.replace('blmptrim_', "blmptrim_seg_", 1).replace('_to_hw', "_tohw", 1))
                                    if fu_item1.exists() and mask_item1.exists():
                                        fname1 = bl_item1.split("/")[-1]
                                        bl_time1 = datetime.strptime(fname1.split("_")[3], date_format)
                                        fu_time1 = datetime.strptime(fname1.split("_")[4], date_format)
                                        date_diff1 = (fu_time1 - bl_time1).days
                                        label_date_diff1 = float(np.greater(date_diff1, 0))
                                    else:
                                        continue

                                    fu_item2 = Path(
                                        bl_item2.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1))
                                    mask_item2 = Path(
                                        bl_item2.replace('blmptrim_', "blmptrim_seg_", 1).replace('_to_hw', "_tohw", 1))

                                    if fu_item2.exists() and mask_item2.exists():
                                        fname2 = bl_item2.split("/")[-1]
                                        bl_time2 = datetime.strptime(fname2.split("_")[3], date_format)
                                        fu_time2 = datetime.strptime(fname2.split("_")[4], date_format)
                                        date_diff2 = (fu_time2 - bl_time2).days
                                        label_date_diff2 = float(np.greater(date_diff2, 0))
                                    else:
                                        continue

                                    date_diff_ratio = abs(date_diff1 / date_diff2)

                                    if date_diff_ratio < 0.5:
                                        label_time_interval = 0
                                    elif date_diff_ratio < 1:
                                        label_time_interval = 1
                                    elif date_diff_ratio < 2:
                                        label_time_interval = 2
                                    else:
                                        label_time_interval = 3

                                    if abs(date_diff1) > abs(date_diff2):
                                        if date_diff1 > 0:
                                            if bl_time2 <= fu_time1 and bl_time2 >= bl_time1 and fu_time2 <= fu_time1 and fu_time2 >= bl_time1:
                                                wr.writerow(
                                                    [bl_item1, bl_item2, bl_time1, fu_time1, bl_time2, fu_time2, stage,
                                                     date_diff1, date_diff2,
                                                     label_date_diff1, label_date_diff2, label_time_interval, subjectID,
                                                     side])
                                        else:
                                            if bl_time2 <= bl_time1 and bl_time2 >= fu_time1 and fu_time2 <= bl_time1 and fu_time2 >= fu_time1:
                                                wr.writerow(
                                                    [bl_item1, bl_item2, bl_time1, fu_time1, bl_time2, fu_time2, stage,
                                                     date_diff1, date_diff2,
                                                     label_date_diff1, label_date_diff2, label_time_interval, subjectID,
                                                     side])

                                    elif abs(date_diff1) < abs(date_diff2):
                                        if date_diff2 > 0:
                                            if bl_time1 <= fu_time2 and bl_time1 >= bl_time2 \
                                                    and fu_time1 <= fu_time2 and fu_time1 >= bl_time2:
                                                wr.writerow(
                                                    [bl_item1, bl_item2, bl_time1, fu_time1, bl_time2, fu_time2, stage,
                                                     date_diff1, date_diff2,
                                                     label_date_diff1, label_date_diff2, label_time_interval, subjectID,
                                                     side])
                                        else:
                                            if bl_time1 <= bl_time2 and bl_time1 >= fu_time2 \
                                                    and fu_time1 <= bl_time2 and fu_time1 >= fu_time2:
                                                wr.writerow(
                                                    [bl_item1, bl_item2, bl_time1, fu_time1, bl_time2, fu_time2, stage,
                                                     date_diff1, date_diff2,
                                                     label_date_diff1, label_date_diff2, label_time_interval, subjectID,
                                                     side])

    with open(csv_out, 'r') as f:
        reader = csv.reader(f)
        image_frame = list(reader)

    print("Intra-subject registration with RISI, stages", stages, "groups", groups)
    print("Number of two-image pairs in total:", len(image_frame))


def get_file_list_inter_subject(root_dir, data_dir, groups, csv_out, stages):

    date_format = "%Y-%m-%d"

    print("stages = ", stages)

    if os.path.exists(csv_out):
        os.remove(csv_out)

    with open(csv_out, 'w') as filename:
        for group in groups:
            for stage in stages:
                print(group, stage)
                wr = csv.writer(filename, lineterminator='\n')

                subject_list = root_dir + "/" + data_dir + "/subject_list_" + group + stage + ".csv"
                if os.path.exists(subject_list):
                    with open(subject_list) as f:
                        for subjectID in f:
                            subjectID = subjectID.strip('\n')
                            for side in ["left", "right"]:
                                # All filenames are useful. Randomly take two baseline images and expand them into two pairs.
                                scan_list = glob.glob(
                                    root_dir + "/T1_Input_3d" + "/*/" + stage + "/" + subjectID + "*blmptrim_" + side + "_to_hw.nii.gz")

                                for bl_item1 in list(scan_list):
                                    # print(bl_item1, bl_item2)

                                    fu_item1 = Path(
                                        bl_item1.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1))
                                    if fu_item1.exists():
                                        fname1 = bl_item1.split("/")[-1]
                                        bl_time1 = datetime.strptime(fname1.split("_")[3], date_format)
                                        fu_time1 = datetime.strptime(fname1.split("_")[4], date_format)
                                        date_diff1 = (fu_time1 - bl_time1).days

                                    else:
                                        continue
                                    wr.writerow(
                                        [bl_item1, bl_time1, stage, date_diff1, subjectID, side])
                                    wr.writerow(
                                        [fu_item1, fu_time1, stage, date_diff1, subjectID, side])

    with open(csv_out, 'r') as f:
        reader = csv.reader(f)
        image_frame = list(reader)

    print("Inter-subject registration, stages", stages, "groups", groups, "number of images in total:", len(image_frame))


def read_file_list(filename, prefix=None, suffix=None):
    '''
    Reads a list of files from a line-seperated text file.

    Parameters:
        filename: Filename to load.
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    with open(filename, 'r') as file:
        content = file.readlines()
    filelist = [x.strip().split(",")[0] for x in content if x.strip()]
    if prefix is not None:
        filelist = [prefix + f for f in filelist]
    if suffix is not None:
        filelist = [f + suffix for f in filelist]
    return filelist

def read_file_list_RISI(filename):
    '''
    Reads a list of files from a line-seperated text file.

    Parameters:
        filename: Filename to load.
    '''
    with open(filename, 'r') as file:
        content = file.readlines()
    return content

def read_pair_list(filename, delim=None, prefix=None, suffix=None):
    '''
    Reads a list of registration file pairs from a line-seperated text file.

    Parameters:
        filename: Filename to load.
        delim: File pair delimiter. Default is a whitespace seperator (None).
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    pairlist = [f.split(delim) for f in read_file_list(filename)]
    if prefix is not None:
        pairlist = [[prefix + f for f in pair] for pair in pairlist]
    if suffix is not None:
        pairlist = [[f + suffix for f in pair] for pair in pairlist]
    return pairlist


def load_inter_volfile(
    filename,
    augment,
    np_var='vol',
    add_batch_axis=False,
    add_feat_axis=False,
    pad_shape=None,
    resize_factor=1,
    ret_affine=False,
    max_angle = 15,
    rotate_prob = 0.5,
    # output_size = [48, 80, 64],
    output_size=(96, 160, 128),
):




    """
    Loads a file in nii, nii.gz, mgz, npz, or npy format. If input file is not a string,
    returns it directly (allows files preloaded in memory to be passed to a generator)

    Parameters:
        filename: Filename to load, or preloaded volume to be returned.
        np_var: If the file is a npz (compressed numpy) with multiple variables,
            the desired variable can be specified with np_var. Default is 'vol'.
        add_batch_axis: Adds an axis to the beginning of the array. Default is False.
        add_feat_axis: Adds an axis to the end of the array. Default is False.
        pad_shape: Zero-pad the array to a target shape. Default is None.
        resize: Volume resize factor. Default is 1
        ret_affine: Additionally returns the affine transform (or None if it doesn't exist).
    """
    if isinstance(filename, str) and not os.path.isfile(filename):
        raise ValueError("'%s' is not a file." % filename)

    if not os.path.isfile(filename):
        if ret_affine:
            (vol, affine) = filename
        else:
            vol = filename
    elif filename.endswith(('.nii', '.nii.gz', '.mgz')):
        import nibabel as nib
        img = nib.load(filename)
        vol = img.get_data().squeeze()
        affine = img.affine
    elif filename.endswith('.npy'):
        vol = np.load(filename)
        affine = None
    elif filename.endswith('.npz'):
        npz = np.load(filename)
        vol = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]
        affine = None
    else:
        raise ValueError('unknown filetype for %s' % filename)

    if 'normalize' in augment:
        [vol] = data_aug_cpu.Normalize([vol])

    image_list1 = [vol]
    if 'left' in filename:
        image_list1 = data_aug_cpu.fixedFlip3d(image_list1)

    if 'flip' in augment:
        image_list1 = data_aug_cpu.randomFlip3d(image_list1)

    # Random 3D rotate image
    if 'rotate' in augment and max_angle > 0:
        image_list1 = data_aug_cpu.randomRotation3d(image_list1, max_angle, rotate_prob)

    if 'crop' in augment:
        # previously, for cropping we should read segmentation image: randomCropBySeg3d()
        # now, we randomly crop to the desired size
        image_list1 = data_aug_cpu.fixedCrop3d(image_list1, output_size)

    vol = image_list1[0]

    if pad_shape:
        vol, _ = pad(vol, pad_shape)

    if add_feat_axis:
        vol = vol[..., np.newaxis]

    if resize_factor != 1:
        vol = resize(vol, resize_factor)

    if add_batch_axis:
        vol = vol[np.newaxis, ...]

    return [vol, affine] if ret_affine else vol


def load_bl_fu_file(
        bl_name,
        augment='normalize',
        np_var='vol',
        add_batch_axis=False,
        add_feat_axis=False,
        pad_shape=None,
        resize_factor=1,
        ret_affine=False,
        segs=None,
        max_angle=15,
        rotate_prob=0.5,
        # output_size=[48, 80, 64],
        output_size=(96, 160, 128),
):

    """
    Loads a file in nii, nii.gz, mgz, npz, or npy format. If input file is not a string,
    returns it directly (allows files preloaded in memory to be passed to a generator)

    Parameters:
        bl_name: bl_name to load, or preloaded volume to be returned.
        np_var: If the file is a npz (compressed numpy) with multiple variables,
            the desired variable can be specified with np_var. Default is 'vol'.
        add_batch_axis: Adds an axis to the beginning of the array. Default is False.
        add_feat_axis: Adds an axis to the end of the array. Default is False.
        pad_shape: Zero-pad the array to a target shape. Default is None.
        resize: Volume resize factor. Default is 1
        ret_affine: Additionally returns the affine transform (or None if it doesn't exist).
    """
    if isinstance(bl_name, str) and not os.path.isfile(bl_name):
        raise ValueError("'%s' is not a file." % bl_name)

    # extract date difference and date difference label
    bl_time = datetime.strptime(bl_name.split("/")[-1].split("_")[3], date_format)
    fu_time = datetime.strptime(bl_name.split("/")[-1].split("_")[4], date_format)
    date_diff_true = int((fu_time - bl_time).days)

    fu_name = bl_name.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1)
    if segs:
        seg_name = bl_name.replace('blmptrim_', "blmptrim_seg_", 1).replace('_to_hw', '', 1)
    if not os.path.isfile(bl_name):
        if ret_affine:
            (vol_bl, affine_bl) = bl_name
            (vol_fu, affine_fu) = fu_name
            if segs:
                (vol_seg, affine_seg) = seg_name
        else:
            vol_bl = bl_name
            vol_fu = fu_name
            if segs:
                vol_seg = seg_name

    elif bl_name.endswith(('.nii', '.nii.gz', '.mgz')):
        import nibabel as nib
        img = nib.load(bl_name)
        vol_bl = img.get_data().squeeze()
        if ret_affine:
            affine_bl = img.affine

        img = nib.load(fu_name)
        vol_fu = img.get_data().squeeze()
        if ret_affine:
            affine_fu = img.affine
            if not np.array_equal(affine_bl, affine_fu):
                print("bl and fu are not in the same space!")

        if segs:
            img = nib.load(seg_name)
            vol_seg = img.get_data().squeeze()

    elif bl_name.endswith('.npy'):
        vol_bl = np.load(bl_name)
        if ret_affine:
            affine_bl = None
        vol_fu = np.load(fu_name)
        if segs:
            vol_seg = np.load(seg_name)

    elif bl_name.endswith('.npz'):
        npz = np.load(bl_name)
        vol_bl = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]
        if ret_affine:
            affine_bl = None

        npz = np.load(fu_name)
        vol_fu = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]

        if segs:
            npz = np.load(seg_name)
            vol_seg = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]

    else:
            raise ValueError('unknown filetype for %s' % bl_name)

    orig_shape = vol_bl.shape

    if 'normalize' in augment:
        vol_bl, vol_fu = data_aug_cpu.Normalize([vol_bl, vol_fu])

    if 'erase' in augment:
        vol_bl, vol_fu = data_aug_cpu.randomErase3d([vol_bl, vol_fu])

    if segs:
        image_list1 = [vol_bl, vol_fu, vol_seg]
    else:
        image_list1 = [vol_bl, vol_fu]

    if 'flip' in augment:
        image_list1 = data_aug_cpu.randomFlip3d(image_list1)

    # Random 3D rotate image
    if 'rotate' in augment and max_angle > 0:
        image_list1 = data_aug_cpu.randomRotation3d(image_list1, max_angle, rotate_prob)

    if 'crop' in augment:
        if segs:
            image_list1, crop_pos = data_aug_cpu.randomCropBySeg3d(image_list1, output_size)
        else:
            # previously, for cropping we should read segmentation image: randomCropBySeg3d()
            # now, we randomly crop to the desired size
            image_list1, crop_pos = data_aug_cpu.randomCrop3d(image_list1, output_size)

    if segs:
        vol_bl = image_list1[0]
        vol_fu = image_list1[1]
        vol_seg = image_list1[2]
    else:
        vol_bl = image_list1[0]
        vol_fu = image_list1[1]

    if add_feat_axis:
        vol_bl = vol_bl[..., np.newaxis]
        vol_fu = vol_fu[..., np.newaxis]
        if segs:
            vol_seg = vol_seg[..., np.newaxis]

    if pad_shape:
        vol_bl, _ = pad(vol_bl, pad_shape)
        vol_fu, _ = pad(vol_fu, pad_shape)
        if segs:
            vol_seg, _ = pad(vol_seg, pad_shape)

    if resize_factor != 1:
        vol_bl = resize(vol_bl, resize_factor)
        vol_fu = resize(vol_fu, resize_factor)
        if segs:
            vol_seg = resize(vol_seg, resize_factor)

    if add_batch_axis:
        vol_bl = vol_bl[np.newaxis, ...]
        vol_fu = vol_fu[np.newaxis, ...]
        if segs:
            vol_seg = vol_seg[np.newaxis, ...]

    # return only baseline affine, not followup affine
    if ret_affine:
        if segs:
            return [vol_bl, vol_fu, vol_seg, date_diff_true, affine_bl, orig_shape, crop_pos]
        else:
            return [vol_bl, vol_fu, date_diff_true, affine_bl, orig_shape, crop_pos]
    else:
        if segs:
            return [vol_bl, vol_fu, vol_seg, date_diff_true]
        else:
            return [vol_bl, vol_fu, date_diff_true]

    # return [vol_bl, affine_bl, vol_fu] if ret_affine else [vol_bl, vol_fu]


def load_bl_fu_file_RISI(
        line,
        augment='normalize',
        np_var='vol',
        add_batch_axis=False,
        add_feat_axis=False,
        pad_shape=None,
        resize_factor=1,
        ret_affine=False,
        segs=None,
        max_angle=15,
        rotate_prob=0.5,
        # output_size=[48, 80, 64],
        output_size=(96, 160, 128),
        num_attn_maps=1
):

    """
    Loads a file in nii, nii.gz, mgz, npz, or npy format. If input file is not a string,
    returns it directly (allows files preloaded in memory to be passed to a generator)

    Parameters:
        bl_name: bl_name to load, or preloaded volume to be returned.
        np_var: If the file is a npz (compressed numpy) with multiple variables,
            the desired variable can be specified with np_var. Default is 'vol'.
        add_batch_axis: Adds an axis to the beginning of the array. Default is False.
        add_feat_axis: Adds an axis to the end of the array. Default is False.
        pad_shape: Zero-pad the array to a target shape. Default is None.
        resize: Volume resize factor. Default is 1
        ret_affine: Additionally returns the affine transform (or None if it doesn't exist).
    """
    line = line.split(",")
    bl_name1 = line[0]
    bl_name2 = line[1]
    if isinstance(bl_name1, str) and not os.path.isfile(bl_name1):
        raise ValueError("'%s' is not a file." % bl_name1)

    if isinstance(bl_name2, str) and not os.path.isfile(bl_name2):
        raise ValueError("'%s' is not a file." % bl_name2)

    single_sample1 = read_bl_fu_files(bl_name1,
                                      ret_affine=ret_affine,
                                      segs=segs,
                                      np_var=np_var,
                                      num_attn_maps=num_attn_maps)
    single_sample2 = read_bl_fu_files(bl_name2,
                                      ret_affine=ret_affine,
                                      segs=segs,
                                      np_var=np_var,
                                      num_attn_maps=num_attn_maps)

    single_sample1['orig_shape'] = single_sample1['vol_bl'].shape
    single_sample2['orig_shape'] = single_sample2['vol_bl'].shape

    if num_attn_maps == 2:
        if segs:
            single_sample1["vol_bl"], \
            single_sample1["vol_fu"], \
            single_sample1["crop_pos"], \
            single_sample1["vol_seg"] = single_sample_aug(single_sample1["vol_bl"],
                                                          single_sample1["vol_fu"],
                                                          augment=augment,
                                                          add_batch_axis=add_batch_axis,
                                                          add_feat_axis=add_feat_axis,
                                                          pad_shape=pad_shape,
                                                          resize_factor=resize_factor,
                                                          segs=segs,
                                                          vol_seg=single_sample1["vol_seg"],
                                                          max_angle=max_angle,
                                                          rotate_prob=rotate_prob,
                                                          # output_size=[48, 80, 64],
                                                          output_size=output_size
                                                          )

            single_sample2["vol_bl"], \
            single_sample2["vol_fu"], \
            single_sample2["crop_pos"], \
            single_sample2["vol_seg"] = single_sample_aug(single_sample2["vol_bl"],
                                                          single_sample2["vol_fu"],
                                                          augment=augment,
                                                          add_batch_axis=add_batch_axis,
                                                          add_feat_axis=add_feat_axis,
                                                          pad_shape=pad_shape,
                                                          resize_factor=resize_factor,
                                                          segs=segs,
                                                          vol_seg=single_sample2["vol_seg"],
                                                          max_angle=max_angle,
                                                          rotate_prob=rotate_prob,
                                                          # output_size=[48, 80, 64],
                                                          output_size=output_size)

        else:
            single_sample1["vol_bl"], \
            single_sample1["vol_fu"], \
            single_sample1["crop_pos"] = single_sample_aug(single_sample1["vol_bl"],
                                         single_sample1["vol_fu"],
                                         augment=augment,
                                         add_batch_axis=add_batch_axis,
                                         add_feat_axis=add_feat_axis,
                                         pad_shape=pad_shape,
                                         resize_factor=resize_factor,
                                         segs=segs,
                                         max_angle=max_angle,
                                         rotate_prob=rotate_prob,
                                         # output_size=[48, 80, 64],
                                         output_size=output_size)


            single_sample2["vol_bl"], \
            single_sample2["vol_fu"], \
            single_sample2["crop_pos"] = single_sample_aug(single_sample2["vol_bl"],
                                         single_sample2["vol_fu"],
                                         augment=augment,
                                         add_batch_axis=add_batch_axis,
                                         add_feat_axis=add_feat_axis,
                                         pad_shape=pad_shape,
                                         resize_factor=resize_factor,
                                         segs=segs,
                                         max_angle=max_angle,
                                         rotate_prob=rotate_prob,
                                         # output_size=[48, 80, 64],
                                         output_size=output_size)

    elif num_attn_maps == 1:
        if segs:
            single_sample1["vol_bl"], \
            single_sample1["vol_fu"], \
            single_sample1["crop_pos"], \
            single_sample2["vol_bl"], \
            single_sample2["vol_fu"], \
            single_sample2["crop_pos"], \
            single_sample1["vol_seg"], \
            single_sample2["vol_seg"] = two_samples_aug(single_sample1["vol_bl"],
                                                        single_sample1["vol_fu"],
                                                        single_sample2["vol_bl"],
                                                        single_sample2["vol_fu"],
                                                        augment=augment,
                                                        add_batch_axis=add_batch_axis,
                                                        add_feat_axis=add_feat_axis,
                                                        pad_shape=pad_shape,
                                                        resize_factor=resize_factor,
                                                        segs=segs,
                                                        vol_seg_orig1=single_sample1["vol_seg"],
                                                        vol_seg_orig2=single_sample2["vol_seg"],
                                                        max_angle=max_angle,
                                                        rotate_prob=rotate_prob,
                                                        # output_size=[48, 80, 64],
                                                        output_size=output_size
                                                        )

        else:
            single_sample1["vol_bl"], \
            single_sample1["vol_fu"], \
            single_sample1["crop_pos"], \
            single_sample2["vol_bl"], \
            single_sample2["vol_fu"], \
            single_sample2["crop_pos"] = two_samples_aug(single_sample1["vol_bl"],
                                                         single_sample1["vol_fu"],
                                                         single_sample2["vol_bl"],
                                                         single_sample2["vol_fu"],
                                                         augment=augment,
                                                         add_batch_axis=add_batch_axis,
                                                         add_feat_axis=add_feat_axis,
                                                         pad_shape=pad_shape,
                                                         resize_factor=resize_factor,
                                                         segs=segs,
                                                         max_angle=max_angle,
                                                         rotate_prob=rotate_prob,
                                                         # output_size=[48, 80, 64],
                                                         output_size=output_size)

    # return only baseline affine, not followup affine
    return single_sample1, single_sample2

def read_bl_fu_files(bl_name, ret_affine=False, segs=False, np_var='vol', num_attn_maps=2):
    # extract date difference and date difference label
    bl_time = datetime.strptime(bl_name.split("/")[-1].split("_")[3], date_format)
    fu_time = datetime.strptime(bl_name.split("/")[-1].split("_")[4], date_format)
    date_diff_true = int((fu_time - bl_time).days)

    fu_name = bl_name.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1)

    if segs and num_attn_maps == 2: # use hippo and all surrounding areas
        seg_name = bl_name.replace('blmptrim_', "blmptrim_seg_", 1).replace('_to_hw', '', 1)
    elif segs and num_attn_maps == 1: # use hippo area only, "blmptrim_seg_ippo_"
        seg_name = bl_name.replace('blmptrim_', "blmptrim_seg_", 1).replace('_to_hw', '', 1)

    single_sample = {}
    single_sample["bl_time"] = bl_time
    single_sample["fu_time"] = fu_time
    single_sample["date_diff_true"] = date_diff_true

    if not os.path.isfile(bl_name):
        if ret_affine:
            (vol_bl, affine_bl) = bl_name
            (vol_fu, affine_fu) = fu_name
            if segs:
                (vol_seg, affine_seg) = seg_name
        else:
            vol_bl = bl_name
            vol_fu = fu_name
            if segs:
                vol_seg = seg_name


    elif bl_name.endswith(('.nii', '.nii.gz', '.mgz')):
        import nibabel as nib
        img = nib.load(bl_name)
        vol_bl = img.get_data().squeeze()
        if ret_affine:
            affine_bl = img.affine

        img = nib.load(fu_name)
        vol_fu = img.get_data().squeeze()
        if ret_affine:
            affine_fu = img.affine
            if not np.array_equal(affine_bl, affine_fu):
                print("bl and fu are not in the same space!")

        if segs:
            img = nib.load(seg_name)
            vol_seg = img.get_data().squeeze()

    elif bl_name.endswith('.npy'):
        vol_bl = np.load(bl_name)
        if ret_affine:
            affine_bl = None
        vol_fu = np.load(fu_name)
        if segs:
            vol_seg = np.load(seg_name)

    elif bl_name.endswith('.npz'):
        npz = np.load(bl_name)
        vol_bl = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]
        if ret_affine:
            affine_bl = None

        npz = np.load(fu_name)
        vol_fu = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]

        if segs:
            npz = np.load(seg_name)
            vol_seg = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]

    else:
        raise ValueError('unknown filetype for %s' % bl_name)

    single_sample["vol_bl"] = vol_bl
    single_sample["vol_fu"] = vol_fu

    if segs:
        single_sample["vol_seg"] = vol_seg

    if ret_affine:
        single_sample["affine_bl"] = affine_bl
        single_sample["affine_fu"] = affine_fu

    if segs and ret_affine:
        single_sample["affine_seg"] = affine_seg

    return single_sample


def single_sample_aug(vol_bl,
                      vol_fu,
                      augment='normalize',
                      add_batch_axis=False,
                      add_feat_axis=False,
                      pad_shape=None,
                      resize_factor=1,
                      segs=None,
                      vol_seg=None,
                      max_angle=15,
                      rotate_prob=0.5,
                      # output_size=[48, 80, 64],
                      output_size=(96, 160, 128)):

    if 'normalize' in augment:
        vol_bl, vol_fu = data_aug_cpu.Normalize([vol_bl, vol_fu])

    if 'erase' in augment:
        vol_bl, vol_fu = data_aug_cpu.randomErase3d([vol_bl, vol_fu])

    if segs:
        image_list1 = [vol_bl, vol_fu, vol_seg]
    else:
        image_list1 = [vol_bl, vol_fu]

    if 'flip' in augment:
        image_list1 = data_aug_cpu.randomFlip3d(image_list1)

    # Random 3D rotate image
    if 'rotate' in augment and max_angle > 0:
        image_list1 = data_aug_cpu.randomRotation3d(image_list1, max_angle, rotate_prob, segs=segs)

    if 'crop' in augment:
        if segs:
            image_list1, crop_pos = data_aug_cpu.randomCropBySeg3d(image_list1, output_size)
        else:
            # previously, for cropping we should read segmentation image: randomCropBySeg3d()
            # now, we randomly crop to the desired size
            image_list1, crop_pos = data_aug_cpu.randomCrop3d(image_list1, output_size)

    if segs:
        vol_bl = image_list1[0]
        vol_fu = image_list1[1]
        vol_seg = image_list1[2]
    else:
        vol_bl = image_list1[0]
        vol_fu = image_list1[1]

    if add_feat_axis:
        vol_bl = vol_bl[..., np.newaxis]
        vol_fu = vol_fu[..., np.newaxis]
        if segs:
            vol_seg = vol_seg[..., np.newaxis]

    if pad_shape:
        vol_bl, _ = pad(vol_bl, pad_shape)
        vol_fu, _ = pad(vol_fu, pad_shape)
        if segs:
            vol_seg, _ = pad(vol_seg, pad_shape)

    if resize_factor != 1:
        vol_bl = resize(vol_bl, resize_factor)
        vol_fu = resize(vol_fu, resize_factor)
        if segs:
            vol_seg = resize(vol_seg, resize_factor)

    if add_batch_axis:
        vol_bl = vol_bl[np.newaxis, ...]
        vol_fu = vol_fu[np.newaxis, ...]
        if segs:
            vol_seg = vol_seg[np.newaxis, ...]

    if segs:
        return vol_bl, vol_fu, crop_pos, vol_seg
    else:
        return vol_bl, vol_fu, crop_pos


def two_samples_aug(vol_bl1,
                    vol_fu1,
                    vol_bl2,
                    vol_fu2,
                    augment='normalize',
                    add_batch_axis=False,
                    add_feat_axis=False,
                    pad_shape=None,
                    resize_factor=1,
                    segs=None,
                    vol_seg_orig1=None,
                    vol_seg_orig2=None,
                    max_angle=15,
                    rotate_prob=0.5,
                    # output_size=[48, 80, 64],
                    output_size=(96, 160, 128),
                    ):

    if 'normalize' in augment:
        vol_bl1, vol_fu1 = data_aug_cpu.Normalize([vol_bl1, vol_fu1])
        vol_bl2, vol_fu2 = data_aug_cpu.Normalize([vol_bl2, vol_fu2])

    # TODO: match dimensions of pair1 and pair2 by seg


    if 'erase' in augment:
        vol_bl1, vol_fu1, vol_bl2, vol_fu2 = data_aug_cpu.randomErase3d([vol_bl1, vol_fu1, vol_bl2, vol_fu2])

    if segs:
        image_list1 = [vol_bl1, vol_fu1, vol_seg_orig1, vol_bl2, vol_fu2, vol_seg_orig2]
    else:
        image_list1 = [vol_bl1, vol_fu1, vol_bl2, vol_fu2]

    if 'flip' in augment:
        image_list1 = data_aug_cpu.randomFlip3d(image_list1)

    # Random 3D rotate image
    if 'rotate' in augment and max_angle > 0:
        image_list1 = data_aug_cpu.randomRotation3d(image_list1, max_angle, rotate_prob)

    if 'crop' in augment:
        if segs:
            [vol_bl1, vol_fu1, vol_seg1], crop_pos1 = data_aug_cpu.randomCropBySeg3d(image_list1[:3], output_size)
            image_list1[3:], crop_pos2 = data_aug_cpu.alignCenterCrop3d(image_list1[3:], crop_pos1, output_size, segs=segs,
                                                                vol_seg1=vol_seg_orig1)

        else:
            # previously, for cropping we should read segmentation image: randomCropBySeg3d()
            # now, we randomly crop to the desired size
            vol_bl1, vol_fu1, crop_pos1 = data_aug_cpu.randomCrop3d(image_list1[:2], output_size)
            image_list1[2:], crop_pos2 = data_aug_cpu.alignCenterCrop3d(image_list1[2:], crop_pos1, output_size, segs=segs)


    if segs:
        vol_bl2 = image_list1[3]
        vol_fu2 = image_list1[4]
        vol_seg2 = image_list1[5]
    else:
        vol_bl2 = image_list1[2]
        vol_fu2 = image_list1[3]

    if add_feat_axis:
        vol_bl1 = vol_bl1[..., np.newaxis]
        vol_fu1 = vol_fu1[..., np.newaxis]
        vol_bl2 = vol_bl2[..., np.newaxis]
        vol_fu2 = vol_fu2[..., np.newaxis]
        if segs:
            vol_seg1 = vol_seg1[..., np.newaxis]
            vol_seg2 = vol_seg2[..., np.newaxis]

    if pad_shape:
        vol_bl1, _ = pad(vol_bl1, pad_shape)
        vol_fu1, _ = pad(vol_fu1, pad_shape)
        vol_bl2, _ = pad(vol_bl2, pad_shape)
        vol_fu2, _ = pad(vol_fu2, pad_shape)
        if segs:
            vol_seg1, _ = pad(vol_seg1, pad_shape)
            vol_seg2, _ = pad(vol_seg2, pad_shape)

    if resize_factor != 1:
        vol_bl1 = resize(vol_bl1, resize_factor)
        vol_fu1 = resize(vol_fu1, resize_factor)
        vol_bl2 = resize(vol_bl2, resize_factor)
        vol_fu2 = resize(vol_fu2, resize_factor)
        if segs:
            vol_seg1 = resize(vol_seg1, resize_factor)
            vol_seg2 = resize(vol_seg2, resize_factor)

    if add_batch_axis:
        vol_bl1 = vol_bl1[np.newaxis, ...]
        vol_fu1 = vol_fu1[np.newaxis, ...]
        vol_bl2 = vol_bl2[np.newaxis, ...]
        vol_fu2 = vol_fu2[np.newaxis, ...]
        if segs:
            vol_seg1 = vol_seg1[np.newaxis, ...]
            vol_seg2 = vol_seg2[np.newaxis, ...]

    if segs:
        return vol_bl1, vol_fu1, crop_pos1, vol_bl2, vol_fu2, crop_pos2, vol_seg1, vol_seg2
    else:
        return vol_bl1, vol_fu1, crop_pos1, vol_bl2, vol_fu2, crop_pos2


def save_volfile(array, filename, affine=None):
    """
    Saves an array to nii, nii.gz, or npz format.

    Parameters:
        array: The array to save.
        filename: Filename to save to.
        affine: Affine vox-to-ras matrix. Saves LIA matrix if None (default).
    """
    if filename.endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        if affine is None and array.ndim >= 3:
            # use LIA transform as default affine
            affine = np.array([[-1, 0, 0, 0],  # nopep8
                               [0, 0, 1, 0],  # nopep8
                               [0, -1, 0, 0],  # nopep8
                               [0, 0, 0, 1]], dtype=float)  # nopep8
            pcrs = np.append(np.array(array.shape[:3]) / 2, 1)
            affine[:3, 3] = -np.matmul(affine, pcrs)[:3]
            nib.save(nib.Nifti1Image(array, affine), filename)
        elif len(affine) == 3 and array.ndim >= 3:
            aff = affine[0]
            orig_shape = affine[1]
            crop_pos = affine[2]
            arr = restore_original_image(array, orig_shape, crop_pos)
            nib.save(nib.Nifti1Image(arr, aff), filename)
        else:
            nib.save(nib.Nifti1Image(array, affine), filename)
    elif filename.endswith('.npz'):
        np.savez_compressed(filename, vol=array)
    else:
        raise ValueError('unknown filetype for %s' % filename)


def restore_original_image(array, orig_shape, crop_pos):
    image_shape = array.shape[-3:]
    big_shape = np.max([image_shape, orig_shape], axis=0)
    # a big shape to maintain all data, and then crop
    arr1 = np.zeros(big_shape)
    # if max(crop_pos) > 0:
    #     print("here, max(crop_pos) > 0")
    start10 = max(crop_pos[0], 0)
    start11 = max(crop_pos[1], 0)
    start12 = max(crop_pos[2], 0)
    arr1[start10:start10 + image_shape[0],
         start11:start11 + image_shape[1],
         start12:start12 + image_shape[2]] = array

    # This part might be wrong because there's no case crop_pos is positive.
    # So no testing data.
    start20 = -min(crop_pos[0], 0)
    start21 = -min(crop_pos[1], 0)
    start22 = -min(crop_pos[2], 0)
    arr2 = arr1[start20:start20 + orig_shape[0],
                start21:start21 + orig_shape[1],
                start22:start22 + orig_shape[2]]

    if arr2.shape != tuple(orig_shape):
        print("didn't map to original shape")

    return arr2


def load_labels(arg):
    """
    Load label maps and return a list of unique labels as well as all maps.

    Parameters:
        arg: Path to folder containing label maps, string for globbing, or a list of these.

    Returns:
        np.array: List of unique labels.
        list: List of label maps, each as a np.array.
    """
    if not isinstance(arg, (tuple, list)):
        arg = [arg]

    # List files.
    import glob
    ext = ('.nii.gz', '.nii', '.mgz', '.npy', '.npz')
    files = [os.path.join(f, '*') if os.path.isdir(f) else f for f in arg]
    files = sum((glob.glob(f) for f in files), [])
    files = [f for f in files if f.endswith(ext)]

    # Load labels.
    if len(files) == 0:
        raise ValueError(f'no labels found for argument "{files}"')
    label_maps = []
    shape = None
    for f in files:
        x = np.squeeze(load_volfile(f))
        if shape is None:
            shape = np.shape(x)
        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError(f'file "{f}" has non-integral data type')
        if not np.all(x.shape == shape):
            raise ValueError(f'shape {x.shape} of file "{f}" is not {shape}')
        label_maps.append(x)

    return np.unique(label_maps), label_maps


def load_pheno_csv(filename, training_files=None):
    """
    Loads an attribute csv file into a dictionary. Each line in the csv should represent
    attributes for a single training file and should be formatted as:

    filename,attr1,attr2,attr2...

    Where filename is the file basename and each attr is a floating point number. If
    a list of training_files is specified, the dictionary file keys will be updated
    to match the paths specified in the list. Any training files not found in the
    loaded dictionary are pruned.
    """

    # load csv into dictionary
    pheno = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)
        for row in csv_reader:
            pheno[row[0]] = np.array([float(f) for f in row[1:]])

    # make list of valid training files
    if training_files is None:
        training_files = list(training_files.keys())
    else:
        training_files = [f for f in training_files if os.path.basename(f) in pheno.keys()]
        # make sure pheno dictionary includes the correct path to training data
        for f in training_files:
            pheno[f] = pheno[os.path.basename(f)]

    return pheno, training_files


def pad(array, shape):
    """
    Zero-pads an array to a given shape. Returns the padded array and crop slices.
    """
    if array.shape == tuple(shape):
        return array, ...

    padded = np.zeros(shape, dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
    slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape)])
    padded[slices] = array

    return padded, slices


def resize(array, factor, batch_axis=False):
    """
    Resizes an array by a given factor. This expects the input array to include a feature dimension.
    Use batch_axis=True to avoid resizing the first (batch) dimension.
    """
    if factor == 1:
        return array
    else:
        if not batch_axis:
            dim_factors = [factor for _ in array.shape[:-1]] + [1]
        else:
            dim_factors = [1] + [factor for _ in array.shape[1:-1]] + [1]
        return scipy.ndimage.interpolation.zoom(array, dim_factors, order=0)


def dice(array1, array2, labels=None, include_zero=False):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.

    Parameters:
        array1: Input array 1.
        array2: Input array 2.
        labels: List of labels to compute dice on. If None, all labels will be used.
        include_zero: Include label 0 in label list. Default is False.
    """
    if labels is None:
        labels = np.concatenate([np.unique(a) for a in [array1, array2]])
        labels = np.sort(np.unique(labels))
    if not include_zero:
        labels = np.delete(labels, np.argwhere(labels == 0)) 

    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem


def affine_shift_to_matrix(trf, resize=None, unshift_shape=None):
    """
    Converts an affine shift to a matrix (over the identity).
    To convert back from center-shifted transform, provide image shape
    to unshift_shape.

    TODO: make ND compatible - currently just 3D
    """
    matrix = np.concatenate([trf.reshape((3, 4)), np.zeros((1, 4))], 0) + np.eye(4)
    if resize is not None:
        matrix[:3, -1] *= resize
    if unshift_shape is not None:
        T = np.zeros((4, 4))
        T[:3, 3] = (np.array(unshift_shape) - 1) / 2
        matrix = (np.eye(4) + T) @ matrix @ (np.eye(4) - T)
    return matrix


def extract_largest_vol(bw, connectivity=1):
    """
    Extracts the binary (boolean) image with just the largest component.
    TODO: This might be less than efficiently implemented.
    """
    lab = measure.label(bw.astype('int'), connectivity=connectivity)
    regions = measure.regionprops(lab, cache=False)
    areas = [f.area for f in regions]
    ai = np.argsort(areas)[::-1]
    bw = lab == ai[0] + 1
    return bw


def clean_seg(x, std=1):
    """
    Cleans a segmentation image.
    """

    # take out islands, fill in holes, and gaussian blur
    bw = extract_largest_vol(x)
    bw = 1 - extract_largest_vol(1 - bw)
    gadt = scipy.ndimage.gaussian_filter(bw.astype('float'), std)

    # figure out the proper threshold to maintain the total volume
    sgadt = np.sort(gadt.flatten())[::-1]
    thr = sgadt[np.ceil(bw.sum()).astype(int)]
    clean_bw = gadt > thr

    assert np.isclose(bw.sum(), clean_bw.sum(), atol=5), 'cleaning segmentation failed'
    return clean_bw.astype(float)


def clean_seg_batch(X_label, std=1):
    """
    Cleans batches of segmentation images.
    """
    if not X_label.dtype == 'float':
        X_label = X_label.astype('float')

    data = np.zeros(X_label.shape)
    for xi, x in enumerate(X_label):
        data[xi, ..., 0] = clean_seg(x[..., 0], std)

    return data


def filter_labels(atlas_vol, labels):
    """
    Filters given volumes to only include given labels, all other voxels are set to 0.
    """
    mask = np.zeros(atlas_vol.shape, 'bool')
    for label in labels:
        mask = np.logical_or(mask, atlas_vol == label)
    return atlas_vol * mask


def dist_trf(bwvol):
    """
    Computes positive distance transform from positive entries in a logical image.
    """
    revbwvol = np.logical_not(bwvol)
    return scipy.ndimage.morphology.distance_transform_edt(revbwvol)


def signed_dist_trf(bwvol):
    """
    Computes the signed distance transform from the surface between the binary
    elements of an image
    NOTE: The distance transform on either side of the surface will be +/- 1,
    so there are no voxels for which the distance should be 0.
    NOTE: Currently the function uses bwdist twice. If there is a quick way to
    compute the surface, bwdist could be used only once.
    """

    # get the positive transform (outside the positive island)
    posdst = dist_trf(bwvol)

    # get the negative transform (distance inside the island)
    notbwvol = np.logical_not(bwvol)
    negdst = dist_trf(notbwvol)

    # combine the positive and negative map
    return posdst * notbwvol - negdst * bwvol


def vol_to_sdt(X_label, sdt=True, sdt_vol_resize=1):
    """
    Computes the signed distance transform from a volume.
    """

    X_dt = signed_dist_trf(X_label)

    if not (sdt_vol_resize == 1):
        if not isinstance(sdt_vol_resize, (list, tuple)):
            sdt_vol_resize = [sdt_vol_resize] * X_dt.ndim
        if any([f != 1 for f in sdt_vol_resize]):
            X_dt = scipy.ndimage.interpolation.zoom(X_dt, sdt_vol_resize, order=1, mode='reflect')

    if not sdt:
        X_dt = np.abs(X_dt)

    return X_dt


def vol_to_sdt_batch(X_label, sdt=True, sdt_vol_resize=1):
    """
    Computes the signed distance transforms from volume batches.
    """

    # assume X_label is [batch_size, *vol_shape, 1]
    assert X_label.shape[-1] == 1, 'implemented assuming size is [batch_size, *vol_shape, 1]'
    X_lst = [f[..., 0] for f in X_label]  # get rows
    X_dt_lst = [vol_to_sdt(f, sdt=sdt, sdt_vol_resize=sdt_vol_resize)
                for f in X_lst]  # distance transform
    X_dt = np.stack(X_dt_lst, 0)[..., np.newaxis]
    return X_dt


def  get_moving_volume(moving_segmentation):
    # get sum of positive labels multiplied by resolution
    if len(moving_segmentation.shape) == 5: # 3d
        return torch.sum(moving_segmentation, dim=(1, 2, 3, 4)) * 0.5 * 0.6
    else:
        # do not multiply by resolution on 2D images * 0.5 * 0.6
        return torch.sum(moving_segmentation, dim=(1, 2, 3))

def get_moved_volume(moving_segmentation, jdet):
    # get sum of positive labels * Jac * resolution
    if len(moving_segmentation.shape) == 5:
        return torch.sum(torch.multiply(moving_segmentation, jdet), dim=(1, 2, 3, 4)) * 0.5 * 0.6
    else:
        return torch.sum(torch.multiply(moving_segmentation, jdet), dim=(1, 2, 3))


def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims], 
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def jacobian_determinant_batch(disp):
    """
    jacobian determinant of a batch of displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    batchsize = disp.shape[0]
    volshape = disp.shape[2:]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = torch.from_numpy(np.stack(grid_lst, len(volshape))).cuda()

    Jdet = torch.zeros(batchsize, 1, *volshape).cuda()
    if nb_dims == 3:
        disp = disp.permute(0, 2, 3, 4, 1)
    elif nb_dims == 2:
        disp = disp.permute(0, 2, 3, 1)
    # compute gradients
    for i in range(batchsize):
        J = torch.gradient(disp[i] + grid)

        # 3D glow
        if nb_dims == 3:
            dx = J[0]
            dy = J[1]
            dz = J[2]

            # compute jacobian components
            Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
            Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
            Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

            Jdet[i, 0, ...] = Jdet0 - Jdet1 + Jdet2

        else:  # must be 2

            dfdx = J[0]
            dfdy = J[1]

            Jdet[i, 0, ...] =  dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

    return Jdet


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


class _TensorboardFormatter:
    """
    Tensorboard formatters converts a given batch of images (be it input/output to the network or the target segmentation
    image) to a series of images that can be displayed in tensorboard. This is the parent class for all tensorboard
    formatters which ensures that returned images are in the 'CHW' format.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, name, batch, normalize = True):
        """
        Transform a batch to a series of tuples of the form (tag, img), where `tag` corresponds to the image tag
        and `img` is the image itself.

        Args:
             name (str): one of 'inputs'/'targets'/'predictions'
             batch (torch.tensor): 4D or 5D torch tensor
             normalize (bool): whether normalize the image
        """

        def _check_img(tag_img):
            tag, img = tag_img

            assert img.ndim == 2 or img.ndim == 3, 'Only 2D (HW) and 3D (CHW) images are accepted for display'

            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            else:
                C = img.shape[0]
                assert C == 1 or C == 3, 'Only (1, H, W) or (3, H, W) images are supported'

            return tag, img

        tagged_images = self.process_batch(name, batch, normalize=normalize)

        return list(map(_check_img, tagged_images))

    def process_batch(self, name, batch, normalize = True):
        raise NotImplementedError


class DefaultTensorboardFormatter(_TensorboardFormatter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process_batch(self, name, batch, image_names = None, normalize = True):
        if image_names:
            tag_template = '{}/{}'
        else:
            tag_template = '{}/batch_{}/channel_{}/slice_{}'

        tagged_images = []

        if batch.ndim == 5:
            # NCDHW
            slice_idx = batch.shape[2] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):

                    if image_names:
                        image_name = "/".join(image_names[batch_idx].split("/")[-3:])
                        tag = tag_template.format(name, image_name)
                    else:
                        tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)

                    img = batch[batch_idx, channel_idx, slice_idx, ...]
                    if normalize:
                        tagged_images.append((tag, self._normalize_img(img)))
                    else:
                        tagged_images.append((tag, img))
        else:
            # batch hafrom sklearn.decomposition import PCAs no channel dim: NDHW
            slice_idx = batch.shape[1] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):

                if image_names:
                    image_name = "/".join(image_names[batch_idx].split("/")[-3:])
                    tag = tag_template.format(name, image_name)
                else:
                    tag = tag_template.format(name, batch_idx, 0, slice_idx)

                img = batch[batch_idx, slice_idx, ...]
                if normalize:
                    tagged_images.append((tag, self._normalize_img(img)))
                else:
                    tagged_images.append((tag, img))

        return tagged_images

    @staticmethod
    def _normalize_img(img):
        return np.nan_to_num((img - np.min(img)) / np.ptp(img))


def get_tensorboard_formatter():
    return DefaultTensorboardFormatter()


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
