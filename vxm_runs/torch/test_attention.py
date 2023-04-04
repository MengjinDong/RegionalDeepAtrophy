#!/usr/bin/env python

"""
Example script to test a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.
"""

import os
import random
import argparse
import time
import shutil
import csv
import numpy as np
import math
import importlib
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from functools import partial
import pickle

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import vxm_model as vxm  # nopep8
from vxm_model.py.utils import RunningAverage, get_logger, jacobian_determinant, get_moving_volume, get_moved_volume
from vxm_model.torch import layers

# parse the commandline
parser = argparse.ArgumentParser()

############################# data organization parameters ############################
parser.add_argument('--curr-machine', default='lambda',
                    help='specify on which machine the model is running')
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--test-img-list', default=None, help='line-seperated list of test files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', default='Model',
                    help='model output directory (default: Model)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--data-dir', default='final_paper',
                    help='the data split used for this experiment (default: final_paper)')
parser.add_argument('--root-dir', metavar='DIR',
                    default="/Longi_T1_2GO_QC",
                    help='path to dataset')
parser.add_argument('--vxm-dir', default='/vxm_attention_2_RISI_two_maps_20220727',
                    help='model output directory (default: Model)')
parser.add_argument('--test-stages', default="[0, 1, 3, 5]",  # "[0, 1, 3, 5]"
                    help='Input stages to test the model')
parser.add_argument('--registration-type', default="intra-subject",
                    help=' "intra-subject" registration or "inter-subject" registration ')
parser.add_argument('--save-image', default=False, type=bool,
                    help='Whether to save tested images to .nii.gz file formats')
parser.add_argument('--ret-affine', default=True,
                    help='Whether to keep affine transformations to save .nii.gz file')

############################# test parameters ############################
parser.add_argument('--n-classes', type=int, default=2, help='number of output classes for unet segmentation')
                                   # default = 2: [nothing, shrink, expand]
parser.add_argument('--in-dim', type=int, default=2, help='default dimension for unet seg input')
                    # 2: input 2 images for [bl1, bl2] for a single attention map (num_attn_maps=1),
                    #                    or [bl, fu] for two attention maps,each attention map (num_attn_maps=2) map(num_attn_maps=1)
parser.add_argument('--out-dim', type=int, default=3, help='default dimension for unet seg output')
                    # 3: output 3 channels of [shrink, bg, expand] voxels
parser.add_argument('--num-attn-maps', type=int, default=2, help='default number of attention maps for two pairs')
                    # 2: generate separate attention maps for each input pair.
parser.add_argument('--hyper-a-sigmoid', type=float, default=0.02, help='default hyper parameter for sigmoid for RISI loss')
parser.add_argument('--risi-categories', type=int, default=8, help='default number of label categories for RISI loss')
parser.add_argument('--risi-loss-weight', type=float, default=100, help='default weight of risi loss')
parser.add_argument('--test-batch-size', type=int, default=20, help='batch size (default: 1)')
parser.add_argument('--steps-per-test-epoch', type=int, default=500, help='how many steps per test epoch')
parser.add_argument('--log-after-iters', type=int, default=20, help='')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')
parser.add_argument('--radius', type=int, default=5, help='Radius for NCC loss')

############################# network architecture parameters ############################
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

############################ loss hyperparameters ############################
parser.add_argument('--test-grp', default="val", # train, val, test
                    help=' "val" or "test", first test on evaluation group, last test on test group ')
parser.add_argument('--segs', default=True,
                    help='Whether to include segmentation in data loading,' +
                         'set to True for Jacobian calculation (default: True)')
parser.add_argument('--load-attention',
                    default="/vxm_attention_2_RISI_two_maps_20220727/Model/2022-07-27_21-17/last_checkpoint_0020.pt",
                    help='optional model file to initialize with')
parser.add_argument('--load-vxm',
                    # vxm model
                    default="",
                    help='optional model file to initialize with')
parser.add_argument('--load-unet',
                    # vxm model
                    default="",
                    help='optional model file to initialize with')
parser.add_argument('--image-loss', default='mse', # 'mse', 'ncc'
                    help='image reconstruction loss - can be mse or ncc (default: ncc)')
parser.add_argument('--model-loss', default='ce',
                    help='model loss - for scan temporal order (default: ce)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.85,
                    help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()


if args.curr_machine == "lambda":
    root_dir = "/data/mengjin"
    vxm_dir = "/data/mengjin/Longi_T1_Aim2"

args.vxm_dir = vxm_dir + args.vxm_dir
args.root_dir = root_dir + args.root_dir
args.load_vxm = vxm_dir + args.load_vxm

if args.load_attention:
    args.load_attention = vxm_dir + args.load_attention


def _create_optimizer(model):
    learning_rate = args.lr
    weight_decay = args.weight_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def _create_lr_scheduler(optimizer):
    lr_config = {"name": args.lr_scheduler,
                 "milestones": [30, 40, 50],
                 "gamma": 0.2 }

    if lr_config is None:
        # use ReduceLROnPlateau as a default scheduler
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)
    else:
        class_name = lr_config.pop('name')
        m = importlib.import_module('torch.optim.lr_scheduler')
        clazz = getattr(m, class_name)
        # add optimizer to the config
        lr_config['optimizer'] = optimizer
        return clazz(**lr_config)


def log_epoch_lr(writer, epoch, optimizer, num_iterations):
    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('z_learning_rate', lr, num_iterations)
    writer.add_scalar('z_epoch', epoch, num_iterations)


def log_stats(writer, phase, loss, loss_list, num_iterations):
    tag_value = {
        f'{phase}_loss_avg': loss.avg
    }
    for n, subloss in enumerate(loss_list):
        tag_value[f'{phase}_subloss_{n}'] = subloss.avg
    for tag, value in tag_value.items():
        writer.add_scalar(tag, value, num_iterations)

############################## Start Testing #############################
bidir = args.bidir
print("Test on", args.load_attention)
# These must match the saved model

# load and prepare test data
if args.test_img_list:
    test_img_list = args.test_img_list
else:
    test = args.test_grp
    test_group = [test]
    test_stages = args.test_stages.strip('[]').split(', ')
    test_img_list = args.root_dir + "/" + test + "_list.csv"
    if args.registration_type == "intra-subject":
        print("Intra-subject registration")
        vxm.py.utils.get_file_list_intra_subject(args.root_dir, args.data_dir, test_group,
                                                 test_img_list, test_stages)
    elif args.registration_type == "inter-subject":
        print("Inter-subject registration")
        vxm.py.utils.get_file_list_inter_subject(args.root_dir, args.data_dir, test_group,
                                                 test_img_list, test_stages)
    else:
        print("Inter-subject registration or intra-subject registration? Please double check!")

test_files = vxm.py.utils.read_file_list(test_img_list,
                                         prefix=args.img_prefix,
                                         suffix=args.img_suffix)
assert len(test_files) > 0, 'Could not find any test data.'
print("Number of test images:", len(test_files))
steps_per_test_epoch = math.ceil(len(test_files)/args.test_batch_size)
# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

# extract shape from sampled input
inshape = (48, 80, 64)

if args.atlas: # code to be updated
    # scan-to-atlas generator
    atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol',
                                      add_batch_axis=True, add_feat_axis=add_feat_axis)
    test_generator = vxm.generators.scan_to_atlas(test_files,
                                                  atlas,
                                                  batch_size=args.test_batch_size,
                                                  bidir=args.bidir,
                                                  add_feat_axis=add_feat_axis,
                                                  segs=args.segs,
                                                  output_size=inshape,
                                                  ret_affine=args.ret_affine)
else:
    # scan-to-scan generator, intra-subject registration
    if args.registration_type == "intra-subject":
        test_augment = ['normalize', 'crop']
        test_generator = vxm.generators.intra_scan_pair(
            test_files,
            test_augment,
            mode="test",
            batch_size=args.test_batch_size,
            bidir=args.bidir,
            add_feat_axis=add_feat_axis,
            segs=args.segs,
            output_size=inshape,
            ret_affine=args.ret_affine)


# prepare model folder


# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.test_batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]
print("load model")
if args.load_attention:
    # load the whole model (if specified)
    model = vxm.networks.VxmSingleAttentionRISI.load(args.load_attention, device)
else:
    # otherwise configure new model
    if args.load_vxm:
        # load initial model (if specified)
        VXM_model = vxm.networks.VxmDense.load(args.load_vxm, device)
        for param in VXM_model.parameters():
            param.requires_grad = False
    else:
        # otherwise configure new model
        VXM_model = vxm.networks.VxmDense(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir=bidir,
            int_steps=args.int_steps,
            int_downsize=args.int_downsize
        )

    if args.load_unet:
        UNET_model = vxm.networks.UNet3D_Seg.load(args.load_unet, device)
    else:
        UNET_model = vxm.networks.UNet3D_Seg(out_dim=args.out_dim)

    model = vxm.networks.VxmSingleAttentionRISI(inshape=inshape,
                                                inmodel=VXM_model,
                                                unet_model=UNET_model,
                                                hyper_a=args.hyper_a_sigmoid,
                                                risi_categories = args.risi_categories,
                                                num_attn_maps = args.num_attn_maps)

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for test and send to device
model.to(device)
model.eval()

# prepare image loss
if args.image_loss == 'ncc':
    radius = [args.radius] * len(inshape)
    image_loss_func = vxm.losses.NCC(radius).loss # best practice is radius = 2 or window size = 5 # win=5
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss

if args.model_loss == 'ce':
    image_loss_func = torch.nn.CrossEntropyLoss().cuda(args.gpu)
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional
if bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]

# prepare deformation loss
losses = [image_loss_func, image_loss_func]
print("loss is:", losses)

# log parameters into comet

model_dir = args.load_attention[:-3] + test + "_combined_atrophy"
try:
    shutil.rmtree(model_dir)
except OSError as e:
    print ("Error: %s - %s." % (e.filename, e.strerror))

os.makedirs(model_dir, exist_ok=True)

# create logger
logger = get_logger('MTLVXM')
logger.info(f'Start test on {device}.')


 # initialize log variables
test_epoch_loss = []

for n in range(len(losses)):
    test_epoch_loss.append(RunningAverage())
test_epoch_total_loss = RunningAverage()

# test
epoch_step_time = []

# reset log variables
for n in range(len(losses)):
    test_epoch_loss[n].reset()
test_epoch_total_loss.reset()

# test
csv_name = model_dir + "/" + args.registration_type + "_" + test + ".csv"
if os.path.exists(csv_name):
    os.remove(csv_name)
with torch.no_grad():
    with open(csv_name, 'a', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        if args.registration_type == "intra-subject":
            wr.writerow(["moving_fname", "fixed_fname",
                         "subjectID", "side", "stage", "bl_time", "fu_time",
                         "date_diff_true", "date_diff_label",
                         "date_diff_pred1", "date_diff_pred2",
                         "date_diff_acc1", "date_diff_acc2",
                         "vol_diff_ratio_hippo", "vol_diff_ratio_sulcus",
                         "bl_hippo_volume", "fu_hippo_volume",
                         "bl_sulcus_volume", "fu_sulcus_volume",
                         "hippo_mask_vol", "sulcus_mask_vol",
                         ])

        for step in range(steps_per_test_epoch):

            step_start_time = time.time()

            # generate inputs (and true outputs) and convert them to tensors
            if args.segs:
                if args.ret_affine:
                    inputs, y_true, imgs_segs, affine, orig_shape, crop_pos, \
                    filename, date_diff_true, date_diff_label = next(test_generator)
                else:
                    inputs, y_true, imgs_segs, filename, date_diff_true, date_diff_label = next(test_generator)
                imgs_segs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in imgs_segs]
            else:
                if args.ret_affine:
                    inputs, y_true, affine, orig_shape, crop_pos, filename, date_diff_true, date_diff_label = next(
                        test_generator)
                else:
                    inputs, y_true, filename, date_diff_true, date_diff_label = next(test_generator)

            inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
            y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
            date_diff_label = torch.tensor(date_diff_label).to(device).float()

            bl, fu = inputs
            x = torch.cat([bl, fu], dim=1)
            out_seg = model.UNet3D_Seg(x)
            out_seg, moved_img, warp, volume_diff, volume, warped_volume, jdet = model.attention(bl, fu,
                                                                                                 out_seg,
                                                                                                 registration=True)

            volume_diff_divide_hippo, volume_diff_divide_sulcus = volume_diff
            hippo_volume, sulcus_volume = volume
            warped_hippo_volume, warped_sulcus_volume = warped_volume

            volume_diff_divide_hippo = 50 * torch.log(volume_diff_divide_hippo)
            volume_diff_divide_hippo = torch.stack((volume_diff_divide_hippo, - volume_diff_divide_hippo), dim=1)

            volume_diff_divide_sulcus = 50 * torch.log(volume_diff_divide_sulcus)
            volume_diff_divide_sulcus = torch.stack((- volume_diff_divide_sulcus, volume_diff_divide_sulcus), dim=1)

            vol_diff_label = [volume_diff_divide_hippo, volume_diff_divide_sulcus]
            attention_pred = torch.argmax(out_seg, dim=1)

            hippo_mask = out_seg[:, 1:2, :, :, :] # 0 bg, 1 hippo
            sulcus_mask = out_seg[:, 2:3, :, :, :] # 0 bg, 2 sulcus

            hippo_mask = torch.sum(hippo_mask, dim=(1, 2, 3, 4))
            sulcus_mask = torch.sum(sulcus_mask, dim=(1, 2, 3, 4))

            # calculate total loss
            loss = 0
            for i in range(len(date_diff_label)):
                for n, loss_function in enumerate(losses):

                    moving_name = filename[i].split("/")[-1]
                    moving_seg_name = model_dir + "/" + moving_name.replace('blmptrim_', "blmptrim_seg", 1).replace(
                        '_to_hw', '', 1)

                    curr_loss = loss_function(vol_diff_label[n], date_diff_label.long())
                    # loss_list.append(curr_loss.item())
                    test_epoch_loss[n].update(curr_loss.item(), args.test_batch_size)
                    loss += curr_loss

                date_diff_pred1 = torch.argmax(vol_diff_label[0], dim=1)
                date_diff_pred2 = torch.argmax(vol_diff_label[1], dim=1)
                # acc = torch.mean((date_diff_pred == date_diff_label).double())

                if args.registration_type == "intra-subject":
                    subjectID = moving_name[:10]
                    stage = filename[i].split("/")[-2]
                    side = moving_name.split("_")[-3]
                    bl_time = moving_name[11:21]
                    fu_time = moving_name[22:32]

                    moving_name = model_dir + "/" + moving_name
                    fixed_name = moving_name.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1)
                    moved_name = moving_name.replace('.nii.gz', "_moved.nii.gz", 1)
                    warp_name = moving_name.replace('.nii.gz', "_warp.nii.gz", 1)

                    # gather information of subjects
                    row = [moving_name, fixed_name,
                           subjectID, side, stage, bl_time, fu_time,
                           date_diff_true[i], date_diff_label[i].item(),
                           date_diff_pred1[i].item(), date_diff_pred2[i].item(),
                           int(date_diff_label[i].item() == date_diff_pred1[i].item()), int(date_diff_label[i].item() == date_diff_pred2[i].item()),
                           vol_diff_label[0][i][0].item(), vol_diff_label[1][i][0].item(),
                           hippo_volume[i].item(), warped_hippo_volume[i].item(),
                           sulcus_volume[i].item(), warped_sulcus_volume[i].item(),
                           hippo_mask[i].item(), sulcus_mask[i].item(),
                           ]
                    wr.writerow(row)

                # save example images, sample the last images of each batch
                if args.save_image:
                    if not args.ret_affine:
                        # save moving image
                        vxm.py.utils.save_volfile(inputs[0][i, 0, :].detach().cpu().numpy().squeeze(), moving_name)
                        # save fixed image
                        vxm.py.utils.save_volfile(inputs[1][i, 0, :].detach().cpu().numpy().squeeze(), fixed_name)

                        # save attention image
                        moving_attention_name = moving_name.replace('.nii.gz', "_attention.nii.gz", 1)
                        vxm.py.utils.save_volfile(attention_pred[i, :].detach().cpu().numpy().squeeze().astype(float), moving_attention_name)

                        # save jacobian determinant
                        moving_jdet_name = moving_name.replace('.nii.gz', "_jdet.nii.gz", 1)
                        vxm.py.utils.save_volfile(jdet[i, 0, :].detach().cpu().numpy().squeeze(), moving_jdet_name)

                        # save seg image before registration
                        vxm.py.utils.save_volfile(imgs_segs[0][i].detach().cpu().numpy().squeeze(), moving_seg_name)

                        # save moved image
                        vxm.py.utils.save_volfile(moved_img[i, 0, :].detach().cpu().numpy().squeeze(), moved_name)

                        # save warp image
                        vxm.py.utils.save_volfile(warp[i, :].detach().cpu().numpy().squeeze(), warp_name)
                    else:
                        affine_trans = [affine[i, :], orig_shape[i, :], crop_pos[i, :]]
                        
                        shutil.copyfile(filename[i], moving_name)  # directly copy original file
                        # save attention image
                        moving_attention_name = moving_name.replace('.nii.gz', "_attention.nii.gz", 1)
                        vxm.py.utils.save_volfile(attention_pred[i, :].detach().cpu().numpy().squeeze().astype(float),
                                                  moving_attention_name, affine_trans)
                        moving_pos_mask = moving_name.replace('.nii.gz', "_position_mask.nii.gz", 1)
                        vxm.py.utils.save_volfile(np.ones(inshape), moving_pos_mask, affine_trans)
                        # save jacobian determinant
                        moving_jdet_name = moving_name.replace('.nii.gz', "_jdet.nii.gz", 1)
                        vxm.py.utils.save_volfile(jdet[i, 0, :].detach().cpu().numpy().squeeze(),
                                                  moving_jdet_name, affine_trans)

                        # save seg image before registration
                        vxm.py.utils.save_volfile(imgs_segs[0][i].detach().cpu().numpy().squeeze(),
                                                  moving_seg_name, affine_trans)

                        # save moved image
                        vxm.py.utils.save_volfile(moved_img[i, 0, :].detach().cpu().numpy().squeeze(),
                                                  moved_name, affine_trans)


            loss /= len(date_diff_label)
            test_epoch_total_loss.update(loss.item(), len(date_diff_label))

            # get compute time
            epoch_step_time.append(time.time() - step_start_time)

            # log test information
            interations = steps_per_test_epoch + step
            if interations % args.log_after_iters == 0:
                loss_text = ""
                for n in range(len(losses)):
                    loss_text = loss_text + f'{test_epoch_loss[n].avg},'
                logger.info(
                    f'Test stats. Step: {step}/{steps_per_test_epoch} Loss: {test_epoch_total_loss.avg} (' + loss_text[:-1] + ')')


