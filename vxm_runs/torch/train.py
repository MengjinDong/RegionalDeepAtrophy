#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.
"""


from comet_ml import Experiment
import os
import sys
import random
import argparse
import time
import math
import shutil
import numpy as np
import importlib
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from functools import partial
import pickle
# from monai.networks.nets import UNet
# from monai.networks.nets import SwinUNETR

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import vxm_model as vxm  # nopep8
from vxm_model.py.utils import get_logger, get_tensorboard_formatter, RunningAverage
from vxm_model.torch import layers
import matplotlib



# import comet_ml at the top of your file

# Create an experiment with your api key
experiment = Experiment(
    api_key="8pO1ROZQ8g3OqjqYpnZX7DCdR",
    project_name="voxelmorph",
    workspace="wuxiaoxiao",
)

# parse the commandline
parser = argparse.ArgumentParser()

############################# data organization parameters ############################
parser.add_argument('--curr-machine', default='lambda',
                    help='specify on which machine the model is running')
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--vxm-dir', default='/vxm_attention_2_RISI_two_maps_20220727',
                    help='model output directory (default: Model)')
parser.add_argument('--train-img-list', default=None, help='line-seperated list of training files')
parser.add_argument('--val-img-list', default=None, help='line-seperated list of validation files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', default='Model',
                    help='model output directory (default: Model)')
parser.add_argument('--log-dir', default='log',
                    help='log saving directory (default: log)')
parser.add_argument('--data-dir', default='final_paper',
                    help='the data split used for this experiment (default: final_paper)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--root-dir', metavar='DIR',
                    default="/Longi_T1_2GO_QC",
                    help='path to dataset')
parser.add_argument('--train-stages', default="[0, 1, 3, 5]",  # "[0, 1, 3, 5]"
                    help='Input stages to train the model')
parser.add_argument('--val-stages', default="[0]",
                    help='Input stages to evaluate the model')
parser.add_argument('--registration-type', default="intra-subject",
                    help=' "intra-subject" registration or "inter-subject" registration ')
parser.add_argument('--save-image', default=True,
                    help='Whether to save tested images to .nii.gz file formats')
parser.add_argument('--save-image-freq', default=200,
                    help='Whether to save tested images to .nii.gz file formats')


############################# training parameters ############################
parser.add_argument('--n-classes', type=int, default=2, help='number of output classes for unet segmentation')
                                   # default = 2: [nothing, shrink, expand]
parser.add_argument('--in-dim', type=int, default=2, help='default dimension for unet seg input')
                    # 2: input 2 images for [bl1, bl2] for a single attention map (num_attn_maps=1),
                    #                    or [bl, fu] for two attention maps,each attention map (num_attn_maps=2)
parser.add_argument('--out-dim', type=int, default=3, help='default dimension for unet seg output')
                    # 3: output 3 channels of [shrink, bg, expand] voxels
parser.add_argument('--num-attn-maps', type=int, default=2, help='default number of attention maps for two pairs')
                    # 2: generate separate attention maps for each input pair.
parser.add_argument('--hyper-a-sigmoid', type=float, default=1, help='default hyper parameter for sigmoid for RISI loss') # a = 1, 0.5, 0.25
parser.add_argument('--risi-categories', type=int, default=4, help='default number of label categories for RISI loss')
parser.add_argument('--risi-loss-weight', type=float, default=100, help='default weight of risi loss')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-train-epoch', type=int, default=500, help='how many steps per train epoch')
parser.add_argument('--steps-per-val-epoch', type=int, default=20,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--log-after-iters', type=int, default=20, help='')
parser.add_argument('--validate-after-epochs', type=int, default=1, help='validate after # epochs')
parser.add_argument('--save-model-per-epochs', type=int, default=1, help='save mocel after each # epochs')
parser.add_argument('--load-model',
                    default="",
                    help='optional model file to initialize with')
parser.add_argument('--load-vxm',
                    # vxm model
                    # default="",
                    default="/voxelmorph_20210907/Model/2021-12-02_00-39/best_checkpoint.pt", 
                    help='optional model file to initialize with')
parser.add_argument('--load-attention',
                    # unet model
                    # default="",
                    default="/vxm_attention_2_20220223/Model/2022-04-29_17-14/last_checkpoint_0040.pt",
                    help='optional model file to initialize with')
parser.add_argument('--load-unet',
                    # unet model
                    default="",
                    help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 1e-4)') # was 5e-4
parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay for learning rate (default: 1e-4)') # was 1e-4
parser.add_argument('--lr-scheduler', type=str, default="MultiStepLR", help='learning rate scheduler') # was 1e-4
parser.add_argument('--milestones', default=[30, 40, 50], help='milestones for learning rate scheduler')
parser.add_argument('--gamma', type=float, default=0.2, help='gamma for learning rate scheduler')
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
parser.add_argument('--segs', default=True,
                    help='Whether to include segmentation in data loading,' +
                         'set to True for Jacobian calculation (default: True)')
parser.add_argument('--risi-loss', default='ce',
                    help='risi loss - for relative interscan interval (default: ce)')
parser.add_argument('--sto-loss', default='ce',
                    help='sto loss - for scan temporal order (default: ce)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.85,  # 0.85 for intrasubject from hypermorph; 0.15 for intersubject.
                    help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()

if args.curr_machine == "lambda":
    root_dir = "/data/mengjin"
    vxm_dir = "/data/mengjin/Longi_T1"
    if args.load_attention:
        args.train_batch_size = 20
        args.val_batch_size = 20
    elif args.load_unet:
        args.train_batch_size = 20
        args.val_batch_size = 20
    elif "monai" in sys.modules:
        args.train_batch_size = 14  # unetr
        args.val_batch_size = 14
    else:
        args.train_batch_size = 20
        args.val_batch_size = 20

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


def log_stats(writer, phase, total_loss, loss_list, acc, num_iterations):
    tag_value = {
        f'{phase}_loss_avg': total_loss.avg,
        f'{phase}_acc_hippo_sto1': acc[0].avg,
        f'{phase}_acc_sulcus_sto1': acc[1].avg,
        f'{phase}_acc_hippo_sto2': acc[2].avg,
        f'{phase}_acc_sulcus_sto2': acc[3].avg,
        f'{phase}_acc_hippo_risi': acc[4].avg,
        f'{phase}_acc_sulcus_risi': acc[5].avg,
    }
    for n, subloss in enumerate(loss_list):
        tag_value[f'{phase}_subloss_{n}'] = subloss.avg

    for tag, value in tag_value.items():
        writer.add_scalar(tag, value, num_iterations)

def log_images_subtle(writer, sample, out_seg1, num_iterations, out_seg2=None):
    # for images, log images to tensorboardX
    nums_display = 4 # was 10
    inputs_map = {
        'moving1':          sample["imgs_bl1"][0:nums_display, 0, :], # input bl and fu at the same time
        'fixed1':           sample["imgs_fu1"][0:nums_display, 0, :],
        'moving2':          sample["imgs_bl2"][0:nums_display, 0, :],  # input bl and fu at the same time
        'fixed2':           sample["imgs_fu2"][0:nums_display, 0, :],
        'seg_bg1':          out_seg1[0:nums_display, 0, :],
        'seg_hippo1':       out_seg1[0:nums_display, 1, :],
        'seg_sulcus1':      out_seg1[0:nums_display, 2, :],
    }
    if out_seg2 != None:
        inputs_map['seg_bg2'] = out_seg2[0:nums_display, 0, :],
        inputs_map['seg_hippo2'] = out_seg2[0:nums_display, 1, :],
        inputs_map['seg_sulcus2'] = out_seg2[0:nums_display, 2, :],
    if len(sample["imgs_seg1"].shape) > 0:
        inputs_map['segs_gt1'] = sample["imgs_seg1"][0:nums_display, 0, :]
        inputs_map['segs_gt2'] = sample["imgs_seg2"][0:nums_display, 0, :]
    img_sources = {}
    for name, batch in inputs_map.items():
        if isinstance(batch, list) or isinstance(batch, tuple):
            for i, b in enumerate(batch):
                img_sources[f'{name}{i}'] = b.data.cpu().numpy()
        else:
            img_sources[name] = batch.data.cpu().numpy()
    for name, batch in img_sources.items():
        for tag, image in tensorboard_formatter(name, batch, normalize=True): # image_names
            writer.add_image(tag, image, num_iterations, dataformats='CHW')


bidir = args.bidir

# load and prepare training data
if args.train_img_list and args.val_img_list:
    train_img_list = args.train_img_list
    val_img_list = args.val_img_list
else:
    train_group = ["train"]
    val_group = ["val"]
    train_stages = args.train_stages.strip('[]').split(', ')
    val_stages = args.val_stages.strip('[]').split(', ')
    train_img_list = args.root_dir + "/" + args.data_dir + "/train_list.csv"
    val_img_list = args.root_dir + "/" + args.data_dir + "/val_list.csv"
    if args.registration_type == "intra-subject":
        print("Intra-subject registration")
        # output saved to train_image_list
        vxm.py.utils.get_file_list_intra_subject_RISI(args.root_dir, args.data_dir, train_group,
                                                      train_img_list, train_stages)
        vxm.py.utils.get_file_list_intra_subject_RISI(args.root_dir, args.data_dir, val_group,
                                                      val_img_list, val_stages)
    elif args.registration_type == "inter-subject":
        print("Inter-subject registration")
        vxm.py.utils.get_file_list_inter_subject(args.root_dir, args.data_dir, train_group,
                                                 train_img_list, train_stages)
        vxm.py.utils.get_file_list_inter_subject(args.root_dir, args.data_dir, val_group,
                                                 val_img_list, val_stages)
    else:
        print("Inter-subject registration or intra-subject registration? Please double check!")

train_files = vxm.py.utils.read_file_list_RISI(train_img_list)
assert len(train_files) > 0, 'Could not find any training data.'
print("Number of training images:", len(train_files))

val_files = vxm.py.utils.read_file_list_RISI(val_img_list)
assert len(val_files) > 0, 'Could not find any validation data.'
print("Number of validation images:", len(val_files))

# extract shape from sampled input
inshape = (48, 80, 64)
print("inshape is", inshape)

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

if args.atlas:
    # scan-to-atlas generator
    atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol',
                                      add_batch_axis=True, add_feat_axis=add_feat_axis)
    train_generator = vxm.generators.scan_to_atlas(train_files,
                                                   atlas,
                                                   batch_size=args.train_batch_size,
                                                   bidir=args.bidir,
                                                   add_feat_axis=add_feat_axis)
else:
    # scan-to-scan generator, intra-subject registration
    if args.registration_type == "intra-subject":
        train_augment = ['normalize', 'crop', 'rotate', 'flip'] # 'erase'
        val_augment = ['normalize', 'crop']
        train_generator = vxm.generators.intra_scan_pair_RISI(train_files,
                                                              train_augment,
                                                              mode="train",
                                                              batch_size=args.train_batch_size,
                                                              no_warp=True,
                                                              bidir=args.bidir,
                                                              add_feat_axis=add_feat_axis,
                                                              segs=args.segs,
                                                              output_size=inshape,
                                                              num_attn_maps=args.num_attn_maps,
                                                              risi_categories=args.risi_categories)
        val_generator = vxm.generators.intra_scan_pair_RISI(val_files,
                                                            val_augment,
                                                            mode="val",
                                                            batch_size=args.val_batch_size,
                                                            no_warp=True,
                                                            bidir=args.bidir,
                                                            add_feat_axis=add_feat_axis,
                                                            segs=args.segs,
                                                            output_size=inshape,
                                                            num_attn_maps=args.num_attn_maps,
                                                            risi_categories=args.risi_categories)

# prepare model folder


# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.train_batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# voxelmorph architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

print("load model")

if args.load_model:
    # load the whole model (if specified)
    model = vxm.networks.VxmAttentionConsistentSTORISI.load(args.load_model, device)

else:
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

    if args.load_attention:
        UNET_model = vxm.networks.VxmAttention.load(args.load_attention, device).UNet3D_Seg
    elif args.load_unet:
        UNET_model = vxm.networks.UNet3D_Seg.load(args.load_unet, device)
    elif "monai" in sys.modules:
        # if monai package is imported
        UNET_model = SwinUNETR(
            img_size=inshape,
            in_channels=args.in_dim,
            out_channels=args.out_dim,
            feature_size=48,
            use_checkpoint=True,
        )
        weight = torch.load(args.vxm_dir + "/Pretrained/model_swinvit.pt")
        UNET_model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights for attention part!")
    else:
        UNET_model = vxm.networks.UNet3D_Seg(in_dim=args.in_dim, n_classes = args.n_classes, out_dim=args.out_dim)

    model = vxm.networks.VxmAttentionConsistentSTORISI(inshape=inshape,
                                                       inmodel=VXM_model,
                                                       unet_model=UNET_model,
                                                       hyper_a=args.hyper_a_sigmoid,
                                                       risi_categories = args.risi_categories,
                                                       num_attn_maps = args.num_attn_maps)

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for training and send to device
model.to(device)
model.train()

# set optimizer
optimizer = _create_optimizer(model)
# Create learning rate adjustment strategy
lr_scheduler = _create_lr_scheduler(optimizer)

# prepare image loss

if args.sto_loss == 'ce':
    sto_loss_func = torch.nn.CrossEntropyLoss().cuda(args.gpu)
else:
    raise ValueError('STO loss should be "ce", but found "%s"' % args.sto_loss)

if args.risi_loss == 'ce':
    risi_loss_func = torch.nn.CrossEntropyLoss().cuda(args.gpu)
elif args.risi_loss == 'mse':
    risi_loss_func = torch.nn.MSELoss()
else:
    raise ValueError('RISI loss should be "ce" or "mse", but found "%s"' % args.risi_loss)

losses = [sto_loss_func, sto_loss_func, sto_loss_func, sto_loss_func, risi_loss_func, risi_loss_func]

print("losses: ", losses)

# log parameters into comet

curr_time = time.strftime("%Y-%m-%d_%H-%M")
model_name = curr_time + \
             'train_' + args.train_stages.strip('[]').replace(', ', '')

model_dir = args.vxm_dir + "/" + args.model_dir + "/" + curr_time
os.makedirs(model_dir, exist_ok=True)

hyper_params = vars(args)
hyper_params["sto_loss"] = sto_loss_func
hyper_params["risi_loss"] = risi_loss_func
hyper_params["optimizer"] = optimizer
hyper_params["model_name"] = model_name
experiment.log_parameters(hyper_params)

log_name = (args.vxm_dir + "/" + args.log_dir
            + "/" + args.data_dir
            + "/" + curr_time)
writer = SummaryWriter(log_name)
tensorboard_formatter = get_tensorboard_formatter()

# create logger
logger = get_logger('MTLVXM')
logger.info(f'Start training on {device}.')


 # initialize log variables
train_epoch_loss = []
val_epoch_loss = []

train_epoch_total_acc = []
val_epoch_total_acc = []

for n in range(len(losses)):
    train_epoch_loss.append(RunningAverage())
    val_epoch_loss.append(RunningAverage())

    train_epoch_total_acc.append(RunningAverage())
    val_epoch_total_acc.append(RunningAverage())

train_epoch_total_loss = RunningAverage()
val_epoch_total_loss = RunningAverage()

# training loops
best_val_score = 1000000000
for epoch in range(args.initial_epoch, args.epochs):

    epoch_step_time = []

    # reset log variables
    for n in range(len(losses)):
        train_epoch_loss[n].reset()
        train_epoch_total_acc[n].reset()
    train_epoch_total_loss.reset()


    # training
    for step in range(args.steps_per_train_epoch):

        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        sample = next(train_generator)
        if args.segs:
            sample["imgs_seg1"] = torch.from_numpy(sample["imgs_seg1"]).to(device).float().permute(0, 4, 1, 2, 3)
            sample["imgs_seg2"] = torch.from_numpy(sample["imgs_seg2"]).to(device).float().permute(0, 4, 1, 2, 3)

        sample["imgs_bl1"] = torch.from_numpy(sample["imgs_bl1"]).to(device).float().permute(0, 4, 1, 2, 3)
        sample["imgs_bl2"] = torch.from_numpy(sample["imgs_bl2"]).to(device).float().permute(0, 4, 1, 2, 3)
        sample["imgs_fu1"] = torch.from_numpy(sample["imgs_fu1"]).to(device).float().permute(0, 4, 1, 2, 3)
        sample["imgs_fu2"] = torch.from_numpy(sample["imgs_fu2"]).to(device).float().permute(0, 4, 1, 2, 3)

        sample["date_diff_label1"] = torch.tensor(sample["date_diff_label1"]).to(device).float()
        sample["date_diff_label2"] = torch.tensor(sample["date_diff_label2"]).to(device).float() # date_diff_label
        sample["date_diff_ratio"] = torch.tensor(sample["date_diff_ratio"]).to(device).float() # date_diff_ratio

        # run inputs through the model to produce a warped image and flow field
        # prediction: volume difference between image 1 and 2
        if args.num_attn_maps == 1:
            out_seg1, warp1, warp2, vol_diff_label = model(sample["imgs_bl1"],
                                                           sample["imgs_fu1"],
                                                           sample["imgs_bl2"],
                                                           sample["imgs_fu2"],
                                                           registration=True)
            attention_pred1 = torch.argmax(out_seg1, dim=1)
        else: # currently using this branch
            out_seg1, out_seg2, warp1, warp2, vol_diff_label = model(sample["imgs_bl1"],
                                                                     sample["imgs_fu1"],
                                                                     sample["imgs_bl2"],
                                                                     sample["imgs_fu2"],
                                                                     registration=True)
            attention_pred1 = torch.argmax(out_seg1, dim=1)
            attention_pred2 = torch.argmax(out_seg2, dim=1)

        loss = 0
        for n, loss_function in enumerate(losses):

            if n < 2:
                # loss 0: ce for scan temporal order for the first pair
                curr_loss = loss_function(vol_diff_label[n], sample["date_diff_label1"].long())  
                date_diff_pred = torch.argmax(vol_diff_label[n], dim=1)
                train_acc = torch.mean((date_diff_pred == sample["date_diff_label1"]).double())

                train_epoch_loss[n].update(curr_loss.item(), args.train_batch_size)
                train_epoch_total_acc[n].update(train_acc.item(), args.train_batch_size)
                loss += curr_loss

            elif n < 4:
                # loss 1: ce for scan temporal order for the second pair
                curr_loss = loss_function(vol_diff_label[n], sample["date_diff_label2"].long()) 
                date_diff_pred = torch.argmax(vol_diff_label[n], dim=1)
                train_acc = torch.mean((date_diff_pred == sample["date_diff_label2"]).double())

                train_epoch_loss[n].update(curr_loss.item(), args.train_batch_size)
                train_epoch_total_acc[n].update(train_acc.item(), args.train_batch_size)
                loss += curr_loss

            else:
                # loss 2: mse for RISI
                curr_loss = loss_function(vol_diff_label[n], sample["date_diff_ratio"].long())  
                train_epoch_loss[n].update(curr_loss.item(), args.train_batch_size)

                date_diff_ratio_pred = torch.argmax(vol_diff_label[n], dim=1)
                train_acc = torch.mean((date_diff_ratio_pred == sample["date_diff_ratio"]).double())
                train_epoch_total_acc[n].update(train_acc.item(), args.train_batch_size)

                loss += args.risi_loss_weight * curr_loss


        train_epoch_total_loss.update(loss.item(), args.train_batch_size)

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

        # log training information
        iterations = args.steps_per_train_epoch * epoch + step
        if iterations % args.log_after_iters == 0:
            loss_text = ""
            for n in range(len(losses)):
                loss_text = loss_text + f'{train_epoch_loss[n].avg},'
            logger.info(
                f'Training stats. Epoch: {epoch + 1}/{args.epochs}. ' +
                f'Step: {step}/{args.steps_per_train_epoch} ' +
                f'Loss: {train_epoch_total_loss.avg} (' + loss_text[:-1] + ') ' +
                f'Acc hippo1: {train_epoch_total_acc[0].avg} ' +
                f'Acc sulcus1: {train_epoch_total_acc[1].avg} ' +
                f'Acc hippo2: {train_epoch_total_acc[2].avg} ' +
                f'Acc sulcus2: {train_epoch_total_acc[3].avg} ' +
                f'Acc hippo RISI: {train_epoch_total_acc[4].avg} ' +
                f'Acc sulcus RISI: {train_epoch_total_acc[5].avg} ' +
                f'Loss ratio_hippo: {train_epoch_loss[4].avg} ' +
                f'Loss ratio_sulcus: {train_epoch_loss[5].avg} ')
            log_stats(writer, 'train', train_epoch_total_loss, train_epoch_loss, train_epoch_total_acc, iterations)

            if args.num_attn_maps == 1:
                log_images_subtle(writer, sample, out_seg1, iterations)
            else:
                log_images_subtle(writer, sample, out_seg1, iterations, out_seg2)

        if args.save_image and iterations % args.save_image_freq == 0: # save_image_freq = 100
            # save a batch of (10) images at a time
            for i in range(len(sample["date_diff_label1"])//2):

                moving_name1 = sample["img_names1"][i].split("/")[-1]
                moving_seg_name1 = model_dir + "/" + moving_name1.replace('blmptrim_', "blmptrim_seg", 1).replace(
                    '_to_hw', '', 1)

                moving_name2 = sample["img_names2"][i].split("/")[-1]
                moving_seg_name2 = model_dir + "/" + moving_name2.replace('blmptrim_', "blmptrim_seg", 1).replace(
                    '_to_hw', '', 1)

                if args.registration_type == "intra-subject":
                    subjectID = moving_name1[:10]
                    stage = sample["img_names1"][i].split("/")[-2]
                    side = moving_name1.split("_")[-3]
                    bl_time = moving_name1[11:21]
                    fu_time = moving_name1[22:32]

                    moving_name1 = model_dir + "/" + str(iterations) + "__" + moving_name1
                    fixed_name1 = moving_name1.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1)
                    moved_name1 = moving_name1.replace('.nii.gz', "_moved.nii.gz", 1)
                    warp_name1 = moving_name1.replace('.nii.gz', "_warp.nii.gz", 1)

                    moving_name2 = model_dir + "/" + str(iterations) + "__" + moving_name2
                    fixed_name2 = moving_name2.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1)
                    moved_name2 = moving_name2.replace('.nii.gz', "_moved.nii.gz", 1)
                    warp_name2 = moving_name2.replace('.nii.gz', "_warp.nii.gz", 1)
                    # gather information of subjects

                # save moving image
                vxm.py.utils.save_volfile(sample["imgs_bl1"][i, 0, :].detach().cpu().numpy().squeeze(), moving_name1)
                vxm.py.utils.save_volfile(sample["imgs_bl2"][i, 0, :].detach().cpu().numpy().squeeze(), moving_name2)

                # save fixed image
                vxm.py.utils.save_volfile(sample["imgs_fu1"][i, 0, :].detach().cpu().numpy().squeeze(), fixed_name1)
                vxm.py.utils.save_volfile(sample["imgs_fu2"][i, 0, :].detach().cpu().numpy().squeeze(), fixed_name2)

                # save attention image
                moving_attention_name1 = moving_name1.replace('.nii.gz', "_attention.nii.gz", 1)
                vxm.py.utils.save_volfile(attention_pred1[i, :].detach().cpu().numpy().squeeze().astype(float), moving_attention_name1)

                moving_attention_name2 = moving_name2.replace('.nii.gz', "_attention.nii.gz", 1)
                vxm.py.utils.save_volfile(attention_pred2[i, :].detach().cpu().numpy().squeeze().astype(float), moving_attention_name2)

                # # save jacobian determinant
 moving_jdet_name)

                # save seg image before registration
                vxm.py.utils.save_volfile(sample["imgs_seg1"][i, 0, :].detach().cpu().numpy().squeeze(), moving_seg_name1)
                vxm.py.utils.save_volfile(sample["imgs_seg2"][i, 0, :].detach().cpu().numpy().squeeze(), moving_seg_name2)

                # save warp image
                vxm.py.utils.save_volfile(warp1[i, :].detach().cpu().permute(1, 2, 3, 0).numpy().squeeze(), warp_name1)
                vxm.py.utils.save_volfile(warp2[i, :].detach().cpu().permute(1, 2, 3, 0).numpy().squeeze(), warp_name2)


    # validation
    if epoch % args.validate_after_epochs == 0:

        logger.info(f'Start performing validation.')

        # reset log variables
        for n in range(len(losses)):
            val_epoch_loss[n].reset()
            val_epoch_total_acc[n].reset()
        val_epoch_total_loss.reset()

        # perform validation
        with torch.no_grad():
            for step in range(args.steps_per_val_epoch):

                # generate inputs (and true outputs) and convert them to tensors
                sample = next(val_generator)
                if args.segs:
                    sample["imgs_seg1"] = torch.from_numpy(sample["imgs_seg1"]).to(device).float().permute(0, 4, 1, 2,
                                                                                                           3)
                    sample["imgs_seg2"] = torch.from_numpy(sample["imgs_seg2"]).to(device).float().permute(0, 4, 1, 2,
                                                                                                           3)

                sample["imgs_bl1"] = torch.from_numpy(sample["imgs_bl1"]).to(device).float().permute(0, 4, 1, 2, 3)
                sample["imgs_bl2"] = torch.from_numpy(sample["imgs_bl2"]).to(device).float().permute(0, 4, 1, 2, 3)
                sample["imgs_fu1"] = torch.from_numpy(sample["imgs_fu1"]).to(device).float().permute(0, 4, 1, 2, 3)
                sample["imgs_fu2"] = torch.from_numpy(sample["imgs_fu2"]).to(device).float().permute(0, 4, 1, 2, 3)

                sample["date_diff_label1"] = torch.tensor(sample["date_diff_label1"]).to(device).float()
                sample["date_diff_label2"] = torch.tensor(sample["date_diff_label2"]).to(
                    device).float()  
                sample["date_diff_ratio"] = torch.tensor(sample["date_diff_ratio"]).to(
                    device).float()

                # run inputs through the model to produce a warped image and flow field
                # prediction: volume difference between image 1 and 2
                if args.num_attn_maps == 1:
                    out_seg1, warp1, warp2, vol_diff_label = model(sample["imgs_bl1"],
                                                                   sample["imgs_fu1"],
                                                                   sample["imgs_bl2"],
                                                                   sample["imgs_fu2"],
                                                                   registration=True)
                    attention_pred1 = torch.argmax(out_seg1, dim=1)
                else:  # currently using this branch
                    out_seg1, out_seg2, warp1, warp2, vol_diff_label = model(sample["imgs_bl1"],
                                                                             sample["imgs_fu1"],
                                                                             sample["imgs_bl2"],
                                                                             sample["imgs_fu2"],
                                                                             registration=True)

                    attention_pred1 = torch.argmax(out_seg1, dim=1)
                    attention_pred2 = torch.argmax(out_seg2, dim=1)

                # calculate total loss
                loss_val = 0
                for n, loss_function in enumerate(losses):

                    if n < 2:
                        # loss 0: ce for scan temporal order for the first pair
                        curr_loss_val = loss_function(vol_diff_label[n], sample[
                            "date_diff_label1"].long())
                        date_diff_pred = torch.argmax(vol_diff_label[n], dim=1)
                        val_acc = torch.mean((date_diff_pred == sample["date_diff_label1"]).double())

                        val_epoch_loss[n].update(curr_loss_val.item(), args.val_batch_size)
                        val_epoch_total_acc[n].update(val_acc.item(), args.val_batch_size)

                    elif n < 4:
                        # loss 1: ce for scan temporal order for the second pair
                        curr_loss_val = loss_function(vol_diff_label[n], sample[
                            "date_diff_label2"].long())
                        date_diff_pred = torch.argmax(vol_diff_label[n], dim=1)
                        val_acc = torch.mean((date_diff_pred == sample["date_diff_label2"]).double())

                        val_epoch_loss[n].update(curr_loss_val.item(), args.val_batch_size)
                        val_epoch_total_acc[n].update(val_acc.item(), args.val_batch_size)

                    else:
                        # loss 2: ce for RISI
                        curr_loss_val = loss_function(vol_diff_label[n], sample[
                            "date_diff_ratio"].long()) 
                        val_epoch_loss[n].update(curr_loss_val.item(), args.val_batch_size)

                        date_diff_ratio_pred = torch.argmax(vol_diff_label[n], dim=1)
                        val_acc = torch.mean((date_diff_ratio_pred == sample["date_diff_ratio"]).double())
                        val_epoch_total_acc[n].update(val_acc.item(), args.val_batch_size)

                    loss_val += curr_loss_val

        val_epoch_total_loss.update(loss_val.item(), args.val_batch_size)

        # log stats
        loss_text = ""
        for n in range(len(losses)):
            loss_text = loss_text + f'{val_epoch_loss[n].avg},'
        logger.info(
            f'Validation stats. Epoch: {epoch + 1}/{args.epochs}. ' +
            f'Loss: {val_epoch_total_loss.avg} (' + loss_text[:-1] + ') ' +
            f'Acc hippo1: {val_epoch_total_acc[0].avg} ' +
            f'Acc sulcus1: {val_epoch_total_acc[1].avg} ' +
            f'Acc hippo2: {val_epoch_total_acc[2].avg} ' +
            f'Acc sulcus2: {val_epoch_total_acc[3].avg} ' +
            f'Acc hippo RISI: {val_epoch_total_acc[4].avg} ' +
            f'Acc sulcus RISI: {val_epoch_total_acc[5].avg} ' +
            f'Loss ratio_hippo: {val_epoch_loss[4].avg} ' +
            f'Loss ratio_sulcus: {val_epoch_loss[5].avg} ')
        log_stats(writer, 'val', val_epoch_total_loss, val_epoch_loss, val_epoch_total_acc,
                  args.steps_per_train_epoch * (epoch + 1))
        log_epoch_lr(writer, epoch, optimizer, args.steps_per_train_epoch * (epoch + 1))

        # save checkpoint
    if epoch % args.save_model_per_epochs == 0:
        last_file_path = os.path.join(model_dir, 'last_checkpoint_%04d.pt' % epoch)
        attention_file_path = os.path.join(model_dir, 'attention_last_checkpoint_%04d.pt' % epoch)
        model.save(last_file_path)
        # model.attention_model.save(attention_file_path)
        if val_epoch_total_loss.avg < best_val_score:
            best_file_path = os.path.join(model_dir, 'best_checkpoint.pt')
            shutil.copyfile(last_file_path, best_file_path)
            best_val_score = val_epoch_total_loss.avg

        # adjust learning rate if necessary after each epoch
    if isinstance(lr_scheduler, ReduceLROnPlateau):
        lr_scheduler.step(val_epoch_total_loss.avg)
    else:
        lr_scheduler.step()

    # print epoch info
    # epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    # time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    # losses_info = ', '.join(['%.4e' % f.avg for f in train_epoch_loss])
    # loss_info = 'loss: %.4e  (%s)' % (train_epoch_total_loss.avg, losses_info)
    # print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

# final model save
model.save(os.path.join(model_dir, model_name + '%04d.pt' % args.epochs))

writer.export_scalars_to_json("./all_scalars.json")
writer.close()
