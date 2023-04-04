import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import matplotlib

#matplotlib.use('TkAgg') # MUST BE CALLED BEFORE IMPORTING plt, or qt4agg

from vxm_model.py.utils import default_unet_features, jacobian_determinant_batch, get_moving_volume, get_moved_volume
from . import layers
from .modelio import LoadableModel, store_config_args


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class UNet3D_Seg(LoadableModel):
    @store_config_args
    def __init__(self, in_dim = 2, n_classes = 1, num_filters = 8, out_dim = 2):
        super(UNet3D_Seg, self).__init__()

        self.in_dim = in_dim
        self.n_classes = n_classes
        self.num_filters = num_filters
        self.out_dim = out_dim + 1
        activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling
        self.down_1 = self.conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = self.max_pooling_3d()
        self.down_2 = self.conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = self.max_pooling_3d()
        self.down_3 = self.conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = self.max_pooling_3d()
        self.down_4 = self.conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = self.max_pooling_3d()

        # Bridge
        self.bridge = self.conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)

        # Up sampling
        self.trans_1 = self.conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_1 = self.conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_2 = self.conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_2 = self.conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_3 = self.conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_3 = self.conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_4 = self.conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_4 = self.conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)

        # Output
        self.out_seg = self.conv_block_3d(self.num_filters, self.num_filters, activation)
        self.out_seg_final =  nn.Conv3d(self.num_filters, n_classes + 1, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1):
        # Down sampling
        down_1 = self.down_1(x1)  # -> [1, 4, 48, 80, 64]
        pool_1 = self.pool_1(down_1)  # -> [1, 4, 24, 40, 32]

        down_2 = self.down_2(pool_1)  # -> [1, 8, 24, 40, 32]
        pool_2 = self.pool_2(down_2)  # -> [1, 8, 12, 20, 16]

        down_3 = self.down_3(pool_2)  # -> [1, 16, 12, 20, 16]
        pool_3 = self.pool_3(down_3)  # -> [1, 16, 6, 10, 8]

        down_4 = self.down_4(pool_3)  # -> [1, 32, 6, 10, 8]
        pool_4 = self.pool_4(down_4)  # -> [1, 32, 3, 5, 4]

        # Bridge
        bridge = self.bridge(pool_4)  # -> [1, 128, 3, 5, 4]

        # Up sampling
        trans_1 = self.trans_1(bridge)  # -> [1, 128, 6, 10, 8] # transcomposition increase the size by 2
        concat_1 = torch.cat([trans_1, down_4], dim=1)  # -> [1, 192, 6, 10, 8]
        up_1 = self.up_1(concat_1)  # -> [1, 64,6, 10, 8]

        trans_2 = self.trans_2(up_1)  # -> [1, 64, 12, 20, 16]
        concat_2 = torch.cat([trans_2, down_3], dim=1)  # -> [1, 96, 12, 20, 16]
        up_2 = self.up_2(concat_2)  # -> [1, 32, 12, 20, 16]

        trans_3 = self.trans_3(up_2)  # -> [1, 32, 24, 40, 32]
        concat_3 = torch.cat([trans_3, down_2], dim=1)  # -> [1, 48, 24, 40, 32]
        up_3 = self.up_3(concat_3)  # -> [1, 16, 24, 40, 32]

        trans_4 = self.trans_4(up_3)  # -> [1, 16, 48, 80, 64]
        concat_4 = torch.cat([trans_4, down_1], dim=1)  # -> [1, 24, 48, 80, 64]
        up_4 = self.up_4(concat_4)  # -> [1, 8, 48, 80, 64]

        out_seg = self.out_seg(up_4)
        out_seg = self.out_seg_final(out_seg)

        return out_seg


    def conv_block_3d(self, in_dim, out_dim, activation):
        return nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_dim),
            activation,)

    def conv_trans_block_3d(self, in_dim, out_dim, activation):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_dim),
            activation,)

    def max_pooling_3d(self):
        return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


    def conv_block_2_3d(self, in_dim, out_dim, activation):
        return nn.Sequential(
            self.conv_block_3d(in_dim, out_dim, activation),
            nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_dim),)


class Unet(nn.Module):
    """
    A unet architecture for voxelmorph. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            # y_source: transformed moving image,
            # pos_flow: integrated positive flow
            # return (y_source, preint_flow), pos_flow
            return y_source, pos_flow


class VxmAttention(LoadableModel):
    @store_config_args
    # def __init__(self, inmodel, unet_model=UNet3D_Seg()):
    def __init__(self, inshape, inmodel, unet_model = UNet3D_Seg()):
        super(VxmAttention, self).__init__()
        self.UNet3D_Seg = unet_model
        self.VxmDense = inmodel # inmodel should be vxm_model
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.transformer = layers.SpatialTransformer(inshape)
        # self.out_dim = out_dim

    def forward(self, source, target, registration=True):

        if torch.sum(torch.isnan(source)):
            print("nan found in source image!")

        if torch.sum(torch.isnan(target)):
            print("nan found in target image!")

        # voxelmorph branch
        moved_img, warp = self.VxmDense(source, target, registration=registration)
        # Q: what would happen to auto differentiation with 2 paths of calculations?
        jdet = jacobian_determinant_batch(warp)

        # attention branch
        x = torch.cat([source, target], dim=1)
        out_seg = self.UNet3D_Seg(x)

        if out_seg.shape[1] == 1:
            # TODO map to [0, 1] range
            out_seg = 1/(1 + torch.exp(-out_seg))

        elif out_seg.shape[1] == 2: # prev1 running this branch
            out_seg = self.softmax(100 * out_seg)
            out_seg = out_seg[:, 1:2, :, :, :] # take the second slice, label of seg
            total_change = jdet * out_seg
            vol_diff_label = torch.sum(total_change, dim=(1, 2, 3, 4)) / (torch.sum(out_seg, dim=(1, 2, 3, 4)) + 1e-7)
            vol_diff_label = 50 * torch.log(vol_diff_label)

            # mimic the atrophy measurement, calculating average jacobian
            vol_diff_label = torch.stack((vol_diff_label, - vol_diff_label), dim=1)

            return [out_seg, moved_img, warp, vol_diff_label, jdet]

        elif out_seg.shape[1] == 3: # currently running this branch
            out_seg = self.softmax(100 * out_seg)
            hippo_mask = out_seg[:, 1:2, :, :, :] # 0 bg, 1 hippo
            sulcus_mask = out_seg[:, 2:3, :, :, :] # 0 bg, 2 sulcus

            #for hippocampus and sulcus: directly warp the mask
            warped_hippo_mask = self.transformer(hippo_mask, warp)

            hippo_volume = get_moving_volume(hippo_mask)
            warped_hippo_volume = get_moving_volume(warped_hippo_mask)

            volume_diff_hippo = warped_hippo_volume / (hippo_volume + 1e-7)

            # # sulcus
            warped_sulcus_mask = self.transformer(sulcus_mask, warp)

            sulcus_volume = get_moving_volume(sulcus_mask)
            warped_sulcus_volume = get_moving_volume(warped_sulcus_mask)

            volume_diff_sulcus = warped_sulcus_volume / (sulcus_volume + 1e-7)

            if torch.sum(torch.isnan(volume_diff_hippo)):
                print("nan found in hippo!")

            if torch.sum(torch.isnan(volume_diff_sulcus)):
                print("nan found in sulcus!")

            return [out_seg, moved_img, warp,
                    [volume_diff_hippo, volume_diff_sulcus],
                    [hippo_volume, sulcus_volume],
                    [warped_hippo_volume, warped_sulcus_volume],
                    jdet]

class VxmAttentionConsistentSTORISI(LoadableModel):
    @store_config_args
    def __init__(self, inshape, inmodel,
                 unet_model = UNet3D_Seg(),
                 num_reg_labels=4,
                 hyper_a=0.5,
                 risi_categories = 4,
                 num_attn_maps = 1):
        super(VxmAttentionConsistentSTORISI, self).__init__()

        self.UNet3D_Seg = unet_model
        self.VxmDense = inmodel

        self.num_reg_labels = num_reg_labels
        self.hyper_a = hyper_a
        self.risi_categories = risi_categories
        self.num_attn_maps = num_attn_maps
        self.fc_h1 = nn.Linear(4, 20)
        self.fc_h2 = nn.Linear(20, 4)
        self.fc_s1 = nn.Linear(4, 20)
        self.fc_s2 = nn.Linear(20, 4)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.transformer = layers.SpatialTransformer(inshape)
        self.sigmoid = nn.Sigmoid()


    # def sigmoid(self, x, a):
    #     return 1 / (1 + torch.exp(-a * x))

    def act_4_0(self, x, y, a):
        sig = torch.max(torch.stack([self.sigmoid((2 * x - y) * a) * self.sigmoid((-2 * x - y) * a),
                                      self.sigmoid((2 * x + y) * a) * self.sigmoid((-2 * x + y) * a)], dim=0), dim=0).values
        return sig

    def act_4_1(self, x, y, a):
        sig = torch.max(
            torch.stack([self.sigmoid((2 * x - y) * a) * self.sigmoid((-x + y) * a),
                         self.sigmoid((-2 * x - y) * a) * self.sigmoid((x + y) * a),
                         self.sigmoid((2 * x + y) * a) * self.sigmoid((-x - y) * a),
                         self.sigmoid((-2 * x + y) * a) * self.sigmoid((x - y) * a)], dim=0), dim=0).values
        return sig

    def act_4_2(self, x, y, a):
        sig = torch.max(
            torch.stack([self.sigmoid((0.5 * x - y) * a) * self.sigmoid((-x + y) * a),
                         self.sigmoid((-0.5 * x - y) * a) * self.sigmoid((x + y) * a),
                         self.sigmoid((0.5 * x + y) * a) * self.sigmoid((-x - y) * a),
                         self.sigmoid((-0.5 * x + y) * a) * self.sigmoid((x - y) * a)], dim=0), dim=0).values
        return sig

    def act_4_3(self, x, y, a):
        sig = torch.max(torch.stack([self.sigmoid((2 * y - x) * a) * self.sigmoid((-2 * y - x) * a),
                                      self.sigmoid((2 * y + x) * a) * self.sigmoid((-2 * y + x) * a)], dim=0), dim=0).values

        return sig

    def act_4_categories(self, x, y, a):
        return torch.stack([self.act_4_0(x, y, a),
                            self.act_4_1(x, y, a),
                            self.act_4_2(x, y, a),
                            self.act_4_3(x, y, a)], dim=1)

    def act_8_0(self, x, y, a):
        return torch.max(torch.stack([self.sigmoid((4 * x - y) * a) * self.sigmoid((-4 * x - y) * a),
                                      self.sigmoid((4 * x + y) * a) * self.sigmoid((-4 * x + y) * a)], dim=0), dim=0).values

    def act_8_1(self, x, y, a):
        return torch.max(
            torch.stack([self.sigmoid((4 * x - y) * a) * self.sigmoid((-2 * x + y) * a),
                         self.sigmoid((-4 * x - y) * a) * self.sigmoid((2 * x + y) * a),
                         self.sigmoid((4 * x + y) * a) * self.sigmoid((-2 * x - y) * a),
                         self.sigmoid((-4 * x + y) * a) * self.sigmoid((2 * x - y) * a)], dim=0), dim=0).values

    def act_8_2(self, x, y, a):
        return torch.max(
            torch.stack([self.sigmoid((2 * x - y) * a) * self.sigmoid((-4 * x + 3 * y) * a),
                         self.sigmoid((-2 * x - y) * a) * self.sigmoid((4 * x + 3 * y) * a),
                         self.sigmoid((2 * x + y) * a) * self.sigmoid((-4 * x - 3 * y) * a),
                         self.sigmoid((-2 * x + y) * a) * self.sigmoid((4 * x - 3 * y) * a)], dim=0), dim=0).values

    def act_8_3(self, x, y, a):
        return torch.max(
            torch.stack([self.sigmoid((4 * x - 3 * y) * a) * self.sigmoid((-x + y) * a),
                         self.sigmoid((-4 * x - 3 * y) * a) * self.sigmoid((x + y) * a),
                         self.sigmoid((4 * x + 3 * y) * a) * self.sigmoid((-x - y) * a),
                         self.sigmoid((-4 * x + 3 * y) * a) * self.sigmoid((x - y) * a)], dim=0), dim=0).values

    def act_8_4(self, x, y, a):
        return torch.max(
            torch.stack([self.sigmoid((3 * x - 4 * y) * a) * self.sigmoid((-x + y) * a),
                         self.sigmoid((-3 * x - 4 * y) * a) * self.sigmoid((x + y) * a),
                         self.sigmoid((3 * x + 4 * y) * a) * self.sigmoid((-x - y) * a),
                         self.sigmoid((-3 * x + 4 * y) * a) * self.sigmoid((x - y) * a)], dim=0), dim=0).values

    def act_8_5(self, x, y, a):
        return torch.max(
            torch.stack([self.sigmoid((x - 2 * y) * a) * self.sigmoid((-3 * x + 4 * y) * a),
                         self.sigmoid((-x - 2 * y) * a) * self.sigmoid((3 * x + 4 * y) * a),
                         self.sigmoid((x + 2 * y) * a) * self.sigmoid((-3 * x - 4 * y) * a),
                         self.sigmoid((-x + 2 * y) * a) * self.sigmoid((3 * x - 4 * y) * a)], dim=0), dim=0).values

    def act_8_6(self, x, y, a):
        return torch.max(
            torch.stack([self.sigmoid((x - 4 * y) * a) * self.sigmoid((-x + 2 * y) * a),
                         self.sigmoid((-x - 4 * y) * a) * self.sigmoid((x + 2 * y) * a),
                         self.sigmoid((x + 4 * y) * a) * self.sigmoid((-x - 2 * y) * a),
                         self.sigmoid((-x + 4 * y) * a) * self.sigmoid((x - 2 * y) * a)], dim=0), dim=0).values

    def act_8_7(self, x, y, a):
        return torch.max(torch.stack([self.sigmoid((4 * y - x) * a) * self.sigmoid((-4 * y - x) * a),
                                      self.sigmoid((4 * y + x) * a) * self.sigmoid((-4 * y + x) * a)], dim=0), dim=0).values

    def act_8_categories(self, x, y, a):
        return torch.stack([self.act_8_0(x, y, a),
                            self.act_8_1(x, y, a),
                            self.act_8_2(x, y, a),
                            self.act_8_3(x, y, a),
                            self.act_8_4(x, y, a),
                            self.act_8_5(x, y, a),
                            self.act_8_6(x, y, a),
                            self.act_8_7(x, y, a)], dim=1)

    def attention(self, source, target, out_seg, registration=True):

        # voxelmorph branch
        moved_img, warp = self.VxmDense(source, target, registration=registration)
        jdet = jacobian_determinant_batch(warp)

        if torch.sum(torch.isnan(out_seg)):
            print("nan found in out_seg!")

        out_seg = self.softmax(100 * out_seg)
        hippo_mask = out_seg[:, 1:2, :, :, :]  # 0 bg, 1 hippo
        sulcus_mask = out_seg[:, 2:3, :, :, :]  # 0 bg, 2 sulcus

        # for hippocampus and sulcus: directly warp the mask
        warped_hippo_mask = self.transformer(hippo_mask, warp)

        hippo_volume = get_moving_volume(hippo_mask)
        warped_hippo_volume = get_moving_volume(warped_hippo_mask)
        # # sulcus
        warped_sulcus_mask = self.transformer(sulcus_mask, warp)

        sulcus_volume = get_moving_volume(sulcus_mask)
        warped_sulcus_volume = get_moving_volume(warped_sulcus_mask)

        return [out_seg, moved_img, warp,
                [hippo_volume,        sulcus_volume],
                [warped_hippo_volume, warped_sulcus_volume],
                jdet
               ]


    def forward(self, bl1, fu1, bl2, fu2, registration=True):
        # attention branch
        if self.num_attn_maps == 1:
            x = torch.cat([bl1, bl2], dim=1)
            out_seg1 = self.UNet3D_Seg(x)
            out_seg2 = out_seg1
        elif self.num_attn_maps == 2:
            x1 = torch.cat([bl1, fu1], dim=1)
            out_seg1 = self.UNet3D_Seg(x1)
            x2 = torch.cat([bl2, fu2], dim=1)
            out_seg2 = self.UNet3D_Seg(x2)

        out_seg1, moved_img1, warp1, volume1, warped_volume1, jdet1 = self.attention(bl1, fu1, out_seg1,
                                                                                                   registration=registration)

        out_seg2, moved_img2, warp2, volume2, warped_volume2, jdet2 = self.attention(bl2, fu2, out_seg2,
                                                                                                   registration=registration)

        hippo_volume1, sulcus_volume1 = volume1
        hippo_volume2, sulcus_volume2 = volume2

        warped_hippo_volume1, warped_sulcus_volume1 = warped_volume1
        warped_hippo_volume2, warped_sulcus_volume2 = warped_volume2

        volume_diff_subtract_hippo1 = (warped_hippo_volume1 - hippo_volume1)
        volume_diff_subtract_hippo2 = (warped_hippo_volume2 - hippo_volume2)

        volume_diff_subtract_sulcus1 = (warped_sulcus_volume1 - sulcus_volume1)
        volume_diff_subtract_sulcus2 = (warped_sulcus_volume2 - sulcus_volume2)

        # To calculate volume_diff_ratio_sulcus and volume_diff_ratio_hippo
        # Use volume_diff_hippo1 and volume_diff_sulcus1 to calculate

        if  self.risi_categories == 4:
            volume_diff_ratio_hippo = self.act_4_categories(volume_diff_subtract_hippo1, volume_diff_subtract_hippo2, self.hyper_a)
            volume_diff_ratio_sulcus = self.act_4_categories(volume_diff_subtract_sulcus1, volume_diff_subtract_sulcus2, self.hyper_a)
        elif self.risi_categories == 8:
            volume_diff_ratio_hippo = self.act_8_categories(volume_diff_subtract_hippo1, volume_diff_subtract_hippo2, self.hyper_a)
            volume_diff_ratio_sulcus = self.act_8_categories(volume_diff_subtract_sulcus1, volume_diff_subtract_sulcus2, self.hyper_a)


        # To calculate volume change score for STO loss
        # mimic the atrophy measurement, calculating average Jacobian
        volume_diff_subtract_hippo1 = torch.stack((volume_diff_subtract_hippo1, - volume_diff_subtract_hippo1), dim=1)

        volume_diff_subtract_sulcus1 = torch.stack((- volume_diff_subtract_sulcus1, volume_diff_subtract_sulcus1), dim=1)

        # mimic the atrophy measurement, calculating average Jacobian
        volume_diff_subtract_hippo2 = torch.stack((volume_diff_subtract_hippo2, - volume_diff_subtract_hippo2), dim=1)

        volume_diff_subtract_sulcus2 = torch.stack((- volume_diff_subtract_sulcus2, volume_diff_subtract_sulcus2), dim=1)
        # return attention maps for both pairs
        # return STO loss for the first pair
        # return STO loss for the second pair
        # return RISI loss
        if self.num_attn_maps == 1:
            return out_seg1, warp1, warp2, \
                   [volume_diff_subtract_hippo1, volume_diff_subtract_sulcus1, \
                    volume_diff_subtract_hippo2, volume_diff_subtract_sulcus2, \
                    volume_diff_ratio_hippo, volume_diff_ratio_sulcus]
        else:
            return out_seg1, out_seg2, warp1, warp2, \
                   [volume_diff_subtract_hippo1, volume_diff_subtract_sulcus1, \
                    volume_diff_subtract_hippo2, volume_diff_subtract_sulcus2, \
                    volume_diff_ratio_hippo, volume_diff_ratio_sulcus]
