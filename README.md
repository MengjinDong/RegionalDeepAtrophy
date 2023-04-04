# Regional Deep Atrophy (RDA): A self-supervised learning method to automatically identify regions associated with Alzheimer's disease progression from longitudinal MRI

**RDA** is a model for longitudinal quantification of AD progression in the medial temporal lobe (MTL) region of the brain, the earliest detectable MRI change region of AD.


# Instructions

To use RDA for training, clone this repository and install the requirements listed in `setup.py`.
This code inherits the structure of voxelmorph code, with only the PyTorch part. New data loader and models are added to the original code for RDA training and analysis.

## Training

If you would like to train your own model, you will likely need to customize some of the data loading code in `voxelmorph/generators.py` for your own datasets and data formats. 

For a given `/path/to/training/data`, the following script will train the dense network (described in MICCAI 2018 by default) using scan-to-scan registration. Model weights will be saved to a path specified by the `--model-dir` flag. Default is to create a "Model" folder parallel to the code folder.

```
./vxm_runs/torch/train.py /path/to/training/data --model-dir /path/to/models/output --gpu 0
```

## Testing

To test the quality of a model by computing dice overlap between an atlas segmentation and warped test scan segmentations, run:

```
./vxm_runs/torch/test_attention.py --model model.h5 --atlas atlas.npz -
```


## Parameter choices

Optimal parameters are set as the default value in the training process.


### CVPR version

For the CC loss function, we found a reg parameter of 1 to work best. For the MSE loss function, we found 0.01 to work best.


### MICCAI version

For our data, we found `image_sigma=0.01` and `prior_lambda=25` to work best.

In the original MICCAI code, the parameters were applied after the scaling of the velocity field. With the newest code, this has been "fixed", with different default parameters reflecting the change. We recommend running the updated code. However, if you'd like to run the very original MICCAI2018 mode, please use `xy` indexing and `use_miccai_int` network option, with MICCAI2018 parameters.


## Spatial Transforms and Integration

- The spatial transform code, found at `voxelmorph.layers.SpatialTransformer`, accepts N-dimensional affine and dense transforms, including linear and nearest neighbor interpolation options. Note that original development of VoxelMorph used `xy` indexing, whereas we are now emphasizing `ij` indexing.

- For the MICCAI2018 version, we integrate the velocity field using `voxelmorph.layers.VecInt`. By default we integrate using scaling and squaring, which we found efficient.


# VoxelMorph Papers

Only the most basic part of voxelmorph was used in RDA:

    **VoxelMorph: A Learning Framework for Deformable Medical Image Registration**  
[Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [Amy Zhao](http://people.csail.mit.edu/xamyzhao/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://adalca.mit.edu)  
IEEE TMI: Transactions on Medical Imaging. 2019. 
[eprint arXiv:1809.05231](https://arxiv.org/abs/1809.05231)

    **An Unsupervised Learning Model for Deformable Medical Image Registration**  
[Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [Amy Zhao](http://people.csail.mit.edu/xamyzhao/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://adalca.mit.edu)  
CVPR 2018. [eprint arXiv:1802.02604](https://arxiv.org/abs/1802.02604)


# Notes:
- **keywords**: Longitudinal MRI, preclinical Alzheimer's Disease, self-supervised learning, attention maps, interpretable AI  
- The `master` branch is still in testing as we roll out a major refactoring of the library.     
- If you'd like to run code from VoxelMorph publications, please use the `legacy` branch.  
- **data in papers**: 

We used longitudinal T1-weighted MRI images from ADNI2 and ADNI GO phase. ADNI data downloading: https://adni.loni.usc.edu/


where we replace `--satit --iscale` with `--cost NMI` for registration across MRI contrasts.


# Contact:
For any problems or questions please [open an issue](https://github.com/MengjinDong/RegionalDeepAtrophy/issues) for code problems/questions or [start a discussion](https://github.com/MengjinDong/RegionalDeepAtrophy/discussions) for longitudinal MRI registration question/discussion.  
