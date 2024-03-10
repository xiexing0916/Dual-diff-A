# DVG-Diffusion
An Official Implementation of DVG-Difusion.

![framework.png](framework.png)

## Setup
This code has been tested on an NVIDIA RTX 4090 GPU. 
Furthermore it was developed using Python v3.8 and CUDA 11.3.

In order to run our model, we suggest you create a virtual environment

`conda create -n dvg_diffusion python=3.8`

and activate it with

`conda activate dvg_diffusion`

Subsequently, download and install the required libraries by running

`pip install -r requirements.txt`

## Dataset
You need to download the publicly available 
[LIDC-IDRI](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254) dataset 
and preprocess the data following [X2CT-GAN](https://github.com/kylekma/X2CT)

Also, you need to compile drr_projector following 
<https://github.com/cpeng93/XraySyn>


## Train
To train the VDG-Diffusino, run this command:

`python train/train_ddpm.py 
model=ddpm 
dataset=lidc
dataset.root_dir=<INSERT_DATASET_PATH> 
model.results_folder_postfix=./checkpoints/ddpm
model.vqgan_ckpt=<INSERT_PATH_TO_CHECKPOINT> 
model.vqgan_ckpt_bp=<INSERT_PATH_TO_CHECKPOINT> 
model.vqgan_ckpt_bp0=<INSERT_PATH_TO_CHECKPOINT> 
model.vqgan_ckpt_bp025=<INSERT_PATH_TO_CHECKPOINT> 
model.diffusion_img_size=32 
model.diffusion_depth_size=32 
model.diffusion_num_channels=8 
model.dim_mults=[1,2,4,8] 
model.batch_size=2
model.gpus=1`


## Evaluation
You can validate by:

`python train/val_ddpm.py 
model=ddpm 
dataset=lidc
dataset.root_dir=<INSERT_DATASET_PATH> 
model.results_folder_postfix=./checkpoints/ddpm
model.vqgan_ckpt=<INSERT_PATH_TO_CHECKPOINT> 
model.vqgan_ckpt_bp=<INSERT_PATH_TO_CHECKPOINT> 
model.vqgan_ckpt_bp0=<INSERT_PATH_TO_CHECKPOINT> 
model.vqgan_ckpt_bp025=<INSERT_PATH_TO_CHECKPOINT> 
model.diffusion_img_size=128
model.diffusion_depth_size=32 
model.diffusion_num_channels=8 
model.dim_mults=[1,2,4,8] 
model.batch_size=2
model.gpus=1`

## Other codes will be released after publication. (VQ-GAN with BP, New view synthesis)

## Acknowledgement
This code is build on the following repositories:

https://github.com/FirasGit/medicaldiffusion


