#!/bin/bash

VENV=../../venv

# We have CUDA 11.3.1 on CLIP
# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
ml load python/3.7.4-gcccore-8.3.0
ml load cuda/11.3.1
#ml load cudnn/7.1.4.18-fosscuda-2018b
source $VENV/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH
export FLYINGCHAIRS_HOME=/scratch-cbe/users/georg.pichler/flyingchairs/FlyingChairs_release/data
export TINYFLYINGCHAIRS_HOME=/scratch-cbe/users/jan.dorazil/flyingchairs/downscaled
