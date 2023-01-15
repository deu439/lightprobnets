#!/bin/bash

TIME=$(date +"%Y%m%d-%H%M%S")

# meta
CHECKPOINT=None
FLYINGCHAIRS_HOME=/home/deu/FlyingChairs_release/downscaled_data
MODEL=TinyFlowNetProbout
PREFIX="train-flyingchairs"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$PWD/../output/$MODEL-$TIME-$PREFIX"

# training configuration
python ../main.py \
--batch_size=14 \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[50, 70, 90]" \
--loss=Elbo \
--model=$MODEL \
--num_workers=12 \
--optimizer=Adam \
--optimizer_lr=1e-4 \
--optimizer_weight_decay=4e-4 \
--proctitle=$MODEL \
--save=$SAVE_PATH \
--total_epochs=100 \
--training_augmentation=RandomAffineFlow \
--training_dataset=TinyFlyingChairsTrain \
--training_dataset_num_examples=-1 \
--training_dataset_photometric_augmentations=True \
--training_dataset_root=$FLYINGCHAIRS_HOME \
--training_key=elbo \
--validation_dataset=TinyFlyingChairsValid  \
--validation_dataset_root=$FLYINGCHAIRS_HOME \
--validation_keys="[epe]" \
--validation_keys_minimize="[True]"
