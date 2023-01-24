#!/bin/bash

# Check whether TINYFLYINGCHAIRS_HOME is defined and points to an existing directory
if [ ! -d "$TINYFLYINGCHAIRS_HOME" ]; then
  echo "Please define environment variable TINYFLYINGCHAIRS_HOME that points to the dataset's home directory."
  exit
fi

TIME=$(date +"%Y%m%d-%H%M%S")

# meta
CHECKPOINT=None
MODEL=TinyFlowNet1S
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
--loss=EPE \
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
--training_dataset_root=$TINYFLYINGCHAIRS_HOME \
--training_key=epe \
--validation_dataset=TinyFlyingChairsValid  \
--validation_dataset_root=$TINYFLYINGCHAIRS_HOME \
--validation_keys="[epe]" \
--validation_keys_minimize="[True]"
