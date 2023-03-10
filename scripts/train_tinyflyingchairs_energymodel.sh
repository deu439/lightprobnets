#!/bin/bash

# Check whether TINYFLYINGCHAIRS_HOME is defined and points to an existing directory
if [ ! -d "$TINYFLYINGCHAIRS_HOME" ]; then
  echo "Please define environment variable TINYFLYINGCHAIRS_HOME that points to the dataset's home directory."
  exit
fi

TIME=$(date +"%Y%m%d-%H%M%S")

# meta
MODEL=EnergyModel
CHECKPOINT=None
PREFIX="train-tinyflyingchairs"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$PWD/../output/$MODEL-$TIME-$PREFIX"

# training configuration
python ../main.py \
--batch_size=10 \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[50, 70, 90]" \
--loss=ContrastiveLoss \
--model=$MODEL \
--num_workers=3 \
--optimizer=Adam \
--optimizer_lr=1.0 \
--optimizer_weight_decay=4e-4 \
--proctitle=$MODEL \
--save=$SAVE_PATH \
--total_epochs=100 \
--training_augmentation=RandomAffineFlow \
--training_dataset=TinyFlyingChairsTrain \
--training_dataset_num_examples=-1 \
--training_dataset_photometric_augmentations=False \
--training_dataset_root=$TINYFLYINGCHAIRS_HOME \
--training_key=total_loss \
--validation_dataset=TinyFlyingChairsValid  \
--validation_dataset_root=$TINYFLYINGCHAIRS_HOME \
--validation_keys="[reg_loss, cdiv_loss, total_loss]" \
--validation_keys_minimize="[False, False, False]"
