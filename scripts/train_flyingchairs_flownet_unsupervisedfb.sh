#!/bin/bash

# Check whether FLYINGCHAIRS_HOME is defined and points to an existing directory
if [ ! -d "$FLYINGCHAIRS_HOME" ]; then
  echo "Please define environment variable FLYINGCHAIRS_HOME that points to the dataset's home directory."
  exit
fi

# meta
LOSS=UnsupervisedFB
MODEL=FlowNetFB
PREFIX=train-flyingchairs

# Set SAVE_PATH if not already set
if [[ "$SAVE_PATH" == "" ]]; then
  TIME=$(date +"%Y%m%d-%H%M%S")
  SAVE_PATH="$PWD/../output/$MODEL-$TIME-$PREFIX"
fi

# Set CHECKPOINT if not already set
if [[ "$CHECKPOINT" == "" ]]; then
  CHECKPOINT=None
fi

# training configuration
python ../main.py \
--batch_size=16 \
--cuda="cuda" \
--multi_gpu=True \
--checkpoint=$CHECKPOINT \
--logging_model_graph=True \
--loss=$LOSS \
--loss_color_weight=0.0 \
--loss_gradient_weight=0.0 \
--loss_census_weight=10.0 \
--loss_census_radius=3 \
--loss_smooth_1st_weight=1.0 \
--loss_smooth_2nd_weight=0.0 \
--loss_edge_weight=150.0 \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[108, 144, 180]" \
--model=$MODEL \
--num_workers=12 \
--optimizer=Adam \
--optimizer_lr=1e-4 \
--optimizer_weight_decay=4e-4 \
--proctitle=$MODEL \
--save=$SAVE_PATH \
--total_epochs=216 \
--training_augmentation=RandomAffineFlow \
--training_dataset=FlyingChairsTrain \
--training_dataset_num_examples=-1 \
--training_dataset_photometric_augmentations=True \
--training_dataset_root=$FLYINGCHAIRS_HOME \
--training_key=energy \
--validation_dataset=FlyingChairsValid  \
--validation_dataset_num_examples=-1 \
--validation_dataset_root=$FLYINGCHAIRS_HOME \
--validation_keys="[epe]" \
--validation_keys_minimize="[True]"
