#!/bin/bash

# Check whether FLYINGCHAIRS_HOME is defined and points to an existing directory
if [ ! -d "$FLYINGCHAIRS_HOME" ]; then
  echo "Please define environment variable FLYINGCHAIRS_HOME that points to the dataset's home directory."
  exit
fi

# meta
LOSS=ElboFB
MODEL=FlowNetProbFB
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
--batch_size=8 \
--checkpoint=$CHECKPOINT \
--logging_model_graph=True \
--loss=$LOSS \
--loss_alpha=1.0 \
--loss_beta=1.0 \
--loss_gamma=1.0 \
--loss_delta=1.0 \
--loss_Nsamples=1 \
--loss_mask_cost=12.4 \
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
--training_key=elbo \
--validation_dataset=FlyingChairsValid  \
--validation_dataset_root=$FLYINGCHAIRS_HOME \
--validation_keys="[elbo, epe]" \
--validation_keys_minimize="[True, False]"