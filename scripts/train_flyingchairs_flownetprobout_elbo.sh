#!/bin/bash

# Check whether FLYINGCHAIRS_HOME is defined and points to an existing directory
if [ ! -d "$FLYINGCHAIRS_HOME" ]; then
  echo "Please define environment variable FLYINGCHAIRS_HOME that points to the dataset's home directory."
  exit
fi

TIME=$(date +"%Y%m%d-%H%M%S")

# meta
CHECKPOINT=None
MODEL=FlowNetProbOut
PREFIX=train-flyingchairs
SAVE_PATH="$PWD/../output/$MODEL-$TIME-$PREFIX"

# training configuration
python ../main.py \
--batch_size=8 \
--checkpoint=$CHECKPOINT \
--logging_model_graph=True \
--loss=MultiScaleElbo \
--loss_alpha=1.0 \
--loss_beta=1.0 \
--loss_Nsamples=1 \
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
--training_key=total_loss \
--validation_dataset=FlyingChairsValid  \
--validation_dataset_root=$FLYINGCHAIRS_HOME \
--validation_keys="[elbo, epe, llh]" \
--validation_keys_minimize="[True, False, False]"