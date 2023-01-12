#!/bin/bash

TIME=$(date +"%Y%m%d-%H%M%S")

# meta
CHECKPOINT=None
FLYINGCHAIRS_HOME=/home/deu/FlyingChairs_release/data
MODEL=FlowNet1S
PREFIX="train-flyingchairs"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$PWD/../output/$MODEL-$TIME-$PREFIX"

# training configuration
python ../main.py \
--batch_size=4 \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[108, 144, 180]" \
--loss=MultiScaleEPE \
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
--validation_keys="[epe]" \
--validation_keys_minimize="[True]"
