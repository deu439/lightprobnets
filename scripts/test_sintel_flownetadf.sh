#!/bin/bash

# Check whether SINTEL_HOME is defined and points to an existing directory
if [ ! -d "$SINTEL_HOME" ]; then
  echo "Please define environment variable SINTEL_HOME that points to the dataset's home directory."
  exit
fi

# dataset, output, model, (optional) checkpoint
CHECKPOINT="$PWD/../output/../checkpoint.ckpt"  # set this to your trained model file
MODEL=FlowNetADF

# validate clean configuration
PREFIX="validate-sintel-clean"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$PWD/../output/$MODEL-$TIME-$PREFIX-clean"
python ../main.py \
--batch_size=8 \
--checkpoint=$CHECKPOINT \
--evaluation=True \
--model=$MODEL \
--model_noise_variance=1e-3 \
--model_min_variance=1e-4 \
--loss=MultiScaleLaplacian \
--loss_with_llh=True \
--num_workers=4 \
--proctitle="$MODEL" \
--save=$SAVE_PATH \
--validation_dataset=SintelTrainingCleanFull  \
--validation_dataset_root=$SINTEL_HOME \
--validation_keys="[epe]"

# validate final configuration
PREFIX="validate-sintel-final"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$PWD/../output/$MODEL-$TIME-$PREFIX-final"
python ../main.py \
--batch_size=8 \
--checkpoint=$CHECKPOINT \
--evaluation=True \
--loss=MultiScaleLaplacian \
--loss_with_llh=True \
--model=$MODEL \
--model_noise_variance=1e-3 \
--model_min_variance=1e-4 \
--num_workers=4 \
--proctitle="$MODEL" \
--save=$SAVE_PATH \
--validation_dataset=SintelTrainingFinalFull  \
--validation_dataset_root=$SINTEL_HOME \
--validation_keys="[epe]"