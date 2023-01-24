#!/bin/bash

# Check whether TINYFLYINGCHAIRS_HOME is defined and points to an existing directory
if [ ! -d "$TINYFLYINGCHAIRS_HOME" ]; then
  echo "Please define environment variable TINYFLYINGCHAIRS_HOME that points to the dataset's home directory."
  exit
fi

TIME=$(date +"%Y%m%d-%H%M%S")

# meta
CHECKPOINT="$PWD/../output/TinyFlowNet1S-20230113-225922-train-flyingchairs/checkpoint_best_epe.ckpt"
MODEL=TinyFlowNet1S
PREFIX="test-flyingchairs"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$PWD/../output/$MODEL-$TIME-$PREFIX"

# training configuration
python ../main.py \
--batch_size=8 \
--checkpoint=$CHECKPOINT \
--evaluation=True \
--model=$MODEL \
--num_workers=4 \
--proctitle="$MODEL" \
--save=$SAVE_PATH \
--loss=EPE \
--validation_dataset=TinyFlyingChairsValid  \
--validation_dataset_root=$TINYFLYINGCHAIRS_HOME \
--validation_keys="[epe]"
