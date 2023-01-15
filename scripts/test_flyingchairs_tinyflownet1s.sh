#!/bin/bash

TIME=$(date +"%Y%m%d-%H%M%S")

# meta
CHECKPOINT="$PWD/../output/TinyFlowNet1S-20230113-225922-train-flyingchairs/checkpoint_best_epe.ckpt"
FLYINGCHAIRS_HOME=/home/deu/FlyingChairs_release/downscaled_data
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
--validation_dataset_root=$FLYINGCHAIRS_HOME \
--validation_keys="[epe]"
