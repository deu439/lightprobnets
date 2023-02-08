#!/bin/bash

# only allow a single restart of the job.
max_restarts=3

# just gather some information about the job
scontext=$(scontrol show job $SLURM_JOB_ID)
restarts=$(echo "$scontext" | grep -o 'Restarts=.' | cut -d= -f2)
outfile=$(echo "$scontext"  | grep 'StdOut='       | cut -d= -f2)
errfile=$(echo "$scontext"  | grep 'StdErr='       | cut -d= -f2)
timelimit=$(echo "$scontext" | grep -o 'TimeLimit=.*' | awk '{print $1}' | cut -d= -f2)

# term handler
# the function is executed once the job gets the TERM signal
term_handler()
{
    echo "executing term_handler at $(date)"
    if [[ $restarts -lt $max_restarts ]]; then
       # copy the logfile. will be overwritten by the 2nd run
       cp -v $outfile $outfile.$restarts
       # requeue the job and put it on hold. It's not possible to change partition otherwise
       scontrol requeuehold $SLURM_JOB_ID
       # release the job. It will wait in the queue for 2 minutes before the 2nd run can start
       scontrol release $SLURM_JOB_ID
    fi
}

# declare the function handling the TERM signal
trap 'term_handler' TERM


i=0
while [ -d "${BASE_SAVE_PATH}/$i" ]; do
  i=$(( $i + 1))
done
if [ $i -eq 0 ]; then
  export CHECKPOINT=None
else
  export CHECKPOINT="${BASE_SAVE_PATH}/$(( $i - 1))"
fi
export SAVE_PATH="${BASE_SAVE_PATH}/$i"

VENV=../../venv

# We have CUDA 11.3.1 on CLIP
# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
ml load python/3.7.4-gcccore-8.3.0
ml load cuda/11.3.1
#ml load cudnn/7.1.4.18-fosscuda-2018b
source $VENV/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH
export FLYINGCHAIRS_HOME=$SCRATCHDIR/flyingchairs/FlyingChairs_release/data
