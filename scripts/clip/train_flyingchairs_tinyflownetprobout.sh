#!/bin/bash
#SBATCH --qos=short
#SBATCH --time=6:00:00
## #SBATCH --nodes=1   # number of nodes
## #SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=g
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=R-%x.%a.%j.out

source env.sh

cd ../
bash ./train_flyingchairs_tinyflownetprobout.sh
exit $?
