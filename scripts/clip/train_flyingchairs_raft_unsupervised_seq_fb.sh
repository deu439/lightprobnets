#!/bin/bash
#SBATCH --qos=medium
#SBATCH --time=2-00:00:00
## #SBATCH --nodes=1   # number of nodes
## #SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --gres=gpumem:20g
#SBATCH --partition=g
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=R-%x.%a.%j.out

source env.sh

cd ../
bash ./train_flyingchairs_raft_unsupervised_seq_fb.sh
exit $?
