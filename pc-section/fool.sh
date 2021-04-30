#!/bin/bash

#SBATCH --time=06:00:00
#SBATCH --job-name=fgbi-inc
#SBATCH --output=%x.out
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-user=a.eslami.75@gmail.com
#SBATCH --mail-type=BEGIN

cd /home/aminesi/ase
source ./dep.sh
export BATCH_SIZE=128
export IMAGE_SIZE=299
#export START_BATCH=9
python fool.py
deactivate