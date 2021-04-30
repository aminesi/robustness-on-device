#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --job-name=preprocess
#SBATCH --output=%x.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

cd /home/aminesi/ase
source ./dep.sh
python preprocess.py
deactivate