#!/bin/bash

#SBATCH --time=06:00:00
#SBATCH --job-name=zip
#SBATCH --output=%x.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --mail-user=a.eslami.75@gmail.com
#SBATCH --mail-type=BEGIN

tar -I pigz -cf adv.tar.gz /scratch/aminesi/image_net/torch/inception_v3 /scratch/aminesi/image_net/torch/resnet50/fgsm /scratch/aminesi/image_net/torch/resnet50/bim /scratch/aminesi/image_net/tf/inception_v3 /scratch/aminesi/image_net/tf/resnet50/fgsm /scratch/aminesi/image_net/tf/resnet50/bim
echo "done"
