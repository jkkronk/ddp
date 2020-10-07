#!/bin/bash
#SBATCH  --output=log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /scratch_net/bmicdl03/jonatank/conda/etc/profile.d/conda.sh shell.bash hook
conda activate tfv1

python -u train_vae_original.py --modality FLAIR
