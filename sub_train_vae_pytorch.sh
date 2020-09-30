#!/bin/bash
#SBATCH  --output=log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /scratch_net/bmicdl03/jonatank/conda/etc/profile.d/conda.sh shell.bash hook
conda activate pytorch9

python -u run_train_vae_pytorch.py --model_name vae_pytorch2 --config conf_vae.yaml
