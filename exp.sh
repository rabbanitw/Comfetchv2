#!/usr/bin/env bash
#SBATCH --job-name=iid4
#SBATCH --time=50:00:00
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa4000:2

python Comfetch.py --comp_rate 4  --partition iid  --numclient 4
