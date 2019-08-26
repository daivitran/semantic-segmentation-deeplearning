#!/bin/bash
# Job name:
#SBATCH --job-name=2dunet
#
# Account:
#SBATCH --account=fc_biome
#
# Partition:
#SBATCH --partition=savio2_1080ti
#
# QoS:
#SBATCH --qos=savio_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks(one for each GPU):
#SBATCH --ntasks=1
#
# Processesors per task:
#SBATCH --cpus-per-task=4
# Number of GPUs,
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=72:00:00
#
##
module load python
module load tensorflow/1.10.0-py36-pip-gpu
module load cuda
python /global/scratch/dt111997/project/train_basic.py
