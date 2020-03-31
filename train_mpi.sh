#!/bin/bash

# Slurm sbatch options
#SBATCH -o mpi/mpi_%j.log
#SBATCH -n 17
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --time=10-00:00:00

# Initialize the module command first
source /etc/profile

# Load MPI module
module load mpi/openmpi-4.0

# Load Anaconda module
module load anaconda/2020a

export DISPLAY=':99.0'
Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &

# Call your script as you would from the command line
mpirun python -B train_a3c.py --env_name CarRacing-v0 --model sac --iter 1 --steps 500000
