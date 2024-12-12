#!/bin/bash
#SBATCH -N 1                     # Number of nodes
#SBATCH -C gpu                   # Use GPU nodes
#SBATCH -q regular              # Queue name
#SBATCH -t 04:00:00             # Run time (hh:mm:ss)
#SBATCH --gres=gpu:4            # Request 4 GPUs
#SBATCH --ntasks=4              # Number of tasks (one per GPU)
#SBATCH --gpus-per-task=1       # GPUs per task
#SBATCH -A m4431                # Account number
#SBATCH -J chexpert_ddpm        # Job name
#SBATCH -o logs/chexpert_%j.out # Output file (%j will be replaced by job ID)
#SBATCH -e logs/chexpert_%j.err # Error file

# Change to correct directory
cd /pscratch/sd/m/menaman1/Final_Project/Fast-DDPM

# Load environment
source /pscratch/sd/m/menaman1/Final_Project/myenv/bin/activate

# Set environment variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the training
srun --nodes=1 --ntasks=4 --gpus-per-task=1 python fast_ddpm_main.py \
    --config chexpert_linear.yml \
    --dataset CheXpert \
    --scheduler_type uniform \
    --timesteps 10