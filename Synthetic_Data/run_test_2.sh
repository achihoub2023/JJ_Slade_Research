#!/bin/bash

#SBATCH --partition=gpu     # Partition (job queue)

#SBATCH --requeue                 # Return job to the queue if preempted

#SBATCH --job-name=Run_Synth_Data    # Assign a short name to your job

#SBATCH --nodes=1                 # Number of nodes you require

#SBATCH --ntasks=8                # Total # of tasks across all nodes

#SBATCH --gres=gpu:1              #total number of gpus for the task

#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)

#SBATCH --mem=20G               # Real memory (RAM) required (MB)

#SBATCH --time=48:20:00           # Total run time limit (HH:MM:SS)

#SBATCH --output=Edge_results/slurm.%N.%j.out  # STDOUT output file

#SBATCH --error=Edge_results/slurm.%N.%j.err   # STDERR output file (optional)

module use /projects/community/modulefiles
module load py-data-science-stack/5.1.0-kp807
module load python/3.8
python3.8 Edges.py
