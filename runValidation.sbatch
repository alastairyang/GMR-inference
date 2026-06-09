#!/bin/bash
#SBATCH --job-name=kfold-cv
#SBATCH --account=gts-wchu38
#SBATCH --partition=cpu-small    # or cpu-large
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1               # only 1 process (Python script)
#SBATCH --cpus-per-task=20        # n cores to fork workers into
#SBATCH --mem=4G                
#SBATCH --time=08:00:00           
#SBATCH --array=1-50%30              # Limit simultaneous jobs to avoid overwhelming cluster
#SBATCH --output=/storage/project/r-wchu38-0/dyang379/logs/kfoldcv_%A_%a.out
#SBATCH --error=/storage/project/r-wchu38-0/dyang379/logs/kfoldcv_%A_%a.err

module purge
module load uv
cd $SLURM_SUBMIT_DIR

# set project specific environmental variables
export PROJECT_DATA=/storage/home/hcoda1/6/dyang379/projects/ice-stochastic-filter/
export PROJECT_RESULTS=$STORAGE_BASE/stochastic-fingerprint/results/
export PROJECT_SCRATCH=/storage/home/hcoda1/6/dyang379/scratch/
export ARRAY_ID=$SLURM_ARRAY_TASK_ID

# Print job information
echo "---------------------------------------"
echo "Begin Slurm Job: $(date)"
echo "Job ID:    $SLURM_JOB_ID"
echo "Array ID:  $SLURM_ARRAY_TASK_ID"
echo "User ID:   $USER"
echo "Node:      $SLURMD_NODENAME"
echo "---------------------------------------"


python runJob.py
