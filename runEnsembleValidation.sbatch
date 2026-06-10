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
#SBATCH --array=1-10              # Limit simultaneous jobs to avoid overwhelming cluster
#SBATCH --output=/storage/project/r-wchu38-0/dyang379/logs/kfoldcv_%A_%a.out
#SBATCH --error=/storage/project/r-wchu38-0/dyang379/logs/kfoldcv_%A_%a.err

module purge
module load uv
cd $SLURM_SUBMIT_DIR

PARAMS_FILE="tuning-combination-full.csv"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $PARAMS_FILE)
export KFOLD_NUM=$(echo $LINE | cut -d',' -f1)
export BETA_POSTERIOR=$(echo $LINE | cut -d',' -f2)

# set project specific environmental variables
export DATA_PATH="/storage/project/r-wchu38-0/dyang379/ASE-inference/training/gridded/"
export PARAM_PATH="/storage/home/hcoda1/6/dyang379/projects/GMR-inference/post-processing-parameters.csv"
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

cd $SLURM_SUBMIT_DIR
uv run python src/runValidation.py
