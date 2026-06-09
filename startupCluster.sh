#!/bin/bash
#SBATCH --job-name=data-setup
#SBATCH --account=gts-wchu38
#SBATCH --partition=cpu-small    # or cpu-large
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1               # only 1 process (Python script)
#SBATCH --cpus-per-task=4        # n cores to fork workers into
#SBATCH --mem=4G                
#SBATCH --time=00:10:00           
#SBATCH --array=1          
#SBATCH --output=/storage/project/r-wchu38-0/dyang379/logs/data-setup_%A_%a.out
#SBATCH --error=/storage/project/r-wchu38-0/dyang379/logs/data-setup_%A_%a.err

module purge
module load uv

export DATA_PATH="/storage/project/r-wchu38-0/dyang379/ASE-inference/training/gridded/"
export PARAM_PATH="/storage/home/hcoda1/6/dyang379/projects/GMR-inference/post-processing-parameters.csv"

# run the data setup script
# it checks the consistency of data paths and split the training, validation, and test sets
# to ensure reproducibility 
python runDataSetup.py