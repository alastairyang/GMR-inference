#!/bin/bash

module purge
module load uv

export DATA_PATH="/storage/project/r-wchu38-0/dyang379/ASE-inference/training/gridded/"
export PARAM_PATH="/storage/home/hcoda1/6/dyang379/projects/GMR-inference/post-processing-parameters.csv"

# run the data setup script
# it checks the consistency of data paths and split the training, validation, and test sets
# to ensure reproducibility 
python runDataSetup.py