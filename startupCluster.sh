#!/bin/bash

export DATA_PATH="/home/donglaiyang/Documents/Georgia-Tech/Research/thermal-model/Amundsen-thermal-output-Yang/thermal-training-data/Thwaites-PIG/training/gridded/"
export PARAM_PATH="/home/donglaiyang/Documents/Georgia-Tech/Research/thermal-model/data/post-processing-parameters.csv"

# run the data setup script
# it checks the consistency of data paths and split the training, validation, and test sets
# to ensure reproducibility 
python runDataSetup.py