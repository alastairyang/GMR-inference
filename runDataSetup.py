import numpy as np
import os
import sys
import pandas as pd
import scipy
sys.path.append('../')

# ---------------- PARAMETERS ----------------
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

data_path  = os.getenv('DATA_PATH')
param_path = os.getenv('PARAM_PATH')
folder = data_path
# ------------------------------------------------- LOAD SIMULATION DATA -------------------------------------------------
# get the DATA_PATH from the environment variable
X_filename           = 'trainingAll4D_Eb_sim_standardized.mat'
Y_filename           = 'trainingAll4D_Ns_sim_masked_standardized.mat'
X_data           = scipy.io.loadmat(folder + X_filename)
Y_data           = scipy.io.loadmat(folder + Y_filename)

# load
Eb_standardized = X_data['Eb_standardized']
Ns_standardized = Y_data['Ns_standardized_masked']
# replace nan by 0

# -------------------
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1
assert train_ratio + validation_ratio + test_ratio == 1.0, "The sum of the ratios must be 1."

n_samples_total = Eb_standardized.shape[-1]
n_train = int(n_samples_total * train_ratio)
n_validation = int(n_samples_total * validation_ratio)
n_test = n_samples_total - n_train - n_validation

indices = np.arange(n_samples_total)
random_state = np.random.RandomState(seed=42)
random_state.shuffle(indices)

# print the first 20 randomly shuffled indices for verification
print("First 20 randomly shuffled indices:", indices[:20])

X_train      = Eb_standardized[..., indices[:n_train]]
X_validation = Eb_standardized[..., indices[n_train:n_train + n_validation]]
X_test       = Eb_standardized[..., indices[n_train + n_validation:]]

Y_train      = Ns_standardized[..., indices[:n_train]]
Y_validation = Ns_standardized[..., indices[n_train:n_train + n_validation]]
Y_test       = Ns_standardized[..., indices[n_train + n_validation:]]

print("Shape of X_train:", X_train.shape)
print("Shape of X_validation:", X_validation.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of Y_train:", Y_train.shape)
print("Shape of Y_validation:", Y_validation.shape)
print("Shape of Y_test:", Y_test.shape)

save_folder = "data/train-validate-test/"
os.makedirs(save_folder, exist_ok=True)
np.savez(save_folder + "train_data.npz", X_train=X_train, Y_train=Y_train)
np.savez(save_folder + "validation_data.npz", X_validation=X_validation, Y_validation=Y_validation)
np.savez(save_folder + "test_data.npz", X_test=X_test, Y_test=Y_test)
print(f"Data saved to {save_folder}")