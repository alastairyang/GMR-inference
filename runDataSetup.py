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
Eb_filename           = 'trainingAll4D_Eb_sim_standardized.mat'
Ns_filename           = 'trainingAll4D_Ns_sim_masked_standardized.mat'
GHF_filename          = 'trainingAll4D_GHF_sim_standardized.mat'
Eb_data           = scipy.io.loadmat(folder + Eb_filename)
Ns_data           = scipy.io.loadmat(folder + Ns_filename)
GHF_data          = scipy.io.loadmat(folder + GHF_filename)

# load
Eb_standardized = Eb_data['Eb_standardized']
Ns_standardized = Ns_data['Ns_standardized_masked']
GHF_standardized = GHF_data['GHF_standardized']
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

Eb_train      = Eb_standardized[..., indices[:n_train]]
Eb_validation = Eb_standardized[..., indices[n_train:n_train + n_validation]]
Eb_test       = Eb_standardized[..., indices[n_train + n_validation:]]

Ns_train      = Ns_standardized[..., indices[:n_train]]
Ns_validation = Ns_standardized[..., indices[n_train:n_train + n_validation]]
Ns_test       = Ns_standardized[..., indices[n_train + n_validation:]]

GHF_train      = GHF_standardized[..., indices[:n_train]]
GHF_validation = GHF_standardized[..., indices[n_train:n_train + n_validation]]
GHF_test       = GHF_standardized[..., indices[n_train + n_validation:]]

print("Shape of Eb_train:", Eb_train.shape)
print("Shape of Eb_validation:", Eb_validation.shape)
print("Shape of Eb_test:", Eb_test.shape)
print("Shape of Ns_train:", Ns_train.shape)
print("Shape of Ns_validation:", Ns_validation.shape)
print("Shape of Ns_test:", Ns_test.shape)
print("Shape of GHF_train:", GHF_train.shape)
print("Shape of GHF_validation:", GHF_validation.shape)
print("Shape of GHF_test:", GHF_test.shape)

save_folder = "data/train-validate-test/"
os.makedirs(save_folder, exist_ok=True)
np.savez(save_folder + "train_data.npz", Eb_train=Eb_train, Ns_train=Ns_train, GHF_train=GHF_train)
np.savez(save_folder + "validation_data.npz", Eb_validation=Eb_validation, Ns_validation=Ns_validation, GHF_validation=GHF_validation)
np.savez(save_folder + "test_data.npz", Eb_test=Eb_test, Ns_test=Ns_test, GHF_test=GHF_test)
print(f"Data saved to {save_folder}")
