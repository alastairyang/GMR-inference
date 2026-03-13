# Assume we've got MAP and Hessian at MAP
import matplotlib.pyplot as plt
import numpy as np
from src.importanceSampling import compute_importance_sampling

neg_H = -hessian

eigenvalues = np.linalg.eigvalsh(neg_H)
print(f"Min eigenvalue:  {eigenvalues.min():.4f}")   # must be > 0
print(f"Condition number: {eigenvalues.max() / eigenvalues.min():.2e}")

sigma_map = np.linalg.inv(neg_H + 1e-6 * np.eye(neg_H.shape[0]))
print('Approximate covariance at MAP is ', sigma_map)

std = np.sqrt(np.diag(sigma_map))
corr_map = sigma_map / np.outer(std, std)

plt.figure(figsize=(8, 6))
plt.imshow(corr_map, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.title('Posterior Correlation Matrix at MAP')
plt.tight_layout()
plt.show()


inference_param = {
    "thawed_evidence": thawed_mask_data,
    "frozen_evidence": frozen_mask_data,
    "Eb_sim_std": Eb_std_data,
    "Eb_sim_mean": Eb_mean_data,
    "H": H,
    "nx": 256,
    "ny": 256,
    "sample_size": 700,
    "dw": dw.detach().numpy(),
    "df": df.detach().numpy()
}

beta_temp = 1

mu_map = X_optimized
neg_H = -hessian

# multiplier = np.array([0.01,0.1,0.5,1,2])
# sizes = np.array([100, 500, 1000])
multiplier = np.array([0.02])
sizes = np.array([100000])
multiplier_grid, size_grid = np.meshgrid(multiplier, sizes)
# zip them for iteration
grid = zip(multiplier_grid.flatten(), size_grid.flatten())
ess_vec = []
for multiplier, size in grid:
    print(f"Running importance sampling with multiplier {multiplier} and size {size}...")
    inference_param["sample_size"] = size
    sigma_map = multiplier * np.linalg.inv(neg_H + 1e-6 * np.eye(neg_H.shape[0]))
    weights, ess, posterior_paths = compute_importance_sampling(gmm_propagated, mu_map, sigma_map, 
                                      inference_param, VT_X, beta_temp)
    ess_vec.append(ess)


# --------- stats

from src.importanceSampling import compute_expected_val, compute_var_val
posterior_mean = compute_expected_val(posterior_paths,VT_X)
# reverse standardization
posterior_mean_ori = posterior_mean * Eb_std_data + Eb_mean_data
posterior_mean_temp = enthalpy_to_temperature(posterior_mean_ori, Tpmp.numpy(), istorch=False)

posterior_var  = compute_var_val(posterior_paths, VT_X, posterior_mean_temp, H, Eb_std_data, Eb_mean_data)


# save to data/posterior-stats/
os.makedirs("data/posterior-stats", exist_ok=True)
posterior_mean_temp_img = posterior_mean_temp.reshape(256, 256)
posterior_var_img = posterior_var.reshape(256, 256)
np.save("data/posterior-stats/posterior_mean_temp.npy", posterior_mean_temp_img)
np.save("data/posterior-stats/posterior_var.npy", posterior_var_img)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(posterior_mean_temp.reshape(256,256) * model_bound_data_ori, cmap='RdBu_r', vmin=250, vmax=273.15)
plt.title('Posterior Mean Estimate of T_b')
plt.colorbar()
plt.gca().invert_yaxis()
plt.subplot(1,2,2)
# plot the difference with MAP
posterior_map_diff = posterior_mean_temp.reshape(256,256) - Tb_MAP
plt.imshow(posterior_map_diff * model_bound_data_ori, cmap='bwr', vmin=-1, vmax=1)
plt.title('Difference between Posterior Mean and MAP Estimate of T_b')
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()

# plot the variance
plt.figure(figsize=(6, 5))
plt.imshow(np.sqrt(posterior_var).reshape(256,256) * model_bound_data_ori, cmap='hot', vmin=0, vmax=3)
plt.title('Posterior Standard Deviation of T_b')
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()  