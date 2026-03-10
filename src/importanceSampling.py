import numpy as np
import warnings
from typing import Union
import scipy.io as sio
import os

# import my own functions
from src.optimization import loglikelihood_thawed, loglikelihood_frozen, log_prior
from src.optimization import loglikelihoods_sum
from src.ice import compute_pmp, enthalpy_to_temperature
from src.amortization import to_log_probability_density

def compute_importance_sampling(
    model,
    inference_param,
    inference_sample_size: int,
    evidence_paths,
    beta: float = 10.0
) -> np.ndarray:
    """
    Compute importance sampling weights and update posterior estimates
    using identified thawed/frozen regions from GMM samples.

    Parameters
    ----------
    model                : Gaussian mixture model
    inference_param      : object with fields:
                             .thawed_evidence  – Nx1 array of boolean thawed mask 
                             .frozen_evidence  – Nx1 array of boolean frozen mask
                             .Eb_sim_std       – standardisation std  array
                             .Eb_sim_mean      – standardisation mean array
                             .H                – ice thickness array
                             .sample_size : int
                                                 Number of samples per batch (M).
    beta                  : float
                             e-folding temperature scale for likelihood.

    Returns
    -------
    normalized_weights    : np.ndarray, shape (n_total * M,)
    """
    thawed_evidence     = False
    frozen_evidence     = False

    if inference_param.thawed_evidence is not None:
        print("Thawed evidence is included from inference_param...")
        watermask_val = inference_param.thawed_evidence
        thawed_evidence    = True

    if inference_param.frozen_evidence is not None:
        print("Frozen evidence is included from inference_param...")
        frozenmask_val     = inference_param.frozen_evidence
        frozen_evidence = True

    if thawed_evidence and frozen_evidence:
        evidence_type = "both"
    elif thawed_evidence:
        evidence_type = "thawed"
    elif frozen_evidence:
        evidence_type = "frozen"
    else:
        raise ValueError("No evidence provided for importance sampling.")

    # ------------------------------------------------------------------
    # 3. Pressure melting point from ice thickness
    # ------------------------------------------------------------------
    Tpmp = compute_pmp(inference_param.H)  

    # ------------------------------------------------------------------
    # 4. Pre-broadcast std / mean for de-standardisation
    # ------------------------------------------------------------------
    Eb_sim_std_aug  = np.repeat(inference_param.Eb_sim_std[..., np.newaxis],
                                inference_param.sample_size, axis=-1)
    Eb_sim_mean_aug = np.repeat(inference_param.Eb_sim_mean[..., np.newaxis],
                                inference_param.sample_size, axis=-1)

    log_weight_process  = True
    log_density_process = True
    all_weights         = []

    # --- log-probability density from GMM ---------
    Eb_sample = model.sample(inference_param.sample_size) # dimension is sample_size x D
    log_px = to_log_probability_density(model, Eb_sample)

    # --- enthalpy → temperature (Kelvin) ---
    H_aug    = np.broadcast_to(
        inference_param.H[..., np.newaxis, np.newaxis],
        Eb_sample.shape
    )
    Eb_post  = Eb_sample * Eb_sim_std_aug + Eb_sim_mean_aug
    X_post_K = enthalpy_to_temperature(Eb_post, H_aug)     # shape: (N1, N2, C, M)

    N1, N2   = X_post_K.shape[0], X_post_K.shape[1]
    M        = X_post_K.shape[3]

    # --- flatten spatial dims ---
    X_post_K_reshaped = X_post_K.reshape(N1 * N2, M)       # (N, M)
    Tpmp_reshaped     = Tpmp.reshape(-1, 1)                 # (N, 1)

    # --- debug NaN count ---
    nan_count = np.sum(np.isnan(X_post_K_reshaped))
    print(f"NaN count in X_post_K_reshaped: {nan_count}")

    # --- compute importance weights ---
    if evidence_type == "thawed":
        batch_weights = importance_weight_thawed(
            X_post_K_reshaped, watermask_reshaped, log_px,
            log_density=log_density_process,
            log_weight=log_weight_process,
            beta=beta,
            Tpmp=Tpmp_reshaped
        )
    elif evidence_type == "frozen":
        batch_weights = importance_weight_frozen(
            X_post_K_reshaped, frozenmask_val, log_px,
            log_density=log_density_process,
            log_weight=log_weight_process,
            beta=beta,
            Tpmp=Tpmp_reshaped
        )
    elif evidence_type == "both":
        batch_weights = importance_weight_frozen_and_thawed(
            X_post_K_reshaped, watermask_reshaped,
            frozenmask_val, log_px,
            log_density=log_density_process,
            log_weight=log_weight_process,
            beta=beta,
            Tpmp=Tpmp_reshaped
        )

    all_weights.append(batch_weights)

    # ------------------------------------------------------------------
    # 7. Normalise weights and diagnostics
    # ------------------------------------------------------------------
    weights_vec = np.concatenate(all_weights)                    # (n_total * M,)
    log_w_vec   = weights_vec if log_weight_process else np.log(weights_vec)

    lse_val             = log_sum_exp(log_w_vec)
    normalized_weights  = np.exp(log_w_vec - lse_val)

    max_weight          = float(np.max(normalized_weights))
    ess                 = float(1.0 / np.sum(normalized_weights ** 2))
    sorted_w            = np.sort(normalized_weights)[::-1]      # descending

    total_samples = inference_sample_size * n_total
    print(f"Max normalized weight: {max_weight:.6f}  (should be <<1 for good mixing)")
    print(f"Effective sample size (ESS): {ess:.1f} out of {total_samples} "
          f"(should be >10% of M for reliable variance)")
    print(f"Max weight:    {max_weight:.6f}")
    print(f"Second max weight: {sorted_w[1]:.6f}")
    print(f"Weight ratio (max/second): {max_weight / sorted_w[1]:.4f}")

    return normalized_weights
