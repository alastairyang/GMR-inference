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
    using identified thawed/frozen regions from CNF samples.

    Parameters
    ----------
    model                : Gaussian mixture model

    inference_param      : object with fields:
                             .n_total          – number of batches
                             .paths            – object/dict with .pos_sample, .fig, .evidence_data
                             .Eb_sim_std       – standardisation std  array
                             .Eb_sim_mean      – standardisation mean array
                             .H_struct         – dict with key "H" (ice thickness)
    inference_sample_size : int
                             Number of samples per batch (M).
    evidence_paths        : object/dict with fields:
                             .thawed_path – path string or empty string
                             .frozen_path – path string or empty string
    beta                  : float
                             E-folding temperature scale for likelihood.

    Returns
    -------
    normalized_weights    : np.ndarray, shape (n_total * M,)
    """
    n_total = inference_param.n_total

    watermask_reshaped  = None
    frozenmask_reshaped = None
    watermask_val       = None   # kept for visualisation
    frozenbase_val      = None   # kept for visualisation
    thawed_evidence     = False
    frozen_evidence     = False

    if evidence_paths.thawed_path:
        print("Thawed evidence is included...")
        mat        = sio.loadmat(evidence_paths.thawed_path)
        watermask_val      = mat["water_mask"]["data"][0, 0]          # adjust key nesting if needed
        h, w               = watermask_val.shape
        watermask_reshaped = watermask_val.reshape(h * w, 1)
        thawed_evidence    = True

    if evidence_paths.frozen_path:
        print("Frozen evidence is included...")
        frozen_path = os.path.join(
            inference_param.paths.evidence_data,
            "basal-frozen-mask", "frozen_mask.mat"
        )
        mat             = sio.loadmat(frozen_path)
        frozenbase_val  = mat["frozen_mask"]["data"][0, 0]            # adjust key nesting if needed
        h, w            = frozenbase_val.shape
        frozenmask_reshaped = frozenbase_val.reshape(h * w, 1)
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
    Tpmp = compute_pmp(inference_param.H_struct["H"])   # shape (N1, N2) or (N,)

    # ------------------------------------------------------------------
    # 4. Pre-broadcast std / mean for de-standardisation
    # ------------------------------------------------------------------
    Eb_sim_std_aug  = np.repeat(inference_param.Eb_sim_std[..., np.newaxis],
                                inference_sample_size, axis=-1)
    Eb_sim_mean_aug = np.repeat(inference_param.Eb_sim_mean[..., np.newaxis],
                                inference_sample_size, axis=-1)

    # ------------------------------------------------------------------
    # 5. Main loop over batches
    # ------------------------------------------------------------------
    log_weight_process  = True
    log_density_process = True
    all_weights         = []

    for i in range(1, n_total + 1):

        # --- load posterior batch ---
        sample_path = os.path.join(
            inference_param.paths.pos_sample,
            f"posterior_sample_{i}.mat"
        )
        mat    = sio.loadmat(sample_path)
        X_post = mat["data_standardized"]           # shape: (N1, N2, C, M)
        y      = mat["observation_standardized"]

        # --- log-probability density from GMM ---------
        prior_sample = model.sample(1)
        log_px = to_log_probability_density(model, prior_sample)

        # --- de-standardise enthalpy → temperature (Kelvin) ---
        H_aug    = np.broadcast_to(
            inference_param.H_struct["H"][..., np.newaxis, np.newaxis],
            X_post.shape
        )
        Eb_post  = X_post * Eb_sim_std_aug + Eb_sim_mean_aug
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
                X_post_K_reshaped, frozenmask_reshaped, log_px,
                log_density=log_density_process,
                log_weight=log_weight_process,
                beta=beta,
                Tpmp=Tpmp_reshaped
            )
        elif evidence_type == "both":
            batch_weights = importance_weight_frozen_and_thawed(
                X_post_K_reshaped, watermask_reshaped,
                frozenmask_reshaped, log_px,
                log_density=log_density_process,
                log_weight=log_weight_process,
                beta=beta,
                Tpmp=Tpmp_reshaped
            )

        all_weights.append(batch_weights)
        print(f"Finished calculating weights for sample {i}")

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
