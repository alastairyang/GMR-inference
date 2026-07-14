import numpy as np
import warnings
from typing import Union
import scipy.io as sio
from scipy.stats import multivariate_normal, multivariate_t
import os
import signal
import time

# import my own functions
from probability import loglikelihood_thawed, loglikelihood_frozen, log_prior
from probability import loglikelihoods_sum
from src.ice import compute_pmp, enthalpy_to_temperature
from src.amortization import to_log_probability_density

def log_sum_exp(log_w):
    """Compute log-sum-exp in a numerically stable way."""
    max_log_w = np.max(log_w)
    return max_log_w + np.log(np.sum(np.exp(log_w - max_log_w)))

def compute_importance_sampling(
    model,
    mu,
    sigma,
    inference_param,
    V,
    beta: float = 10.0,
    batch_size: int = 500
) -> np.ndarray:
    """
    Compute importance sampling weights and update posterior estimates
    using identified thawed/frozen regions from GMM samples.

    Parameters
    ----------
    model                : Gaussian mixture model
    mu                   : array.
                             mean of the Gaussian in latent space (MAP point from optimization)
    sigma                : array.
                             covariance of the Gaussian in latent space (specified)
    inference_param      : object with fields:
                             .thawed_evidence  – Nx1 array of boolean thawed mask 
                             .frozen_evidence  – Nx1 array of boolean frozen mask
                             .Eb_sim_std       – standardisation std  array
                             .Eb_sim_mean      – standardisation mean array
                             .H                – ice thickness array
                             .sample_size : int
                             .dw                 array, fractional thawed area per pixel
                             .df                 array, fractional frozen area per pixel
                             .nx                 x dimension of the image (n_features).
                             .ny                 y dimension of the image (n_features).
                                                 Number of samples per batch (M).
    V                     : array. 
                             right singular vectors from PCA (n_features, n_latent_feature)
    beta                  : float
                             e-folding temperature scale for likelihood.
    batch_size            : int
                             Number of samples per batch (M).

    Returns
    -------
    normalized_weights    : np.ndarray, shape (n_total * M,)
    """
    thawed_evidence     = False
    frozen_evidence     = False

    if inference_param['thawed_evidence'] is not None:
        print("Thawed evidence is included from inference_param...")
        watermask_val = inference_param['thawed_evidence']
        thawed_evidence    = True

    if inference_param['frozen_evidence'] is not None:
        print("Frozen evidence is included from inference_param...")
        frozenmask_val     = inference_param['frozen_evidence']
        frozen_evidence = True

    if thawed_evidence and frozen_evidence:
        evidence_type = "both"
    elif thawed_evidence:
        evidence_type = "thawed"
    elif frozen_evidence:
        evidence_type = "frozen"
    else:
        raise ValueError("No evidence provided for importance sampling.")

    Tpmp = compute_pmp(inference_param['H'])  

    Eb_sim_std_aug  = inference_param['Eb_sim_std'][..., np.newaxis]
    Eb_sim_mean_aug = inference_param['Eb_sim_mean'][..., np.newaxis]

    log_weight_process  = True
    log_density_process = True
    
    if inference_param['sample_size'] < batch_size:
        batch_num = 1
        print(f"Total samples: {inference_param['sample_size']} is less than batch size")
        print(f"Drawing {inference_param['sample_size']} samples in one batch.")
        sample_size_per = inference_param['sample_size']
    else:
        batch_num = int(np.ceil(inference_param['sample_size'] / batch_size))
        sample_size_per = batch_size
        print(f"Total samples: {inference_param['sample_size']}, "
            f"Batch size: {sample_size_per}, "
            f"Number of batches: {batch_num}")
    
    # collect all the weights across batches
    all_weights = []
    all_log_q = []
    posterior_paths = []
    for b in range(batch_num):
        print(f"Processing batch {b+1}/{batch_num}...")

        # ---- sampling from the proposal distribution (Gaussian)
        proposal = multivariate_t(loc=mu, shape=sigma, df=3)
        Eb_sample_star = proposal.rvs(size=sample_size_per)

        # log prob of proposal
        log_q = proposal.logpdf(Eb_sample_star)
        # print("Eb_sample shape:", Eb_sample_star.shape)
        # print("log_prob_proposal:", log_prob_proposal)

        log_px = to_log_probability_density(model, Eb_sample_star)

        # bring Eb_sample to data dimension 
        Eb_sample = (Eb_sample_star @ V).T 

        # --- enthalpy → temperature (Kelvin) ---
        H_aug    = np.broadcast_to(
            inference_param['H'][..., np.newaxis],
            Eb_sample.shape
        )
        Eb_post  = Eb_sample * Eb_sim_std_aug + Eb_sim_mean_aug
        Tb = enthalpy_to_temperature(Eb_post, H_aug, istorch=False)     # shape: (N1, N2, C, M)

        nan_count = np.sum(np.isnan(Tb))
        print(f"NaN count in Tb: {nan_count}")

        # --- compute importance weights ---
        if evidence_type == "thawed":
            weights = importance_weight_thawed(
                Tb, watermask_val, log_px,
                log_density=log_density_process,
                log_weight=log_weight_process,
                beta=beta,
                Tpmp=Tpmp,
                dw=inference_param['dw']
            )
        elif evidence_type == "frozen":
            weights = importance_weight_frozen(
                Tb, frozenmask_val, log_px,
                log_density=log_density_process,
                log_weight=log_weight_process,
                beta=beta,
                Tpmp=Tpmp,
                df=inference_param['df']
            )
        elif evidence_type == "both":
            weights = importance_weight_frozen_and_thawed(
                Tb, watermask_val,
                frozenmask_val, log_px,
                log_density=log_density_process,
                log_weight=log_weight_process,
                beta=beta,
                Tpmp=Tpmp,
                dw=inference_param['dw'],
                df=inference_param['df']
            )

        # save the Eb samples, weights, log_q from this batch
        # to "data/posterior-samples-scratch"
        save_dir = "data/posterior-samples-scratch"
        os.makedirs(save_dir, exist_ok=True)

        # combine into one dict
        batch_data = {
            'Eb_sample': Eb_sample_star,
            'weights': weights,
            'log_q': log_q,
        }
        # save as .npz file
        path = os.path.join(save_dir, f"batch_{b+1}.npz")
        np.savez(path, **batch_data)
        print(f"Saved batch {b+1} data to {path}")

        posterior_paths.append(path)
        all_weights.append(weights)
        all_log_q.append(log_q)


    all_weights = np.concatenate(all_weights, axis=0)
    all_log_q = np.concatenate(all_log_q, axis=0)
    print("All weights computed for current batch.")
    print("log proposal: ", all_log_q[-10:])
    # Normalise weights and diagnostics
    # first divided by the proposal weights
    weights_vec = all_weights - all_log_q                  
    log_w_vec   = weights_vec

    lse_val             = log_sum_exp(log_w_vec)
    normalized_weights  = np.exp(log_w_vec - lse_val)

    max_weight          = float(np.max(normalized_weights))
    ess                 = float(1.0 / np.sum(normalized_weights ** 2))
    sorted_w            = np.sort(normalized_weights)[::-1]      # descending

    print(f"max of log weights: {np.max(all_weights):.4f}")
    print(f"max of log proposal weights: {np.max(all_log_q):.4f}")
    print(f"Max normalized weight: {max_weight:.6f}  (should be <<1 for good mixing)")
    print(f"Effective sample size (ESS): {ess:.1f} out of {inference_param['sample_size']} "
          f"(should be >10% of M for reliable variance)")
    print(f"Max weight:    {max_weight:.6f}")
    print(f"Second max weight: {sorted_w[1]:.6f}")
    print(f"Weight ratio (max/second): {max_weight / sorted_w[1]:.4f}")

    return normalized_weights, ess, posterior_paths


def compute_expected_val(data_paths, V):
    """  
    Compute posterior expected values. 

    Parameters
    ----------
    data_paths : list of str
        List of file paths to .npz files containing 'Eb_sample', 'weights', and 'log_q'.
    V          : array
        Right singular vectors from PCA (n_features, n_latent_feature).

    Returns
    -------
    weighted_mean : array
        Posterior expected value computed from importance sampling. Standardized enthalpy in physical space (not latent)
    """
    print("Pass 1/2: Finding global max log-weight for expected value...")
    global_max_log_w = -np.inf
    for path in data_paths:
        data = np.load(path)
        batch_log_w = data['weights'] - data['log_q']
        global_max_log_w = max(global_max_log_w, np.max(batch_log_w))

    print(f"Global max log-weight found: {global_max_log_w:.4f}")
    print("Pass 2/2: Computing expected value...")

    weighted_sum = None
    total_weight = 0.0
    for idx, path in enumerate(data_paths):
        data = np.load(path)
        Eb_sample_star = data['Eb_sample']
        
        # Shift safely and exponentiate to get standard probabilities
        batch_log_w = data['weights'] - data['log_q']
        weight_new = np.exp(batch_log_w - global_max_log_w)
        weight_new = weight_new[np.newaxis, :]  # (1, n_samples)

        Eb_sample = (Eb_sample_star @ V).T  # (n_feature, n_samples)
        batch_contribution = (Eb_sample * weight_new).sum(axis=1)

        if weighted_sum is None:
            weighted_sum = np.zeros(Eb_sample.shape[0])

        weighted_sum += batch_contribution
        total_weight += weight_new.sum()

        if idx % 10 == 0:
            print(f"Mean pass: processed {idx+1}/{len(data_paths)} batches...")

    weighted_mean = weighted_sum / total_weight
    print("Expected value completed.")
    return weighted_mean

def compute_var_val(data_paths, V, posterior_mean, H, Eb_std, Eb_mean):
    """  
    Compute posterior variance values. 

    Parameters
    ----------
    data_paths : list of str
        List of file paths to .npz files containing 'Eb_sample', 'weights', and 'log_q'.
    V          : array
        Right singular vectors from PCA (n_features, n_latent_feature).
    posterior_mean : array
        Posterior mean in temperature space (Kelvin) for each pixel, computed from importance sampling.
    H           : array
        Ice thickness array (N,).
    Eb_std      : array
        Standard deviation used for standardizing enthalpy (N,).
    Eb_mean     : array
        Mean used for standardizing enthalpy (N,).

    Returns
    -------
    weighted_var : array
        Posterior variance computed from importance sampling. Variance of temperature in physical space (not latent
    """
    print("Pass 1/2: Finding global max log-weight for variance...")
    global_max_log_w = -np.inf
    for path in data_paths:
        data = np.load(path)
        batch_log_w = data['weights'] - data['log_q']
        global_max_log_w = max(global_max_log_w, np.max(batch_log_w))

    print(f"Global max log-weight found: {global_max_log_w:.4f}")
    print("Pass 2/2: Computing variance...")

    weighted_var_sum = None
    total_weight = 0.0
    for idx, path in enumerate(data_paths):
        data = np.load(path)
        Eb_sample_star = data['Eb_sample']
        
        # Shift safely and exponentiate to get standard probabilities
        batch_log_w = data['weights'] - data['log_q']
        weight_new = np.exp(batch_log_w - global_max_log_w)
        weight_new = weight_new[np.newaxis, :]  

        Eb_sample = (Eb_sample_star @ V).T  
        Eb_sample_ori = Eb_sample * Eb_std[:, np.newaxis] + Eb_mean[:, np.newaxis]
        Tb_sample_ori = enthalpy_to_temperature(Eb_sample_ori, H[:, np.newaxis], istorch=False)

        if weighted_var_sum is None:
            weighted_var_sum = np.zeros_like(posterior_mean)
            
        var_contribution = (weight_new * (Tb_sample_ori - posterior_mean[:, np.newaxis]) ** 2).sum(axis=1)
        weighted_var_sum += var_contribution
        total_weight += weight_new.sum()

        if idx % 10 == 0:
            print(f"Variance pass: processed {idx+1}/{len(data_paths)} batches...")

    weighted_var = weighted_var_sum / total_weight
    print("Variance computation completed.")
    return weighted_var
def importance_weight_frozen(
    x: np.ndarray,
    y: np.ndarray,
    px: np.ndarray,
    log_density: bool = True,
    log_weight: bool = True,
    beta: float = 5.0,
    Tpmp: Union[float, np.ndarray] = 273.15,
    df: np.ndarray = None
) -> np.ndarray:
    """
    Calculates log(p(T_b|w)p(T_b)) weights for importance sampling.
    Measures the probability that where there is frozen, the temperature
    is less than the pressure melting point.

    Parameters
    ----------
    x           : (N, M) array — Basal temperature T_b. N: flattened image dims, M: num samples.
    y           : (N,)   array — Boolean mask: frozen=1, no evidence=0.
    px          : (M,)   array — Proposal densities (log if log_density=True).
    log_density : If True, px is log-densities.
    log_weight  : If True, return log weights; else raw weights.
    beta        : Scalar e-folding temperature scale.
    Tpmp        : Scalar or (N,) array — Pressure melting point temperature(s).

    Returns
    -------
    weights : (M,) array — Log weights if log_weight=True, else raw weights.
    """
    epsilon = 0.05

    N, M = x.shape
    assert y.shape[0] == N, "Mask y must match x's first dimension (N)."
    assert len(px) == M,    "px must have one density per sample (M)."

    # Handle Tpmp as scalar or vector
    if np.isscalar(Tpmp):
        Tpmp_vec = np.full(N, Tpmp)
    else:
        Tpmp_vec = np.asarray(Tpmp)
        assert len(Tpmp_vec) == N, "Tpmp vector must match x's first dimension (N)."

    frozen_rows = np.where(y == 1)[0]
    num_frozen  = len(frozen_rows)

    if num_frozen == 0:
        warnings.warn("No frozen pixels (sum(y)==0); weights set to uniform.")
        return np.zeros(M) if log_weight else np.ones(M)

    prior_logpx = px if log_density else np.log(px)
    weights        = np.zeros(M)

    # --- Vectorised over all M samples at once ---
    X_frozen    = x[frozen_rows, :]          # (num_frozen, M)
    Tpmp_frozen = Tpmp_vec[frozen_rows]      # (num_frozen,)

    exceedance  = np.maximum(0.0, Tpmp_frozen[:, None] - X_frozen)          # (num_frozen, M)
    inner       = np.maximum(epsilon, 1.0 - np.exp(-(1.0 / beta) * exceedance))
    log_lik_mat = np.log(inner)                                              # (num_frozen, M)
    log_likelihood = log_lik_mat.sum(axis=0)             # (M,)

    log_w = log_likelihood + prior_logpx
    weights = log_w if log_weight else np.exp(log_w)

    # Debug: display values from the last sample
    print(f"log_likelihood (last sample) = {log_likelihood[-1]}")
    print(f"prior_logpx (last sample) = {prior_logpx[-1]}")

    return weights


# ─────────────────────────────────────────────────────────────────────────────


def importance_weight_thawed(
    x: np.ndarray,
    y: np.ndarray,
    px: np.ndarray,
    log_density: bool = True,
    log_weight: bool = True,
    beta: float = 5.0,
    Tpmp: Union[float, np.ndarray] = 273.15,
    dw: np.ndarray = None
) -> np.ndarray:
    """
    Calculates log(p(T_b|w)p(T_b)) weights for importance sampling.
    Measures where there is basal water, the probability that it is
    at or above the pressure melting point.

    Parameters
    ----------
    x           : (N, M) array — Basal temperature T_b.
    y           : (N,)   array — Boolean mask: water=1, unknown=0.
    px          : (M,)   array — Proposal densities (log if log_density=True).
    log_density : If True, px is log-densities.
    log_weight  : If True, return log weights; else raw weights.
    beta        : Scalar e-folding temperature scale.
    Tpmp        : Scalar or (N,) array — Pressure melting point temperature(s).

    Returns
    -------
    weights : (M,) array — Log weights if log_weight=True, else raw weights.
    """
    N, M = x.shape
    assert y.shape[0] == N, "Mask y must match x's first dimension (N)."
    assert len(px) == M,    "px must have one density per sample (M)."

    if np.isscalar(Tpmp):
        Tpmp_vec = np.full(N, Tpmp)
    else:
        Tpmp_vec = np.asarray(Tpmp)
        assert len(Tpmp_vec) == N, "Tpmp vector must match x's first dimension (N)."

    water_rows = np.where(y == 1)[0]
    num_water  = len(water_rows)

    if num_water == 0:
        warnings.warn("No water pixels (sum(y)==0); weights set to uniform.")
        return np.zeros(M) if log_weight else np.ones(M)

    prior_logpx = px if log_density else np.log(px)

    # --- Vectorised over all M samples at once ---
    X_water    = x[water_rows, :]            # (num_water, M)
    Tpmp_water = Tpmp_vec[water_rows]        # (num_water,)

    exceedance     = np.maximum(0.0, Tpmp_water[:, None] - X_water)   # (num_water, M)
    log_likelihood = -exceedance.sum(axis=0) / (beta)      # (M,)

    log_w   = log_likelihood + prior_logpx
    weights = log_w if log_weight else np.exp(log_w)

    print(f"log_likelihood (last sample) = {log_likelihood[-1]}")
    print(f"prior_logpx (last sample) = {prior_logpx[-1]}")

    return weights


# ─────────────────────────────────────────────────────────────────────────────


def importance_weight_frozen_and_thawed(
    x: np.ndarray,
    yw: np.ndarray,
    yf: np.ndarray,
    px: np.ndarray,
    log_density: bool = True,
    log_weight: bool = True,
    beta: float = 5.0,
    Tpmp: Union[float, np.ndarray] = 273.15,
    dw: np.ndarray = None,
    df: np.ndarray = None
) -> np.ndarray:
    """
    Calculates log(p(E_b|w)p(E_b)) weights for importance sampling,
    considering both frozen and thawed regions jointly.

    Parameters
    ----------
    x           : (N, M) array — Basal temperature T_b.
    yw          : (N,)   array — Boolean mask: water=1, unknown=0.
    yf          : (N,)   array — Boolean mask: frozen=1, unknown=0.
    px          : (M,)   array — Proposal densities (log if log_density=True).
    log_density : If True, px is log-densities.
    log_weight  : If True, return log weights; else raw weights.
    beta        : Scalar e-folding temperature scale.
    Tpmp        : Scalar or (N,) array — Pressure melting point temperature(s).
    dw          : (N,) array — Fractional thawed area per pixel.
    df          : (N,) array — Fractional frozen area per pixel.

    Returns
    -------
    weights : (M,) array — Log weights if log_weight=True, else raw weights.
    """
    epsilon = 0.05

    N, M = x.shape
    assert yw.shape[0] == N, "Mask yw must match x's first dimension (N)."
    assert yf.shape[0] == N, "Mask yf must match x's first dimension (N)."
    assert len(px) == M,     "px must have one density per sample (M)."

    if np.isscalar(Tpmp):
        Tpmp_vec = np.full(N, Tpmp)
    else:
        Tpmp_vec = np.asarray(Tpmp)
        assert len(Tpmp_vec) == N, "Tpmp vector must match x's first dimension (N)."

    water_rows  = np.where(yw == 1)[0]
    frozen_rows = np.where(yf == 1)[0]
    num_water   = len(water_rows)
    num_frozen  = len(frozen_rows)

    if dw is None:
        dw = yw.astype(float)
    if df is None:
        df = yf.astype(float)

    # # debug: visualize both yw and yf as imshow, reshape first
    # yw_image = dw.reshape(256,256)
    # yf_image = df.reshape(256,256)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("Thawed Evidence Mask (yw)")
    # plt.imshow(yw_image, cmap='Reds')
    # plt.colorbar()
    # plt.gca().invert_yaxis()
    # plt.subplot(1, 2, 2)
    # plt.title("Frozen Evidence Mask (yf)")
    # plt.imshow(yf_image, cmap='Blues')
    # plt.colorbar()
    # plt.gca().invert_yaxis()
    # plt.show()


    if num_water == 0:
        warnings.warn("No water pixels (sum(yw)==0); weights set to uniform.")
        return np.zeros(M) if log_weight else np.ones(M)
    if num_frozen == 0:
        warnings.warn("No frozen pixels (sum(yf)==0); weights set to uniform.")
        return np.zeros(M) if log_weight else np.ones(M)

    prior_logpx = px if log_density else np.log(px)

    # --- Thawed contribution ---
    X_water    = x[water_rows, :]                                           # (num_water, M)
    Tpmp_water = Tpmp_vec[water_rows]
    water_exc  = np.maximum(0.0, Tpmp_water[:, None] - X_water)            # (num_water, M)
    water_exec_area = water_exc * dw[water_rows][:, None]                                 # (num_water, M)
    log_likelihood = -water_exec_area.sum(axis=0) / (beta)            # (M,)

    # --- Frozen contribution ---
    X_frozen    = x[frozen_rows, :]                                         # (num_frozen, M)
    Tpmp_frozen = Tpmp_vec[frozen_rows]
    frozen_exc  = np.maximum(0.0, Tpmp_frozen[:, None] - X_frozen)         # (num_frozen, M)
    exponential = np.maximum(epsilon, 1.0 - np.exp(-(1.0 / beta) * frozen_exc))
    exponential_area = exponential * df[frozen_rows][:, None]                               # (num_frozen, M)
    
    log_likelihood += np.log(exponential_area).sum(axis=0)         # (M,)

    log_w   = log_likelihood + prior_logpx
    weights = log_w if log_weight else np.exp(log_w)

    print(f"log_likelihood = ...{log_likelihood[-10:]}")
    print(f"prior_logpx  = ...{prior_logpx[-10:]}")

    return weights
