import numpy as np
import warnings
from typing import Union
import scipy.io as sio
from scipy.stats import multivariate_normal
import os

# import my own functions
from src.optimization import loglikelihood_thawed, loglikelihood_frozen, log_prior
from src.optimization import loglikelihoods_sum
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
                             .nx                 x dimension of the image (n_features).
                             .ny                 y dimension of the image (n_features).
                                                 Number of samples per batch (M).
    V                     : array. 
                             right singular vectors from PCA (n_features, n_latent_feature)
    beta                  : float
                             e-folding temperature scale for likelihood.

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

    # ------------------------------------------------------------------
    # 3. Pressure melting point from ice thickness
    # ------------------------------------------------------------------
    Tpmp = compute_pmp(inference_param['H'])  

    # ------------------------------------------------------------------
    # 4. Pre-broadcast std / mean for de-standardisation
    # ------------------------------------------------------------------
    Eb_sim_std_aug  = np.repeat(inference_param['Eb_sim_std'][..., np.newaxis],
                                inference_param['sample_size'], axis=-1)
    Eb_sim_mean_aug = np.repeat(inference_param['Eb_sim_mean'][..., np.newaxis],
                                inference_param['sample_size'], axis=-1)
    
    print("size of Eb_sim_std_aug:", Eb_sim_std_aug.shape)
    print("size of Eb_sim_mean_aug:", Eb_sim_mean_aug.shape)

    log_weight_process  = True
    log_density_process = True
    all_weights         = []

    # ---- sampling from the proposal distribution (Gaussian)
    proposal = multivariate_normal(mean=mu, cov=sigma)
    Eb_sample = proposal.rvs(size=inference_param['sample_size'])
    # log prob of proposal
    log_q = proposal.logpdf(Eb_sample)
    # print("Eb_sample shape:", Eb_sample.shape)
    # print("log_prob_proposal:", log_prob_proposal)

    # --- log-probability density from prior ---------
    log_px = to_log_probability_density(model, Eb_sample)

    # bring Eb_sample to data dimension 
    Eb_sample = (Eb_sample @ V).T 

    # --- enthalpy → temperature (Kelvin) ---
    H_aug    = np.broadcast_to(
        inference_param['H'][..., np.newaxis],
        Eb_sample.shape
    )
    Eb_post  = Eb_sample * Eb_sim_std_aug + Eb_sim_mean_aug
    Tb = enthalpy_to_temperature(Eb_post, H_aug, istorch=False)     # shape: (N1, N2, C, M)

    # --- flatten spatial dims ---

    # --- debug NaN count ---
    nan_count = np.sum(np.isnan(Tb))
    print(f"NaN count in Tb: {nan_count}")

    # --- compute importance weights ---
    if evidence_type == "thawed":
        weights = importance_weight_thawed(
            Tb, watermask_val, log_px,
            log_density=log_density_process,
            log_weight=log_weight_process,
            beta=beta,
            Tpmp=Tpmp
        )
    elif evidence_type == "frozen":
        weights = importance_weight_frozen(
            Tb, frozenmask_val, log_px,
            log_density=log_density_process,
            log_weight=log_weight_process,
            beta=beta,
            Tpmp=Tpmp
        )
    elif evidence_type == "both":
        weights = importance_weight_frozen_and_thawed(
            Tb, watermask_val,
            frozenmask_val, log_px,
            log_density=log_density_process,
            log_weight=log_weight_process,
            beta=beta,
            Tpmp=Tpmp
        )


    print("All weights computed for current batch.")

    # Normalise weights and diagnostics
    # first divided by the proposal weights
    weights_vec = weights - log_q                    # (n_total * M,)
    log_w_vec   = weights_vec if log_weight_process else np.log(weights_vec)

    lse_val             = log_sum_exp(log_w_vec)
    normalized_weights  = np.exp(log_w_vec - lse_val)

    max_weight          = float(np.max(normalized_weights))
    ess                 = float(1.0 / np.sum(normalized_weights ** 2))
    sorted_w            = np.sort(normalized_weights)[::-1]      # descending

    # total_samples = inference_sample_size
    # print(f"Max normalized weight: {max_weight:.6f}  (should be <<1 for good mixing)")
    # print(f"Effective sample size (ESS): {ess:.1f} out of {total_samples} "
    #       f"(should be >10% of M for reliable variance)")
    # print(f"Max weight:    {max_weight:.6f}")
    # print(f"Second max weight: {sorted_w[1]:.6f}")
    # print(f"Weight ratio (max/second): {max_weight / sorted_w[1]:.4f}")

    return normalized_weights



def importance_weight_frozen(
    x: np.ndarray,
    y: np.ndarray,
    px: np.ndarray,
    log_density: bool = True,
    log_weight: bool = True,
    beta: float = 5.0,
    Tpmp: Union[float, np.ndarray] = 273.15
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

    proposal_logpx = px if log_density else np.log(px)
    weights        = np.zeros(M)

    # --- Vectorised over all M samples at once ---
    X_frozen    = x[frozen_rows, :]          # (num_frozen, M)
    Tpmp_frozen = Tpmp_vec[frozen_rows]      # (num_frozen,)

    exceedance  = np.maximum(0.0, Tpmp_frozen[:, None] - X_frozen)          # (num_frozen, M)
    inner       = np.maximum(epsilon, 1.0 - np.exp(-(1.0 / beta) * exceedance))
    log_lik_mat = np.log(inner)                                              # (num_frozen, M)
    log_likelihood = log_lik_mat.sum(axis=0) / num_frozen                   # (M,)

    log_w = log_likelihood + proposal_logpx
    weights = log_w if log_weight else np.exp(log_w)

    # Debug: display values from the last sample
    print(f"log_likelihood (last sample) = {log_likelihood[-1]}")
    print(f"proposal_logpx (last sample) = {proposal_logpx[-1]}")

    return weights


# ─────────────────────────────────────────────────────────────────────────────


def importance_weight_thawed(
    x: np.ndarray,
    y: np.ndarray,
    px: np.ndarray,
    log_density: bool = True,
    log_weight: bool = True,
    beta: float = 5.0,
    Tpmp: Union[float, np.ndarray] = 273.15
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

    proposal_logpx = px if log_density else np.log(px)

    # --- Vectorised over all M samples at once ---
    X_water    = x[water_rows, :]            # (num_water, M)
    Tpmp_water = Tpmp_vec[water_rows]        # (num_water,)

    exceedance     = np.maximum(0.0, Tpmp_water[:, None] - X_water)   # (num_water, M)
    log_likelihood = -exceedance.sum(axis=0) / (beta * num_water)      # (M,)

    log_w   = log_likelihood + proposal_logpx
    weights = log_w if log_weight else np.exp(log_w)

    print(f"log_likelihood (last sample) = {log_likelihood[-1]}")
    print(f"proposal_logpx (last sample) = {proposal_logpx[-1]}")

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
    Tpmp: Union[float, np.ndarray] = 273.15
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

    if num_water == 0:
        warnings.warn("No water pixels (sum(yw)==0); weights set to uniform.")
        return np.zeros(M) if log_weight else np.ones(M)
    if num_frozen == 0:
        warnings.warn("No frozen pixels (sum(yf)==0); weights set to uniform.")
        return np.zeros(M) if log_weight else np.ones(M)

    proposal_logpx = px if log_density else np.log(px)

    # --- Thawed contribution ---
    X_water    = x[water_rows, :]                                           # (num_water, M)
    Tpmp_water = Tpmp_vec[water_rows]
    water_exc  = np.maximum(0.0, Tpmp_water[:, None] - X_water)            # (num_water, M)
    log_likelihood = -water_exc.sum(axis=0) / (beta * num_water)            # (M,)

    # --- Frozen contribution ---
    X_frozen    = x[frozen_rows, :]                                         # (num_frozen, M)
    Tpmp_frozen = Tpmp_vec[frozen_rows]
    frozen_exc  = np.maximum(0.0, Tpmp_frozen[:, None] - X_frozen)         # (num_frozen, M)
    exponential = np.maximum(epsilon, 1.0 - np.exp(-(1.0 / beta) * frozen_exc))
    log_likelihood += np.log(exponential).sum(axis=0) / num_frozen          # (M,)

    log_w   = log_likelihood +proposal_logpx
    weights = log_w if log_weight else np.exp(log_w)

    print(f"log_likelihood (last sample) = {log_likelihood[-1]}")
    print(f"proposal_logpx (last sample) = {proposal_logpx[-1]}")

    return weights