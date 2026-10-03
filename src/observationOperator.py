import torch
import numpy as np
import torch.nn.functional as F
from src.ice import enthalpy_to_temperature, enthalpy_to_water_fraction
from src.utilities import reverse_standardize

# this script defines the observation operator
# which operators on the simulation output (enthalpy or basal temperature)
# to produce binary features of thawed or frozen base
# ----
# We write everything in torch for AD uses 
# ----
def latent_input_check(V, Eb_star, method=None, epsilon=None):
    """
    Checking the dimensions and compatibility of the latent input tensors.
    """
    if epsilon is None and method == "relaxation":
        raise ValueError(
            "epsilon must be provided for the relaxation method."
        )

    if V.ndim != 2:
        raise ValueError("V must be a two-dimensional tensor.")

    if Eb_star.ndim != 2:
        raise ValueError(
            "Eb_star must be a two-dimensional tensor."
        )

    if V.shape[0] != Eb_star.shape[0]:
        raise ValueError(
            "The latent dimension of V must match the first "
            "dimension of Eb_star."
        )
    return True

def latent_waterfraction_operator_enthalpy(
    V,
    Eb_star,
    Eb_mean,
    Eb_std,
    Tpmp,
    method,
    epsilon=None,
):
    """
    Vectorized obs. operator ...
        mapping from latent PCA coefficients to water fraction.

    Parameters
    ----------
    V : torch.Tensor
        PCA component matrix with shape:

            (n_latent_features, n_physical_features)

    Eb_star : torch.Tensor
        Latent PCA coefficients with shape:

            (n_latent_features, n_samples)

    Eb_mean : torch.Tensor
        Mean enthalpy with shape:

            (n_physical_features,)

    Eb_std : torch.Tensor
        Enthalpy standard deviation with shape:

            (n_physical_features,)

    Tpmp : torch.Tensor
        Pressure-melting-point temperature with shape:

            (n_physical_features,)

    method : str
        Reverse-standardization method.

    epsilon : float, optional
        Relaxation constant.
    """
    latent_input_check(V, Eb_star, method=method, epsilon=epsilon)

    Eb = V.T @ Eb_star

    Eb_mean_column = Eb_mean.reshape(-1, 1)
    Eb_std_column = Eb_std.reshape(-1, 1)
    Tpmp_column = Tpmp.reshape(-1, 1)

    Eb_original = reverse_standardize(
        Eb,
        Eb_mean_column,
        Eb_std_column,
        method=method,
        epsilon=epsilon,
    )

    wf = enthalpy_to_water_fraction(
        Eb_original,
        Tpmp_column,
    )
    
    return wf


def latent_temperature_operator_enthalpy(
    V,
    Eb_star,
    Eb_mean,
    Eb_std,
    Tpmp,
    method,
    epsilon=None,
):
    """
    Vectorized observation operator acting on latent PCA coefficients.

    Parameters
    ----------
    V : torch.Tensor
        PCA component matrix with shape:

            (n_latent_features, n_physical_features)

    Eb_star : torch.Tensor
        Latent PCA coefficients with shape:

            (n_latent_features, n_samples)

    Eb_mean : torch.Tensor
        Mean enthalpy with shape:

            (n_physical_features,)

    Eb_std : torch.Tensor
        Enthalpy standard deviation with shape:

            (n_physical_features,)

    Tpmp : torch.Tensor
        Pressure-melting-point temperature with shape:

            (n_physical_features,)

    method : str
        Reverse-standardization method.

    epsilon : float, optional
        Relaxation constant.
    """
    latent_input_check(V, Eb_star, method=method, epsilon=epsilon)

    # Reconstruct every simulation simultaneously.
    #
    # V.T:     (n_physical_features, n_latent_features)
    # Eb_star: (n_latent_features, n_samples)
    # Eb:      (n_physical_features, n_samples)
    Eb = V.T @ Eb_star

    # Add a singleton sample dimension so the physical-location
    # quantities broadcast across all ensemble members.
    Eb_mean_column = Eb_mean.reshape(-1, 1)
    Eb_std_column = Eb_std.reshape(-1, 1)
    Tpmp_column = Tpmp.reshape(-1, 1)

    # Expected result:
    #     Eb_original.shape == (n_physical_features, n_samples)
    Eb_original = reverse_standardize(
        Eb,
        Eb_mean_column,
        Eb_std_column,
        method=method,
        epsilon=epsilon,
    )

    # This function should use only elementwise tensor operations so that
    # Tpmp_column broadcasts over all simulations.
    Tb_original = enthalpy_to_temperature(
        Eb_original,
        Tpmp_column,
    )

    # Broadcasting avoids constructing Tpmp with .repeat().
    delta_T = Tpmp_column - Tb_original

    return delta_T

def operator_temperature(Tb, Tpmp):
    """
    Observation operator directly on temperature in its physical unit and domain
    """
    return Tpmp - Tb

def temperature_binary_hard_operator(delta_T, dT_cutoff, mask=None):
    """
    A simple binary operator that classifies whether the base is thawed or frozen
    based on degree to melting point. Thresholds are user input to acknowledge 
    uncertainty in pre-melting or observations.
    
    Parameters
    ----------
    delta_T : torch.Tensor, (n_physical_feature, n_sample)
        Degree to melting point.
    dT_cutoff : float
        positive; threshold for classifying as thawed or frozen.
    mask : torch.Tensor, optional, (n_physical_feature,)
        Boolean mask indicating which elements to consider for classification. If None, all elements are considered.

    Returns
    -------
    binary_class : torch.Tensor
        Binary classification of thawed (1) or frozen (0) base based on the cutoff.
    """
    binary_class = torch.nan * torch.ones_like(delta_T)
    # if delta_T > dT_cutoff, it is frozen (0), since it is farther from PMP
    # else it is thawed 
    binary_class[delta_T > dT_cutoff] = 0 # frozen
    binary_class[delta_T <= dT_cutoff] = 1 # thawed
    if mask is not None:
        mask = torch.where(mask, torch.tensor(1), torch.nan)
        n_sample = delta_T.shape[1]
        for ii in range(n_sample):
            binary_class[:, ii] = binary_class[:, ii] * mask

    return binary_class


def temperature_binary_soft_operator(
    delta_T,
    beta,
    clamp_sharpness=20.0,
):
    """
    Soft binary classification operator for thawed and frozen base. 
    """
    if beta <= 0:
        raise ValueError("beta must be strictly positive.")

    if clamp_sharpness <= 0:
        raise ValueError(
            "clamp_sharpness must be strictly positive."
        )

    # Smooth approximation to max(delta_T, 0).
    positive_delta = (
        F.softplus(
            clamp_sharpness * delta_T
        )
        / clamp_sharpness
    )

    return torch.exp(-positive_delta / beta)

def waterfraction_binary_soft_operator(
    water_fraction,
    beta,
    wf_threshold = 0.02,
    eps = 0.01
):
    """
    Soft binary classification operator for water fraction 
    where we consider water fraction above a certain threshold
    to be physically unrealistic

    Parameters
    ----------
    water_fraction : torch.Tensor, (n_physical_feature, n_sample)
        Water fraction values.
    beta : float
        Positive; parameter controlling the softness of the classification.
    wf_threshold : float
        Threshold for water fraction above which it is considered physically unrealistic.
    eps : float
        Small positive value to avoid numerical issues.

    Returns
    -------
    soft_class : torch.Tensor
        Soft classification of water fraction, values between 0 and 1.
    """

    return 1 / (1 + torch.exp((1/beta) * (water_fraction - wf_threshold))) + eps
