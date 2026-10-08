from numpy.random import beta
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

def gp_latent_enthalpy_operator(
    V,
    singular_val_mtx,
    xi_val,
    Eb_mean,
    Eb_std,
    method,
    epsilon=None,
):
    """
    Transforming latent Gaussian Process variable to physical enthalpy.
    """
    latent_input_check(V, xi_val, method=method, epsilon=epsilon)

    Eb = V.T @ (singular_val_mtx @ xi_val)

    Eb_mean_column = Eb_mean.reshape(-1, 1)
    Eb_std_column = Eb_std.reshape(-1, 1)

    Eb_physical = reverse_standardize(
        Eb,
        Eb_mean_column,
        Eb_std_column,
        method=method,
        epsilon=epsilon,
    )

    return Eb_physical

def pca_latent_enthalpy_operator(
    V,
    Eb_star,
    Eb_mean,
    Eb_std,
    method,
    epsilon=None,
):
    """
    Transforming latent PCA coefficients to physical enthalpy.
    """
    latent_input_check(V, Eb_star, method=method, epsilon=epsilon)

    Eb = V.T @ Eb_star

    Eb_mean_column = Eb_mean.reshape(-1, 1)
    Eb_std_column = Eb_std.reshape(-1, 1)

    Eb_physical = reverse_standardize(
        Eb,
        Eb_mean_column,
        Eb_std_column,
        method=method,
        epsilon=epsilon,
    )

    return Eb_physical

def enthalpy_to_delta_temperature_operator(
        Eb,
        Tpmp
):
    """
    Operator converting basal enthalpy to temperature to pressure melting [0,Tpmp) .

    """
    T = enthalpy_to_temperature(Eb, Tpmp)
    delta_T = Tpmp - T
    return delta_T


def pca_latent_temperature_operator(
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
    The operator returns the degree to pressure melting point

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

    Eb_physical = pca_latent_enthalpy_operator(
        V,
        Eb_star,
        Eb_mean,
        Eb_std,
        method=method,
        epsilon=epsilon,
    )

    Eb_original = Eb_physical
    Tpmp_column = Tpmp.reshape(-1, 1)

    delta_T = enthalpy_to_delta_temperature_operator(
        Eb_original, 
        Tpmp_column
    )

    return delta_T

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
    gamma,
    wf_threshold = 0.02,
):
    """
    Soft binary classification operator for water fraction 
    where we consider water fraction above a certain threshold
    to be physically unrealistic
    """
    if gamma <= 0:
        raise ValueError("gamma must be strictly positive.")

    normalized_exceedance = (water_fraction - wf_threshold) / gamma

    pointwise_prob = F.softplus(normalized_exceedance)

    return pointwise_prob