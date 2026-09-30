import torch
import numpy as np
from src.ice import enthalpy_to_temperature
from src.utilities import reverse_standardize

# this script defines the observation operator
# which operators on the simulation output (enthalpy or basal temperature)
# to produce binary features of thawed or frozen base

# We write everything in torch for AD uses 

def latent_operator_enthalpy(V, Eb_star, Eb_mean, Eb_std, Tpmp, method, epsilon=None):
    """  
    Observation operator, operating on latent PCA coefficients of standized enthalpy.

    This operator only produces degree to melting point, which is interpretable
    to both thawed and frozen states. The downstream applications (E.g. adding
    a likelihood model, or just thresholding for a binary outcome) are specified 
    elsewhere such that this is meant to be kept modular and reusable.
    
    Parameters
    ----------
    V : torch.Tensor, (n_latent_feature, n_physical_feature)
        PCA components matrix.
    Eb_star : torch.Tensor, (n_latent_feature, n_sample)
        Latent PCA coefficients.
    Eb_mean : torch.Tensor (n_physical_feature,)
        Mean of the original enthalpy data.
    Eb_std : torch.Tensor, (n_physical_feature,)
        Standard deviation of the original enthalpy data.
    Tpmp : torch.Tensor, (n_physical_feature,)
        Pressure melting point.
    method : str
        Method for reverse standardization.
    epsilon : float, optional
        Small value for relaxation method.
    
    """

    if epsilon is None and method == "relaxation":
        # error
        raise ValueError("Epsilon must be provided for relaxation method")
    
    Eb = V.T @ Eb_star

    n_sample = Eb_star.shape[1]

    Tb_ori = torch.zeros_like(Eb)
    for ii in range(n_sample):
        Eb_ori = reverse_standardize(Eb[:, ii], Eb_mean, Eb_std, 
                                           method=method, epsilon=epsilon)

        Tb_ori[:, ii] = enthalpy_to_temperature(Eb_ori, Tpmp)

    Tpmp_array = Tpmp.unsqueeze(1).repeat(1, n_sample)
    delta_T = operator_temperature(Tb_ori, Tpmp_array)
    return delta_T

def operator_temperature(Tb, Tpmp):
    """
    Observation operator directly on temperature in its physical unit and domain
    """
    return Tpmp - Tb

def binary_operator(delta_T, dT_cutoff, mask=None):
    """
    A simple binary operator that classifies whether the base is thawed or frozen
    based on degree to melting point. Thresholds are user input to acknowledge 
    uncertainty in pre-melting or observations.
    
    Parameters
    ----------
    delta_T : torch.Tensor
        Degree to melting point.
    dT_cutoff : float
        positive; threshold for classifying as thawed or frozen.
    mask : torch.Tensor, optional
        Boolean mask indicating which elements to consider for classification. If None, all elements are considered.

    Returns
    -------
    binary_class : torch.Tensor
        Binary classification of thawed (1) or frozen (0) base based on the cutoff.
    """
    binary_class = torch.nan * torch.ones_like(delta_T)
    # if delta_T > dT_cutoff, it is frozen (0), since it is farther from PMP
    # else it is thawed 
    binary_class[delta_T > dT_cutoff] = 0
    binary_class[delta_T <= dT_cutoff] = 1
    if mask is not None:
        mask = torch.where(mask, torch.tensor(1), torch.nan)
        mask_full = torch.tile(mask, (delta_T.shape[0], 1))
        binary_class = binary_class * mask_full

    return binary_class
