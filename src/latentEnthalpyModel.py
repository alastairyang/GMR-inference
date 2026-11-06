from dataclasses import dataclass

import numpy as np
import torch

from observationOperator import (
    enthalpy_to_delta_temperature_operator,
    gp_latent_enthalpy_operator,
    pca_latent_enthalpy_operator,
)
from ice import enthalpy_to_water_fraction


def as_torch(x, *, dtype=torch.float64, device=None):
    """
    Convert the input to a PyTorch tensor if it is not already one.
    """
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.as_tensor(x, dtype=dtype, device=device)

class latentEnthalpyModel:

    def __init__(
        self,
        pca,
        gp,
        scaler,
        Tpmp,
        *,
        dtype=torch.float64,
        device=None
    ):
        self.dtype  = dtype
        self.device = device
        self.gp  = gp  # lowRankGP class
        self.pca = pca # sklearn PCA model

        self.Tpmp   = as_torch(
            Tpmp, 
            dtype=self.dtype, 
            device=self.device).reshape(-1, 1)
        self.scaler = scaler

        self.pca_eigenvec        = as_torch(
            self.pca.components_.T, 
            dtype=self.dtype, 
            device=self.device
        )
        self.gp_eigenvec         = as_torch(
            self.gp.eigenvec, 
            dtype=self.dtype, 
            device=self.device
        )
        self.gp_singular_val_mtx = as_torch(
            self.gp.singular_val_mtx, 
            dtype=self.dtype, 
            device=self.device
        )

    def to_tensor(self, x, *, requires_grad=False):
        x = as_torch(x, dtype=self.dtype, device=self.device)
        return x.clone().detach().requires_grad_(requires_grad)

    def decode_gp(self, z_gp):
        """
        Decode the latent Gaussian process representation into 
        the physical basal enthalpy field.
        """
        return gp_latent_enthalpy_operator(
            self.gp_eigenvec.T,
            self.gp_singular_val_mtx,
            z_gp,
        )

    def decode_gmm(self, z_gmm):
        """
        Decode the latent Gaussian mixture model + PCA latent
        representation into the physical basal enthalpy field.
        """
    
        return pca_latent_enthalpy_operator(
            self.pca_eigenvec.T,
            z_gmm,
        )

    def enthalpy(self, z_gp, z_gmm):
        """
        Construct the total physical basal enthalpy field
        Decode standardized enthalpy from latent representation,
        then reverse-standardize back to the physical enthalpy field.
        """
        # make z_gp and z_gmm two dimensional
        z_gp  = z_gp.reshape(z_gp.shape[0], -1)
        z_gmm = z_gmm.reshape(z_gmm.shape[0], -1)

        enthalpy_gp  = self.decode_gp(z_gp)
        enthalpy_gmm = self.decode_gmm(z_gmm)

        # enthalpy_total = enthalpy_gp + enthalpy_gmm
        e_mean = torch.as_tensor(
            self.scaler.mean_,
            dtype=enthalpy_gmm.dtype,
            device=enthalpy_gmm.device,
        )        
        e_scale = torch.as_tensor(
            self.scaler.scale_,
            dtype=enthalpy_gmm.dtype,
            device=enthalpy_gmm.device,
        )
        enthalpy_gmm_physical = enthalpy_gmm * e_scale[:, None] + e_mean[:, None]
        enthalpy_gp_shifted = enthalpy_gp

        enthalpy_total = enthalpy_gp_shifted + enthalpy_gmm_physical
        return enthalpy_total

    def physical_state(self, z_gp, z_gmm):
        """
        Construct physical basal thermal state.
        """
        Eb = self.enthalpy(z_gp, z_gmm)

        delta_T = enthalpy_to_delta_temperature_operator(
            Eb,
            self.Tpmp,
            istorch=True
        )

        water_fraction = enthalpy_to_water_fraction(
            Eb,
            self.Tpmp,
        )

        return {
            "enthalpy": Eb,
            "delta_T": delta_T,
            "water_fraction": water_fraction,
        }


class latentEnthalpyModelGMMOnly:
    """
    Bayesian inference over Gaussian Mixture Model latent representations learned over 
    the PCA-reduced latent space of the simulation ensemble. 

    This class contains the essential methods for computing log likelihood, prior, and
    posterior (As well as the gradients) of the latent representations.

    """
    def __init__(
        self,
        pca,
        scaler,
        Tpmp,
        *,
        dtype=torch.float64,
        device=None
    ):
        self.dtype  = dtype
        self.device = device
        self.pca = pca # sklearn PCA model

        self.Tpmp   = as_torch(
            Tpmp, 
            dtype=self.dtype, 
            device=self.device).reshape(-1, 1)
        self.scaler = scaler

        self.pca_eigenvec        = as_torch(
            self.pca.components_.T, 
            dtype=self.dtype, 
            device=self.device
        )

    def to_tensor(self, x, *, requires_grad=False):
        x = as_torch(x, dtype=self.dtype, device=self.device)
        return x.clone().detach().requires_grad_(requires_grad)

    def decode_gmm(self, z_gmm):
        """
        Decode the latent Gaussian mixture model + PCA latent
        representation into the physical basal enthalpy field.
        """
    
        return pca_latent_enthalpy_operator(
            self.pca_eigenvec.T,
            z_gmm,
        )

    def enthalpy(self, z_gmm):
        """
        Construct the total physical basal enthalpy field
        Decode standardized enthalpy from latent representation,
        then reverse-standardize back to the physical enthalpy field.
        """
        # make z_gmm two dimensional
        z_gmm = z_gmm.reshape(z_gmm.shape[0], -1)

        enthalpy_gmm = self.decode_gmm(z_gmm)

        e_mean = torch.as_tensor(
            self.scaler.mean_,
            dtype=enthalpy_gmm.dtype,
            device=enthalpy_gmm.device,
        )        
        e_scale = torch.as_tensor(
            self.scaler.scale_,
            dtype=enthalpy_gmm.dtype,
            device=enthalpy_gmm.device,
        )
        enthalpy_physical = enthalpy_gmm * e_scale[:, None] + e_mean[:, None]

        return enthalpy_physical

    def physical_state(self, z_gmm):
        """
        Construct physical basal thermal state.
        """
        Eb = self.enthalpy(z_gmm)

        delta_T = enthalpy_to_delta_temperature_operator(
            Eb,
            self.Tpmp,
            istorch=True
        )

        water_fraction = enthalpy_to_water_fraction(
            Eb,
            self.Tpmp,
        )

        return {
            "enthalpy": Eb,
            "delta_T": delta_T,
            "water_fraction": water_fraction,
        }


class latentEnthalpyModelGPOnly:
    """
    Bayesian inference over Gaussian Mixture Model latent representations learned over 
    the PCA-reduced latent space of the simulation ensemble. 

    This class contains the essential methods for computing log likelihood, prior, and
    posterior (As well as the gradients) of the latent representations.

    """
    def __init__(
        self,
        gp,
        scaler,
        Tpmp,
        *,
        dtype=torch.float64,
        device=None
    ):
        self.dtype  = dtype
        self.device = device
        self.gp  = gp  # lowRankGP class

        self.Tpmp   = as_torch(
            Tpmp, 
            dtype=self.dtype, 
            device=self.device).reshape(-1, 1)
        self.scaler = scaler

        self.gp_eigenvec         = as_torch(
            self.gp.eigenvec, 
            dtype=self.dtype, 
            device=self.device
        )
        self.gp_singular_val_mtx = as_torch(
            self.gp.singular_val_mtx, 
            dtype=self.dtype, 
            device=self.device
        )

    def to_tensor(self, x, *, requires_grad=False):
        x = as_torch(x, dtype=self.dtype, device=self.device)
        return x.clone().detach().requires_grad_(requires_grad)

    def decode_gp(self, z_gp):
        """
        Decode the latent Gaussian process representation into 
        the physical basal enthalpy field.
        """
        return gp_latent_enthalpy_operator(
            self.gp_eigenvec.T,
            self.gp_singular_val_mtx,
            z_gp,
        )

    def enthalpy(self, z_gp):
        """
        Construct the total physical basal enthalpy field
        Decode standardized enthalpy from latent representation,
        then reverse-standardize back to the physical enthalpy field.
        """
        # make z_gp and z_gmm two dimensional
        z_gp  = z_gp.reshape(z_gp.shape[0], -1)

        enthalpy_gp  = self.decode_gp(z_gp)

        # enthalpy_total = enthalpy_gp + enthalpy_gmm
        e_mean = torch.as_tensor(
            self.scaler.mean_,
        ).mean()        
        e_scale = torch.as_tensor(
            self.scaler.scale_,
        ).mean()
        enthalpy_gp_scaled = enthalpy_gp * e_scale + e_mean

        return enthalpy_gp_scaled

    def physical_state(self, z_gp):
        """
        Construct physical basal thermal state.
        """
        Eb = self.enthalpy(z_gp)

        delta_T = enthalpy_to_delta_temperature_operator(
            Eb,
            self.Tpmp,
            istorch=True
        )

        water_fraction = enthalpy_to_water_fraction(
            Eb,
            self.Tpmp,
        )

        return {
            "enthalpy": Eb,
            "delta_T": delta_T,
            "water_fraction": water_fraction,
        }


@dataclass
class BasalEvidence:
    Tpmp: torch.Tensor
    thawed_mask: torch.Tensor
    frozen_mask: torch.Tensor
    thawed_fraction: torch.Tensor
    frozen_fraction: torch.Tensor

    @classmethod
    def from_arrays(
        cls,
        Tpmp,
        thawed_mask,
        frozen_mask,
        dw,
        df,
        *,
        dtype=torch.float64,
        device=None,
    ):
        return cls(
            Tpmp=as_torch(
                Tpmp,
                dtype=dtype,
                device=device,
            ).reshape(-1, 1),

            thawed_mask=as_torch(
                thawed_mask,
                dtype=torch.bool,
                device=device,
            ).reshape(-1, 1),

            frozen_mask=as_torch(
                frozen_mask,
                dtype=torch.bool,
                device=device,
            ).reshape(-1, 1),

            thawed_fraction=as_torch(
                dw,
                dtype=dtype,
                device=device,
            ).reshape(-1, 1),

            frozen_fraction=as_torch(
                df,
                dtype=dtype,
                device=device,
            ).reshape(-1, 1),
        )


@dataclass
class PosteriorConfig:
    beta: float
    beta_w: float
    eps: float = 0.01
    water_fraction_threshold: float = 0.02