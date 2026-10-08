from dataclasses import dataclass
import numpy as np
import torch
from src.observationOperator import enthalpy_to_delta_temperature_operator
from src.observationOperator import gp_latent_enthalpy_operator, pca_latent_enthalpy_operator
from src.ice import enthalpy_to_water_fraction 

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
        Eb_mean,
        Eb_std,
        Eb_epsilon,
        method,
        Tpmp,
        *,
        dtype=torch.float64,
        device=None
    ):
        self.dtype  = dtype
        self.device = device
        self.gp  = gp  # lowRankGP class
        self.pca = pca # sklearn PCA model
        self.Eb_mean    = as_torch(Eb_mean, dtype=self.dtype, device=self.device)
        self.Eb_std     = as_torch(Eb_std, dtype=self.dtype, device=self.device)
        self.Eb_epsilon = as_torch(Eb_epsilon, dtype=self.dtype, device=self.device)
        self.Tpmp       = as_torch(Tpmp, dtype=self.dtype, device=self.device)  
        if method not in ["relaxation", "standard"]:
            raise ValueError(f"Unsupported method: {method}")
        else:
            self.method     = method

    def to_tensor(self, x, *, requires_grad=False):
        x = as_torch(x, dtype=self.dtype, device=self.device)
        return x.clone().detach().requires_grad_(requires_grad)

    def decode_gp(self, z_gp):
        """
        Decode the latent Gaussian process representation into 
        the physical basal enthalpy field.
        """
        return gp_latent_enthalpy_operator(
            self.gp.eigenvec.T,
            self.gp.singular_val_mtx,
            z_gp,
            self.Eb_mean,
            self.Eb_std,
            method=self.method,
            epsilon=self.Eb_epsilon,
        )

    def decode_gmm(self, z_gmm):
        """
        Decode the latent Gaussian mixture model + PCA latent
        representation into the physical basal enthalpy field.
        """
        return pca_latent_enthalpy_operator(
            self.pca.components_.T,
            z_gmm,
            self.Eb_mean,
            self.Eb_std,
            method=self.method,
            epsilon=self.Eb_epsilon,
        )

    def enthalpy(self, z_gp, z_gmm):
        """
        Construct the total physical basal enthalpy field
        """
        return self.decode_gp(z_gp) + self.decode_gmm(z_gmm)

    def physical_state(self, z_gp, z_gmm):
        """
        Construct the physical basal thermal state, including delta temperature (w.r.t. Tpmp),
        enthalpy, and water fraction.
        
        """
        Eb = self.enthalpy(z_gp, z_gmm)

        delta_T = enthalpy_to_delta_temperature_operator(
            Eb,
            self.Tpmp,
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
class basalEvidence:
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
            Tpmp=as_torch(Tpmp, dtype=dtype, device=device),
            thawed_mask=as_torch(thawed_mask, dtype=dtype, device=device),
            frozen_mask=as_torch(frozen_mask, dtype=dtype, device=device),
            thawed_fraction=as_torch(dw, dtype=dtype, device=device),
            frozen_fraction=as_torch(df, dtype=dtype, device=device),
        )

@dataclass(frozen=True)
class PosteriorConfig:
    beta: float
    beta_w: float
    eps: float = 0.01
    water_fraction_threshold: float = 0.02