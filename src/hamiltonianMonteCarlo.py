import torch
import numpy as np

class HMCenergy(torch.autograd.Function):
    """
    Bridge the torch log posterior and Pyro. Inherited from torch.autograd.Function.
    """

    @staticmethod
    def forward(ctx, z_gp, z_gmm, posterior):
        ctx.save_for_backward(z_gp, z_gmm)
        ctx.posterior = posterior

        z_gp_np  = z_gp.detach().cpu().numpy()
        z_gmm_np = z_gmm.detach().cpu().numpy()

        with torch.no_grad():
            lp = ctx.posterior.log_prob(
                z_gp_np,
                z_gmm_np,
        )

            if not np.isfinite(lp):
                raise ValueError(f"Non-finite log posterior encountered: {lp}")

            if isinstance(lp, torch.Tensor):
                lp = lp.item()

        return torch.tensor(-lp, dtype=z_gp.dtype, device=z_gp.device)


    @staticmethod
    def backward(ctx, grad_output):
        z_gp, z_gmm = ctx.saved_tensors
        z_gp_np  = z_gp.detach().cpu().numpy()
        z_gmm_np = z_gmm.detach().cpu().numpy()

        with torch.enable_grad():
            grad_gp, grad_gmm = ctx.posterior.gradient(
                z_gp_np,
                z_gmm_np,
            )
            grad_gp_potential = torch.as_tensor(
                    grad_gp,
                    dtype=z_gp.dtype,
                    device=z_gp.device,
                ).reshape_as(z_gp)

            grad_gmm_potential = torch.as_tensor(
                    grad_gmm,
                    dtype=z_gmm.dtype,
                    device=z_gmm.device,
                ).reshape_as(z_gmm)

        return -grad_output * grad_gp_potential, -grad_output * grad_gmm_potential, None


class RegularPotential:
    """
    No preconditioning nor whitening is applied to the HMC variable, hence
    regular potential
    """

    def __init__(self, posterior):
        self.posterior = posterior

    def __call__(self, params_dict):
        z_gp, z_gmm = params_dict["z_gp"], params_dict["z_gmm"]
        return HMCenergy.apply(z_gp, z_gmm, self.posterior)
