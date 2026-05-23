import torch
import numpy as np

from src.optimization import log_posterior, log_posterior_gradient

class custom_energy(torch.autograd.Function):
    """
    Bridges the pure PyTorch/NumPy hybrid log-posterior directly into Pyro.
    """
    @staticmethod
    def forward(ctx, z, V, gmm, beta, beta_w, Tpmp, Eb_mean, Eb_std, dw, df, Eb_epsilon):
        ctx.save_for_backward(z)
        ctx.V = V
        ctx.gmm = gmm
        ctx.beta = beta
        ctx.beta_w = beta_w
        ctx.Tpmp = Tpmp
        ctx.Eb_mean = Eb_mean
        ctx.Eb_std = Eb_std
        ctx.dw = dw
        ctx.df = df
        ctx.Eb_epsilon = Eb_epsilon
        
        z_np = z.detach().cpu().numpy()
        
        with torch.no_grad():
            lp = log_posterior(z_np, V, gmm, beta, beta_w, Tpmp, Eb_mean, Eb_std, dw, df, Eb_epsilon=Eb_epsilon)
            if isinstance(lp, torch.Tensor):
                lp = lp.item()
                
        return torch.tensor(-lp, dtype=z.dtype, device=z.device)

    @staticmethod
    def backward(ctx, grad_output):
        z, = ctx.saved_tensors
        z_np = z.detach().cpu().numpy()
        
        # CRITICAL FIX: Re-enable gradient tracking so the internal AD graph can build!
        with torch.enable_grad():
            grad_np = log_posterior_gradient(
                z_np, ctx.V, ctx.gmm, ctx.beta, ctx.beta_w, ctx.Tpmp, 
                ctx.Eb_mean, ctx.Eb_std, ctx.dw, ctx.df, ctx.Eb_epsilon
            )
        
        grad_potential = -grad_np
        grad_tensor = torch.tensor(grad_potential, dtype=z.dtype, device=z.device)
        
        return grad_tensor * grad_output, None, None, None, None, None, None, None, None, None, None
    
class whitened_potential:
    """ 
    Apply a coordinate transformation to Hamiltonian Monte Carlo (HMC) to 
    make posterior near MAP more isotropic. 
    """
    def __init__(self, X_opt_tensor, L, V, gmm, beta,beta_w,
                       Tpmp, Eb_mean, Eb_std, dw, df, Eb_epsilon):
        self.X_opt_tensor = X_opt_tensor
        self.L = L
        self.V = V
        self.gmm = gmm
        self.beta = beta
        self.beta_w = beta_w
        self.Tpmp = Tpmp
        self.Eb_mean = Eb_mean
        self.Eb_std = Eb_std
        self.dw = dw
        self.df = df
        self.Eb_epsilon = Eb_epsilon

    def __call__(self, params_dict):
        u = params_dict["u"]
        z = self.X_opt_tensor + torch.matmul(self.L, u)

        return custom_energy.apply(
            z, self.V, self.gmm, self.beta, self.beta_w, self.Tpmp,
            self.Eb_mean, self.Eb_std, self.dw, self.df, self.Eb_epsilon
        )
    
class regular_potential:
    """
    Direct potential in z-space (no whitening/Hessian preconditioning).
    Used for exploratory runs where the posterior landscape is unknown.
    """
    def __init__(self, V, gmm, beta, beta_w, Tpmp, Eb_mean, Eb_std, dw, df, Eb_epsilon):
        self.V = V
        self.gmm = gmm
        self.beta = beta
        self.beta_w = beta_w
        self.Tpmp = Tpmp
        self.Eb_mean = Eb_mean
        self.Eb_std = Eb_std
        self.dw = dw
        self.df = df
        self.Eb_epsilon = Eb_epsilon

    def __call__(self, params_dict):
        z = params_dict["z"]
        return custom_energy.apply(
            z, self.V, self.gmm, self.beta, self.beta_w, self.Tpmp,
            self.Eb_mean, self.Eb_std, self.dw, self.df, self.Eb_epsilon
        )
