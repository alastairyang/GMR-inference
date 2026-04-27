import torch
import numpy as np

from src.optimization import log_posterior, log_posterior_gradient

class CustomEnergy(torch.autograd.Function):
    """
    Bridges the pure PyTorch/NumPy hybrid log-posterior directly into Pyro.
    """
    @staticmethod
    def forward(ctx, z, V, gmm, beta, Tpmp, Eb_mean, Eb_std, dw, df, Eb_epsilon):
        ctx.save_for_backward(z)
        ctx.V = V
        ctx.gmm = gmm
        ctx.beta = beta
        ctx.Tpmp = Tpmp
        ctx.Eb_mean = Eb_mean
        ctx.Eb_std = Eb_std
        ctx.dw = dw
        ctx.df = df
        ctx.Eb_epsilon = Eb_epsilon
        
        z_np = z.detach().cpu().numpy()
        
        with torch.no_grad():
            lp = log_posterior(z_np, V, gmm, beta, Tpmp, Eb_mean, Eb_std, dw, df, Eb_epsilon=Eb_epsilon)
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
                z_np, ctx.V, ctx.gmm, ctx.beta, ctx.Tpmp, 
                ctx.Eb_mean, ctx.Eb_std, ctx.dw, ctx.df, ctx.Eb_epsilon
            )
        
        grad_potential = -grad_np
        grad_tensor = torch.tensor(grad_potential, dtype=z.dtype, device=z.device)
        
        return grad_tensor * grad_output, None, None, None, None, None, None, None, None, None
    