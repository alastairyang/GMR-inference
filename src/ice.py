import numpy as np

def compute_pmp(H):
    """  
    compute pressure melting point
    """
    rho_i = 917.0
    g = 9.81
    beta=9.8e-8
    return 273.15 - rho_i * g * H * beta 