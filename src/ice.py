import numpy as np
import torch
def compute_pmp(H):
    """  
    compute pressure melting point
    """
    rho_i = 917.0
    g = 9.81
    beta=9.8e-8
    return 273.15 - rho_i * g * H * beta 

def enthalpy_to_temperature(Eb, Tpmp, Cp=2093.0, T0=223.15, istorch=True):
    """  
    Compute temperature from enthalpy

    Tb = Tpmp if Eb > Cp*(Tpmp-T0)
    Tb = (Eb + Cp*T0)/Cp if Eb <= Cp*(Tpmp-T0)

    """

    if istorch:
        Tb = torch.where(Eb > Cp*(Tpmp-T0), Tpmp, (Eb + Cp*T0)/Cp)
    else: # numpy
        Tb = np.where(Eb > Cp*(Tpmp-T0), Tpmp, (Eb + Cp*T0)/Cp)
    return Tb

