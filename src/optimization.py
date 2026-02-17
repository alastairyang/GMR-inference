import numpy as np

# functions related to finding Maximum A Posteriori (MAP) in the latent space of a trained GMR model

def dTb_dEb(Eb, Tpmp, Cp=2090.0):
    """  
    Compute a smooth gradient from enthalpy to temperature.

    Tb = Tpmp if Eb > Cp*Tpmp
    Tb = (Eb + Cp*To)/Cp if Eb <= Cp*Tpmp

    similar to differentiating a ReLU function, 
    we can compute the gradient dTb/dEb as follows:
    dTb/dEb = 0 if Eb > Cp*Tpmp
    dTb/dEb = 1/Cp if Eb <= Cp*Tpmp
    
    """
    if Eb > Cp * Tpmp:
        return 0.0
    else:
        return 1.0 / Cp

def dL1_dEb(beta, Eb, Tpmp, dw, Cp=2090.0):
    """  
    Compute the gradient of log likelihood 1 (the thawed base evidence) with respect to the
      latent enthalpy at base (Eb asterisk)

    dL1/dEb* = -(1/beta) * \Sum_i^N dTb/dEb * dw

    Parameters:
    -------
    beta: scalar
        temperature scale (K) for the exponential parameterization
    Eb: array
        basal enthalpy
    Tpmp: array
        pressure melting point 
    dw: array
        area of thawed base at each pixel
    Cp: scalar
        specific heat capacity of ice (default 2090 J/kg/K)

    Returns:
    -------
    dL1_dEb: array
        gradient of log likelihood 1 with respect to Eb
    """
    return -(1.0/beta) * np.sum(dTb_dEb(Eb, Tpmp, Cp) * dw)

def dEb_dEbstar(dL1_dEb, V):
    """
    Gradient of enthalpy at base Eb wrt its latent values.
    For PCA-based model reduction, it's just the right singular vectors

    since dL1_dEb is a scalar, whereas V is (n_features, n_components),
    we augment dL1_dEb to be (n_features,) by repeating it, 
    then multiply with V to get the gradient in the latent space (n_components,)
    """
    dL1_dEb = np.full(V.shape[0], dL1_dEb)  # shape (n_features,)
    return dL1_dEb @ V