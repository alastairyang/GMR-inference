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

def enthalpy_to_water_fraction(Eb, Tpmp, Cp=2093.0, T0=223.15, istorch=True):
    """  
    Compute water fraction from enthalpy

    w = 0 if Eb <= Cp*(Tpmp-T0)
    w = (Eb - Cp*(Tpmp-T0))/(L + Cp*(Tpmp-T0)) if Eb > Cp*(Tpmp-T0)

    """
    L = 334000.0  # J/kg
    if istorch:
        w = torch.where(Eb <= Cp*(Tpmp-T0), 0.0, (Eb - Cp*(Tpmp-T0))/L)
    else: # numpy
        w = np.where(Eb <= Cp*(Tpmp-T0), 0.0, (Eb - Cp*(Tpmp-T0))/L)
    return w


def defineActivationEnergies(T):
    """
    Compute activation energies for creep, grain growth, and grain boundary mobility.

    Parameters:
        T : array-like or float — Temperature in Kelvin

    Returns:
        Qg : ndarray — Activation energy for grain growth (J/mol)
        Qc : ndarray — Activation energy for creep (J/mol)
        Qm : ndarray — Activation energy for grain boundary mobility (J/mol)
    """
    T    = np.atleast_1d(np.array(T, dtype=float))
    Temp = T - 273.0  # Convert to Celsius

    # --- Activation Energy for Creep ---
    tp, tm, tc = 0, -20, -10
    Qcp, Qcm   = 100.0, 60.0  # kJ/mol
    c1 = (Qcp - Qcm) / (np.arctan(tp - tc) - np.arctan(tm - tc))
    c2 = Qcp - c1 * np.tanh(tp - tc)

    # --- Activation Energy for Grain Growth ---
    tp, tm, tc = 0, -20, -10
    Qgp, Qgm   = 100.0, 40.0  # kJ/mol
    g1 = (Qgp - Qgm) / (np.arctan(tp - tc) - np.arctan(tm - tc))
    g2 = Qgp - g1 * np.tanh(tp - tc)

    # --- Activation Energy for Grain Boundary Mobility ---
    tp, tm, tc = 0, -20, -10
    Qmp, Qmm   = 40.0, 100.0  # kJ/mol
    m1 = (Qmp - Qmm) / (np.arctan(tp - tc) - np.arctan(tm - tc))
    m2 = Qmp - m1 * np.arctan(tp - tc)

    # --- Compute outputs ---
    Qg = (g1 * np.arctan(Temp - tc) + g2) * 1e3  # J/mol
    Qc = (c1 * np.arctan(Temp - tc) + c2) * 1e3  # J/mol
    Qm = (m1 * np.arctan(Temp - tc) + m2) * 1e3  # J/mol

    return Qg, Qc, Qm


def computeGlenFlowRateParameter(T):
    """
    Compute the Glen flow rate parameter A (Arrhenius-type).

    Parameters:
        T : array-like or float — Temperature in Kelvin

    Returns:
        Aglen : ndarray — Glen flow rate parameter (Pa^-3 s^-1)
    """
    T = np.atleast_1d(np.array(T, dtype=float))

    R = 8.314  # J/(mol·K)

    Qg, Qc, Qm = defineActivationEnergies(T)

    # Reference pre-exponential factor (normalised to 263 K)
    A0 = 2.4e-24 / np.exp(-(115000.0 / R) * ((1.0 / 273.0) - (1.0 / 263.0)))

    Aglen = A0 * np.exp(-(Qc / R) * ((1.0 / T) - (1.0 / 263.0)))

    return Aglen

def driving_stress(alpha, H):
    rho_i = 917  # kg/m³
    g = 9.81     # m/s²
    return rho_i * g * H * alpha

def delta_driving_stress(delta_alpha, H):
    rho_i = 917  # kg/m³
    g = 9.81     # m/s²
    return rho_i * g * H * delta_alpha

def strain_heating(alpha, H, z, n=3, T=250):
    """
    Ice strain heating due to vertical shear deformation.
    
    Parameters:
    alpha : array-like or float — Surface slope (dimensionless)
    H     : array-like or float — Ice thickness (m)
    z     : array-like or float — [0, H] Vertical coordinate (m)
    n     : int — Glen's flow law exponent (default: 3)
    T     : array-like or float — Temperature in Kelvin (default: 250 K)
    """
    rho_i = 917  # kg/m³
    g = 9.81     # m/s²
    A = computeGlenFlowRateParameter(T)
    return 2 *A * (rho_i * g * (H-z) * alpha)**(n+1)

