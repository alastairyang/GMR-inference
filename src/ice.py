import numpy as np
import torch
from scipy.optimize import brentq

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

def shallow_ice(H, alpha, T, E=1):
    """
    Compute the ice surface velocity using the shallow ice approximation (SIA).
    
    Parameters:
    H     : array-like or float — Ice thickness (m)
    alpha : array-like or float — Surface slope (dimensionless)
    T     : array-like or float — Temperature in Kelvin
    E     : float — Enhancement factor (default: 1)
    """
    
    rho_i = 917  # kg/m³
    g = 9.81     # m/s²
    n = 3        # Glen's flow law exponent
    A = computeGlenFlowRateParameter(T)
    return E * (2 * A / (n + 1)) * (rho_i * g * H * alpha)**n * H


    import numpy as np

def temperature_to_conductivity(T, sigma0=6.6e-6, Epure=None, E_Hp=None,
                                 E_ssCl=None, mu_Hp=3.2, mu_ssCl=0.43,
                                 molar_ssCl=4.2e-6, molar_Hp=2.7e-6):
    """
    Calculate ice conductivity from temperature assuming Arrhenius relationship.

    Parameters
    ----------
    T : float or np.ndarray
        Temperature in Kelvin.
    sigma0 : float, optional
        Default 6.6e-6.
    Epure : float, optional
        Activation energy for pure ice (J). Default 0.55 eV.
    E_Hp : float, optional
        Activation energy for H+ (J). Default 0.20 eV.
    E_ssCl : float, optional
        Activation energy for ss-Cl (J). Default 0.19 eV.
    mu_Hp : float, optional
        Default 3.2.
    mu_ssCl : float, optional
        Default 0.43.
    molar_ssCl : float, optional
        Molar concentration of ss-Cl (mol). Default 4.2e-6.
    molar_Hp : float, optional
        Molar concentration of H+ (mol). Default 2.7e-6.
        (Corrected value from Table 2 of MacGregor 2007.)

    Returns
    -------
    sigma : float or np.ndarray
        Conductivity (S/m).

    References
    ----------
    MacGregor (2007), Table 1 & 2.
    """
    T = np.asarray(T, dtype=float)

    Tr  = 251.0          # Reference temperature (K)
    k   = 1.380e-23      # Boltzmann's constant (J/K)
    eV  = 1.602176634e-19  # Joules per eV

    # Apply defaults in eV → J
    if Epure  is None: Epure  = 0.55 * eV
    if E_Hp   is None: E_Hp   = 0.20 * eV
    if E_ssCl is None: E_ssCl = 0.19 * eV

    arrhenius = lambda E: np.exp((E / k) * (1.0 / Tr - 1.0 / T))

    sigma_ice  = sigma0    * arrhenius(Epure)
    sigma_Hp   = mu_Hp     * molar_Hp   * arrhenius(E_Hp)
    sigma_ssCl = mu_ssCl   * molar_ssCl * arrhenius(E_ssCl)

    return sigma_ice + sigma_Hp + sigma_ssCl


def conductivity_to_attenu_rate(sigma):
    """
    Calculate one-way attenuation rate from a conductivity profile.

    Parameters
    ----------
    sigma : float or np.ndarray
        Conductivity (S/m).

    Returns
    -------
    N : float or np.ndarray
        Attenuation rate (dB/km).
    """
    sigma = np.asarray(sigma, dtype=float)

    c    = 3e8           # Speed of light (m/s)
    eps0 = 8.854e-12     # Permittivity of free space (F/m)
    epsr = 3.17          # Real relative permittivity of ice

    N = 1000 * (10 * np.log10(np.exp(1))) * sigma / (c * eps0 * np.sqrt(epsr))

    return N


def attenu_rate_to_temperature(N, T_bounds=(200.0, 273.15), **kwargs):
    """
    Solve for ice temperature T (K) given a target attenuation rate N (dB/km).

    Inverts the chain:
        T  -->  temperature_to_conductivity()  -->  conductivity_to_attenu_rate()  -->  N

    Uses Brent's method (bracketed root-finding) — robust and derivative-free.

    Parameters
    ----------
    N : float or array-like
        Target attenuation rate(s) in dB/km.
    T_bounds : tuple of float, optional
        (T_min, T_max) search bracket in Kelvin. Default (200, 273.15).
        Must bracket the root, i.e. N(T_min) < target < N(T_max).
    **kwargs
        Any keyword argument accepted by temperature_to_conductivity()
        (e.g. sigma0, molar_Hp, Epure, ...).

    Returns
    -------
    T_solved : float or np.ndarray
        Temperature(s) in Kelvin corresponding to each input N value.

    Raises
    ------
    ValueError
        If the target N lies outside the physically reachable range within
        T_bounds, or if the bracket does not contain a sign change.

    Notes
    -----
    N is a monotonically increasing function of T, so a unique solution
    always exists within a valid bracket.
    """
    N = np.asarray(N, dtype=float)
    scalar_input = N.ndim == 0
    N = np.atleast_1d(N)

    T_min, T_max = T_bounds

    # Pre-check bracket validity once (cheap)
    N_min = conductivity_to_attenu_rate(temperature_to_conductivity(T_min, **kwargs))
    N_max = conductivity_to_attenu_rate(temperature_to_conductivity(T_max, **kwargs))

    if not (N_min < N_max):
        raise ValueError(
            f"Bracket check failed: N({T_min} K) = {N_min:.4f}, "
            f"N({T_max} K) = {N_max:.4f}. Expected N_min < N_max."
        )

    def _residual(T_scalar, target):
        sigma = temperature_to_conductivity(T_scalar, **kwargs)
        return conductivity_to_attenu_rate(sigma) - target

    T_solved = np.empty_like(N)

    for i, Ni in enumerate(N):
        if not (N_min <= Ni <= N_max):
            raise ValueError(
                f"Target N = {Ni:.4f} dB/km is outside the reachable range "
                f"[{N_min:.4f}, {N_max:.4f}] dB/km for T in {T_bounds} K.\n"
                f"Widen T_bounds or check your input."
            )
        T_solved[i] = brentq(_residual, T_min, T_max, args=(Ni,), xtol=1e-6, rtol=1e-9)

    return float(T_solved[0]) if scalar_input else T_solved
