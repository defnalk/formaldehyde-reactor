"""
reactor/kinetics.py
-------------------
Kinetic models for methanol partial oxidation to formaldehyde.

Reaction 1 (desired):   CH₃OH + ½O₂  →  HCHO + H₂O
Reaction 2 (undesired): HCHO  + ½O₂  →  CO   + H₂O

Two models implemented:

1. Power-Law — empirical, fit by regression to experimental data:
       R_HCHO = k · [CH₃OH]^0.8742 · [O₂]^0.1124 · [H₂O]^-0.4858

2. LHHW (Langmuir-Hinshelwood-Hougen-Watson) — mechanistic, from
   Deshmukh, Annaland & Kuipers (2005). Accounts for:
     - Competitive adsorption of CH₃OH and H₂O on catalyst sites
     - Surface saturation at high methanol concentrations
     - Temperature-dependent adsorption equilibrium constants

Reference: Group 20 Reaction Engineering Report, §2.1
           Deshmukh et al., Chem. Eng. Sci. (2005)
"""

import numpy as np

R_GAS = 8.314e-3  # kJ/(mol·K)

# ── LHHW Arrhenius parameters (Deshmukh et al. 2005) ─────────────────────────
# Format: A0 (pre-exponential), Ea (kJ/mol)
LHHW_PARAMS = {
    "k_MeOH":   {"A0": 1.7340e11,  "Ea":  93.9},   # mol kg⁻¹ s⁻¹ atm⁻²
    "k_CO":     {"A0": 5.0600e4,   "Ea":  73.2},   # mol kg⁻¹ s⁻¹ atm⁻²
    "K_MeOH":   {"A0": 3.5200e-8,  "Ea": -70.0},   # atm⁻¹  (adsorption: negative Ea)
    "K_O2":     {"A0": 1.3300e-4,  "Ea": -20.0},   # atm⁻⁰·⁵
    "K_H2O":    {"A0": 1.8400e-6,  "Ea": -60.0},   # atm⁻¹
}
ALPHA = 0.44    # fraction of catalyst containing active compounds


def arrhenius(A0: float, Ea: float, T: float) -> float:
    """
    Evaluate the Arrhenius expression.

    k(T) = A0 · exp(−Ea / RT)

    Parameters
    ----------
    A0 : float
        Pre-exponential factor.
    Ea : float
        Activation energy (kJ/mol). Use negative for adsorption constants.
    T : float
        Temperature (K).

    Returns
    -------
    float
        Rate or equilibrium constant at temperature T.
    """
    return A0 * np.exp(-Ea / (R_GAS * T))


class PowerLawKinetics:
    """
    Power-law kinetic model for methanol oxidation.

    Reaction rates expressed in mol kg⁻¹ s⁻¹ as functions of
    molar concentrations (mol m⁻³).

    Exponents fitted by regression to experimental data at 523 K.

    Parameters
    ----------
    A0 : float
        Pre-exponential factor for rate constant k.
    Ea : float
        Activation energy (kJ/mol).

    Examples
    --------
    >>> kin = PowerLawKinetics()
    >>> C = {"MeOH": 2.0, "O2": 1.0, "H2O": 0.5, "HCHO": 0.0, "CO": 0.0}
    >>> r1, r2 = kin.rates(C, T=640)
    """

    # Exponents from regression (eq. 3 in report)
    N_MEOH = 0.8742
    N_O2   = 0.1124
    N_H2O  = -0.4858

    def __init__(self, A0: float = 3.15e7, Ea: float = 90.0):
        self.A0 = A0
        self.Ea = Ea

    def rate_constant(self, T: float) -> float:
        """Rate constant k(T) via Arrhenius."""
        return arrhenius(self.A0, self.Ea, T)

    def rates(
        self, C: dict[str, float], T: float
    ) -> tuple[float, float]:
        """
        Compute reaction rates for both reactions.

        Parameters
        ----------
        C : dict
            Molar concentrations (mol m⁻³):
            keys 'MeOH', 'O2', 'H2O', 'HCHO', 'CO', 'N2'
        T : float
            Temperature (K).

        Returns
        -------
        r1, r2 : float
            Rate of reaction 1 (HCHO formation) and reaction 2 (CO formation)
            in mol kg⁻¹ s⁻¹.
        """
        k = self.rate_constant(T)
        C_MeOH = max(C["MeOH"], 1e-10)
        C_O2   = max(C["O2"],   1e-10)
        C_H2O  = max(C["H2O"],  1e-10)
        C_HCHO = max(C.get("HCHO", 0), 1e-10)

        r1 = k * C_MeOH**self.N_MEOH * C_O2**self.N_O2 * C_H2O**self.N_H2O
        # Secondary reaction — approximate as power law on HCHO
        r2 = 0.01 * k * C_HCHO * C_O2**0.5

        return max(r1, 0.0), max(r2, 0.0)


class LHHWKinetics:
    """
    Langmuir-Hinshelwood-Hougen-Watson (LHHW) kinetic model.

    Mechanistic model accounting for competitive adsorption of
    CH₃OH and H₂O on catalyst active sites, and surface reaction
    kinetics (Deshmukh, Annaland & Kuipers, 2005).

    Rates are in mol kg⁻¹ s⁻¹ as functions of partial pressures (atm).

    Parameters
    ----------
    alpha : float
        Fraction of catalyst mass containing active compounds. Default 0.44.

    Examples
    --------
    >>> kin = LHHWKinetics()
    >>> P = {"MeOH": 0.18, "O2": 0.096, "H2O": 0.048, "HCHO": 0.0, "CO": 0.0}
    >>> r1, r2 = kin.rates(P, T=640)
    """

    def __init__(self, alpha: float = ALPHA):
        self.alpha = alpha

    def _constants(self, T: float) -> dict[str, float]:
        """Evaluate all Arrhenius constants at temperature T."""
        return {key: arrhenius(**vals, T=T) for key, vals in LHHW_PARAMS.items()}

    def rates(
        self, P: dict[str, float], T: float
    ) -> tuple[float, float]:
        """
        Compute LHHW reaction rates.

        Parameters
        ----------
        P : dict
            Partial pressures (atm):
            keys 'MeOH', 'O2', 'H2O', 'HCHO', 'CO', 'N2'
        T : float
            Temperature (K).

        Returns
        -------
        r1, r2 : float
            Rate of HCHO formation and CO formation (mol kg⁻¹ s⁻¹).
        """
        c = self._constants(T)

        P_MeOH = max(P["MeOH"], 1e-12)
        P_O2   = max(P["O2"],   1e-12)
        P_H2O  = max(P.get("H2O", 0), 1e-12)
        P_HCHO = max(P.get("HCHO", 0), 1e-12)

        # Adsorption denominator terms (eq. 4 in report)
        denom_surf = (
            1
            + c["K_MeOH"] * P_MeOH
            + c["K_H2O"]  * P_H2O
        )
        denom_O2 = 1 + c["K_O2"] * P_O2**0.5

        # Reaction 1: CH₃OH oxidation to HCHO (eq. 4)
        r1 = (
            self.alpha
            * c["k_MeOH"] * c["K_MeOH"] * P_MeOH * c["K_O2"] * P_O2**0.5
            / (denom_surf * denom_O2)
        )

        # Reaction 2: HCHO oxidation to CO (eq. 5)
        r2 = (
            self.alpha
            * c["k_CO"] * P_HCHO * c["K_O2"] * P_O2**0.5
            / denom_O2
        )

        return max(r1, 0.0), max(r2, 0.0)

    def selectivity_temperature_sensitivity(
        self, T_range: np.ndarray, P_inlet: dict[str, float]
    ) -> np.ndarray:
        """
        Compute HCHO selectivity as a function of temperature at fixed composition.

        Parameters
        ----------
        T_range : np.ndarray
            Temperature values (K).
        P_inlet : dict
            Inlet partial pressures (atm).

        Returns
        -------
        np.ndarray
            Selectivity S = r1 / (r1 + r2) at each temperature.
        """
        selectivities = []
        for T in T_range:
            r1, r2 = self.rates(P_inlet, T)
            total = r1 + r2
            selectivities.append(r1 / total if total > 0 else 1.0)
        return np.array(selectivities)
