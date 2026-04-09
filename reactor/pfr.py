"""
reactor/pfr.py
--------------
Packed-bed plug-flow reactor (PFR) simulation.

Solves the coupled ODE system for:
  - Material balances (eq. 9):   dn_i/dz = (ν₁ᵢ r₁ + ν₂ᵢ r₂) · Ac · (1−ε) · ρ_cat
  - Energy balance (eq. 14/15):  dT/dz   = f(ΔHr, ṁ, Cp)
  - Pressure drop (eq. 17):      dP/dz   = Ergun equation

Species index: 0=CH₃OH, 1=O₂, 2=H₂O, 3=HCHO, 4=CO, 5=N₂

Reference: Group 20 Reaction Engineering Report, §2.2–2.4
"""

import numpy as np
from scipy.integrate import solve_ivp

from .kinetics import LHHWKinetics, PowerLawKinetics

# ── Reactor geometry & packing ─────────────────────────────────────────────────
TUBE_DIAMETER   = 0.020          # m
TUBE_RADIUS     = TUBE_DIAMETER / 2
TUBE_AREA       = np.pi * TUBE_RADIUS**2   # m²
BED_VOIDAGE     = 0.45           # ε
RHO_CAT         = 1200.0         # kg m⁻³ (catalyst bulk density)
PARTICLE_DIA    = 3e-3           # m  (catalyst particle diameter)

# ── Feed conditions ────────────────────────────────────────────────────────────
T_INLET         = 430.0          # K
P_INLET         = 1.6            # atm
P_OUTLET_MIN    = 1.1            # atm  (downstream constraint)

# Inlet mole fractions: CH₃OH, O₂, H₂O, HCHO, CO, N₂
Y_INLET = np.array([0.11, 0.06, 0.03, 0.0, 0.0, 0.80])
F_TOTAL_INLET   = 0.030          # mol/s per tube

# ── Thermochemistry ────────────────────────────────────────────────────────────
DELTA_H_R1 = -37.4e3            # J/mol  (methanol → formaldehyde, exothermic)
DELTA_H_R2 = -238.0e3           # J/mol  (formaldehyde → CO, more exothermic)
R_GAS      = 8.314              # J/(mol·K)

# ── Cp polynomial coefficients [a, b, c, d, e] (J mol⁻¹ K⁻¹) ─────────────────
# Species: CH₃OH, O₂, H₂O, HCHO, CO, N₂
CP_COEFFS = np.array([
    # a           b           c            d            e
    [19.238,   1.0140e-1, -4.1530e-5,  -1.5800e-8,   0.0],      # CH₃OH
    [28.087,  -3.6800e-6,  1.7460e-5,  -1.0650e-8,   2.2060e-12],# O₂
    [33.933,  -8.4220e-3,  2.9900e-5,  -1.7840e-8,   3.6940e-12],# H₂O
    [23.463,   8.0400e-2, -4.8610e-5,   1.3640e-8,   0.0],       # HCHO
    [29.108,  -1.9160e-3,  4.0030e-6,  -2.0870e-9,   2.7720e-13],# CO
    [29.105,  -1.9160e-3,  4.0030e-6,  -2.0870e-9,   2.7720e-13],# N₂
])


def cp_mix(y: np.ndarray, T: float) -> float:
    """
    Mole-fraction-weighted mixture heat capacity (J mol⁻¹ K⁻¹).

    Cp_mix = Σ yᵢ · Cp_i(T)   where  Cp_i = aᵢ + bᵢT + cᵢT² + dᵢT³ + eᵢT⁴
    """
    T_powers = np.array([1, T, T**2, T**3, T**4])
    Cp_i = CP_COEFFS @ T_powers          # shape (6,)
    return float(np.dot(y, Cp_i))


def ergun_dPdz(
    F_total: float,
    T: float,
    P_atm: float,
    M_mix: float,
) -> float:
    """
    Pressure gradient from the simplified Ergun equation (turbulent term, eq. 17).

    dP/dz = −1.75 G² (1−ε) / (Dₚ ρ ε³)

    Parameters
    ----------
    F_total : float
        Total molar flow (mol/s).
    T : float
        Temperature (K).
    P_atm : float
        Pressure (atm).
    M_mix : float
        Mean molar mass of mixture (kg/mol).

    Returns
    -------
    float
        dP/dz in Pa/m (negative = pressure decreasing along reactor).
    """
    P_pa  = P_atm * 101325.0
    rho   = P_pa * M_mix / (R_GAS * T)          # kg m⁻³ (ideal gas)
    # Superficial mass velocity G = (F_total * M_mix) / Ac
    G = F_total * M_mix / TUBE_AREA              # kg m⁻² s⁻¹
    dPdz = -1.75 * G**2 * (1 - BED_VOIDAGE) / (PARTICLE_DIA * rho * BED_VOIDAGE**3)
    return dPdz / 101325.0   # convert Pa/m → atm/m


# Stoichiometry matrices: ν[reaction, species]
# Species: CH₃OH, O₂, H₂O, HCHO, CO, N₂
NU = np.array([
    [-1, -0.5,  1,  1,  0, 0],   # Reaction 1: MeOH oxidation
    [ 0, -0.5,  1,  -1, 1, 0],   # Reaction 2: HCHO oxidation
], dtype=float)

# Molar masses (kg/mol)
M_SPECIES = np.array([0.03204, 0.03200, 0.01801, 0.03003, 0.02801, 0.02801])


class PackedBedPFR:
    """
    Packed-bed plug-flow reactor simulator for methanol oxidation.

    Integrates material, energy, and pressure-drop ODEs along reactor length.

    Parameters
    ----------
    kinetics : str
        'LHHW' or 'power_law'. Default 'LHHW' (more accurate).
    mode : str
        'isothermal' or 'adiabatic'. Default 'isothermal'.
    T_isothermal : float
        Fixed temperature for isothermal operation (K). Default 640.
    n_tubes : int
        Number of reactor tubes in parallel. Default 23000.

    Examples
    --------
    >>> pfr = PackedBedPFR(kinetics='LHHW', mode='isothermal', T_isothermal=640)
    >>> result = pfr.simulate(L=0.5)
    >>> print(f"Yield: {result['yield_HCHO']:.3f}")
    """

    def __init__(
        self,
        kinetics: str = "LHHW",
        mode: str = "isothermal",
        T_isothermal: float = 640.0,
        n_tubes: int = 23_000,
        F_total_per_tube: float = F_TOTAL_INLET,
    ):
        self.mode        = mode
        self.T_iso       = T_isothermal
        self.n_tubes     = n_tubes
        self.F0          = F_total_per_tube

        if kinetics.upper() == "LHHW":
            self.kin = LHHWKinetics()
            self._use_lhhw = True
        else:
            self.kin = PowerLawKinetics()
            self._use_lhhw = False

    def _rates(self, n: np.ndarray, T: float, P: float) -> tuple[float, float]:
        """Dispatch to correct kinetics model."""
        F_total = n.sum()
        y = n / F_total
        # Concentrations (mol m⁻³) or partial pressures (atm)
        if self._use_lhhw:
            P_dict = {
                "MeOH": y[0] * P, "O2":   y[1] * P, "H2O":  y[2] * P,
                "HCHO": y[3] * P, "CO":   y[4] * P, "N2":   y[5] * P,
            }
            return self.kin.rates(P_dict, T)
        else:
            P_pa  = P * 101325
            M_mix = float(M_SPECIES @ y)
            rho   = P_pa * M_mix / (R_GAS * T)
            C = {
                "MeOH": y[0] * rho / M_SPECIES[0],
                "O2":   y[1] * rho / M_SPECIES[1],
                "H2O":  y[2] * rho / M_SPECIES[2],
                "HCHO": y[3] * rho / M_SPECIES[3],
                "CO":   y[4] * rho / M_SPECIES[4],
                "N2":   y[5] * rho / M_SPECIES[5],
            }
            return self.kin.rates(C, T)

    def _odes(self, z: float, state: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the coupled PFR ODE system.

        State vector: [n_MeOH, n_O2, n_H2O, n_HCHO, n_CO, n_N2, T, P]
        """
        n = state[:6]
        T = state[6]
        P = state[7]

        # Clip species flows non-negative *before* summing so the total
        # used in the denominator is consistent with the numerator. The
        # previous code computed F_total from the raw (possibly slightly
        # negative) sum and divided the clipped vector by it, producing
        # mole fractions that did not sum to 1.
        n_pos = np.maximum(n, 0)
        F_total = max(n_pos.sum(), 1e-12)
        y = n_pos / F_total

        r1, r2 = self._rates(n, T, P)
        rates  = np.array([r1, r2])

        # Material balances: dn_i/dz = Σ_j ν_ji r_j · Ac · (1−ε) · ρ_cat
        factor = TUBE_AREA * (1 - BED_VOIDAGE) * RHO_CAT
        dn_dz  = (NU.T @ rates) * factor     # shape (6,)

        # Energy balance
        M_mix = float(M_SPECIES @ y)
        if self.mode == "isothermal":
            dT_dz = 0.0
        else:
            Cp = cp_mix(y, T)
            dT_dz = (
                -(DELTA_H_R1 * r1 + DELTA_H_R2 * r2)
                * factor
                / (F_total * Cp)
            )

        # Pressure drop (Ergun)
        dP_dz = ergun_dPdz(F_total, T, P, M_mix)

        return np.concatenate([dn_dz, [dT_dz, dP_dz]])

    def simulate(
        self,
        L: float = 1.0,
        n_points: int = 200,
    ) -> dict:
        """
        Simulate reactor performance along tube length L.

        Parameters
        ----------
        L : float
            Tube length (m).
        n_points : int
            Number of output points along the reactor.

        Returns
        -------
        dict with keys:
            z           : np.ndarray, axial position (m)
            n           : np.ndarray (n_points × 6), molar flows per tube (mol/s)
            T           : np.ndarray, temperature profile (K)
            P           : np.ndarray, pressure profile (atm)
            yield_HCHO  : float, molar yield of formaldehyde
            selectivity : float, selectivity to formaldehyde
            W_cat       : np.ndarray, cumulative catalyst weight (kg)
            conversion  : float, methanol conversion
        """
        # Initial state
        n0  = Y_INLET * self.F0
        T0  = self.T_iso if self.mode == "isothermal" else T_INLET
        P0  = P_INLET
        y0  = np.concatenate([n0, [T0, P0]])

        z_eval = np.linspace(0, L, n_points)

        # Terminate the integration if pressure drops to (near-)zero. The
        # Ergun term is unbounded as P → 0 and partial pressures fed to
        # the Arrhenius / LHHW kinetics become meaningless, so without an
        # event the solver can drift into a regime where the rate
        # expressions blow up rather than reporting a clean failure.
        def p_floor(z, state):  # noqa: ARG001
            return state[7] - 1e-3   # atm
        p_floor.terminal = True
        p_floor.direction = -1

        sol = solve_ivp(
            self._odes,
            [0, L],
            y0,
            t_eval=z_eval,
            method="RK45",
            rtol=1e-4,
            atol=1e-6,
            events=p_floor,
        )

        if not sol.success:
            raise RuntimeError(f"PFR ODE solver failed: {sol.message}")
        if sol.t_events[0].size:
            raise RuntimeError(
                f"PFR pressure dropped below 1e-3 atm at z={sol.t_events[0][0]:.3f} m; "
                "tube is too long for the given inlet conditions."
            )

        n = sol.y[:6].T   # shape (n_points, 6)
        T = sol.y[6]
        P = sol.y[7]
        z = sol.t

        # Clip negatives from numerical noise
        n = np.maximum(n, 0)

        # Performance metrics
        n_MeOH_in   = n0[0]
        n_MeOH_out  = n[-1, 0]
        n_HCHO_out  = n[-1, 3]
        n_CO_out    = n[-1, 4]

        conversion  = (n_MeOH_in - n_MeOH_out) / n_MeOH_in
        yield_HCHO  = n_HCHO_out / n_MeOH_in
        total_ox    = n_HCHO_out + n_CO_out
        selectivity = n_HCHO_out / total_ox if total_ox > 1e-12 else 1.0

        # Catalyst weight along reactor
        W_cat = z * TUBE_AREA * (1 - BED_VOIDAGE) * RHO_CAT

        return {
            "z":           z,
            "n":           n,          # mol/s per tube
            "T":           T,
            "P":           P,
            "yield_HCHO":  float(yield_HCHO),
            "selectivity": float(selectivity),
            "conversion":  float(conversion),
            "W_cat":       W_cat,
            "n_tubes":     self.n_tubes,
        }

    def production_rate(self, L: float) -> float:
        """
        Total formaldehyde production (tonnes/day) across all tubes.

        Parameters
        ----------
        L : float
            Tube length (m).

        Returns
        -------
        float
            HCHO production in tonnes/day.
        """
        result = self.simulate(L=L)
        n_HCHO_per_tube = result["n"][-1, 3]   # mol/s per tube
        M_HCHO = 0.03003                         # kg/mol
        kg_per_day = n_HCHO_per_tube * self.n_tubes * M_HCHO * 86_400
        return kg_per_day / 1000  # tonnes/day
