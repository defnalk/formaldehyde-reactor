"""
reactor/analysis.py
-------------------
Sensitivity analysis, flammability limits, and SQP optimisation
for the methanol-to-formaldehyde packed-bed reactor.

Reference: Group 20 Reaction Engineering Report, §4
"""

import numpy as np
from scipy.optimize import minimize

from .pfr import P_INLET, Y_INLET, PackedBedPFR


class ReactorAnalysis:
    """
    Tools for reactor sensitivity analysis, safety, and optimisation.

    Parameters
    ----------
    kinetics : str
        'LHHW' or 'power_law'. Default 'LHHW'.
    n_tubes : int
        Number of parallel reactor tubes. Default 23000.

    Examples
    --------
    >>> ra = ReactorAnalysis()
    >>> result = ra.temperature_sensitivity(T_range=np.linspace(560, 720, 20), L=0.5)
    """

    def __init__(self, kinetics: str = "LHHW", n_tubes: int = 23_000):
        self.kinetics = kinetics
        self.n_tubes  = n_tubes

    def _pfr(self, T: float = 640.0, mode: str = "isothermal") -> PackedBedPFR:
        # PackedBedPFR has no length parameter — L belongs to simulate(),
        # not the constructor — so the previous L kwarg was misleading
        # dead state. Drop it.
        return PackedBedPFR(
            kinetics=self.kinetics,
            mode=mode,
            T_isothermal=T,
            n_tubes=self.n_tubes,
        )

    # ── Sensitivity analyses ────────────────────────────────────────────────

    def temperature_sensitivity(
        self,
        T_range: np.ndarray,
        L: float = 0.5,
    ) -> dict[str, np.ndarray]:
        """
        Sweep isothermal temperature and record yield and selectivity.

        Parameters
        ----------
        T_range : np.ndarray
            Temperature values (K) to simulate.
        L : float
            Tube length (m).

        Returns
        -------
        dict with arrays: 'T', 'yield_HCHO', 'selectivity', 'conversion'
        """
        yields, sels, convs = [], [], []
        for T in T_range:
            res = self._pfr(T=T).simulate(L=L)
            yields.append(res["yield_HCHO"])
            sels.append(res["selectivity"])
            convs.append(res["conversion"])

        return {
            "T":           T_range,
            "yield_HCHO":  np.array(yields),
            "selectivity": np.array(sels),
            "conversion":  np.array(convs),
        }

    def length_sensitivity(
        self,
        L_range: np.ndarray,
        T: float = 640.0,
    ) -> dict[str, np.ndarray]:
        """
        Sweep tube length and record yield, production rate, and pressure drop.

        Parameters
        ----------
        L_range : np.ndarray
            Tube length values (m) to simulate.
        T : float
            Isothermal operating temperature (K).

        Returns
        -------
        dict with arrays: 'L', 'yield_HCHO', 'production_tpd', 'P_outlet'
        """
        # production_rate(L) internally re-runs simulate(L=L), so the
        # original loop solved every reactor twice. Reuse the molar HCHO
        # outflow from the simulate() call we already made and convert
        # locally — same answer, half the ODE solves.
        M_HCHO = 0.03003   # kg/mol
        yields, prods, P_out = [], [], []
        pfr = self._pfr(T=T)
        for L in L_range:
            res = pfr.simulate(L=L)
            yields.append(res["yield_HCHO"])
            P_out.append(res["P"][-1])
            n_HCHO_per_tube = res["n"][-1, 3]   # mol/s per tube
            tpd = n_HCHO_per_tube * pfr.n_tubes * M_HCHO * 86_400 / 1000
            prods.append(tpd)

        return {
            "L":             L_range,
            "yield_HCHO":    np.array(yields),
            "P_outlet":      np.array(P_out),
            "production_tpd": np.array(prods),
        }

    # ── Flammability analysis ───────────────────────────────────────────────

    def flammability_limits(self, fuel: str = "MeOH") -> dict[str, float]:
        """
        Calculate Lower Flammability Limit (LFL) and Lowest Oxygen
        Concentration (LOC) for the feed mixture.

        Uses the Le Chatelier mixing rule for multi-component feeds.

        Methanol flammability data (NFPA / literature):
          LFL_pure = 6.0 vol%, UFL = 36.5 vol%
          LOC ≈ 10.0 vol% O₂ (from Zabetakis, 1965)

        Parameters
        ----------
        fuel : str
            Fuel component (currently 'MeOH').

        Returns
        -------
        dict with: 'LFL', 'UFL', 'LOC', 'y_fuel_inlet', 'y_O2_inlet',
                   'is_flammable', 'safety_margin_LFL'
        """
        # Pure methanol limits (vol% in air)
        LFL_MeOH = 0.060    # 6.0%
        UFL_MeOH = 0.365    # 36.5%
        LOC_MeOH = 0.100    # 10.0%  (O₂ concentration at LFL with air)

        y_fuel = Y_INLET[0]   # 0.11 = 11%
        y_O2   = Y_INLET[1]   # 0.06 = 6%

        # Is the feed in the flammable range?
        is_flammable = LFL_MeOH < y_fuel < UFL_MeOH and y_O2 > LOC_MeOH

        # Safety margin below LFL (positive = below LFL = safer)
        safety_margin = LFL_MeOH - y_fuel

        return {
            "LFL":               LFL_MeOH,
            "UFL":               UFL_MeOH,
            "LOC":               LOC_MeOH,
            "y_fuel_inlet":      y_fuel,
            "y_O2_inlet":        y_O2,
            "is_flammable":      is_flammable,
            "safety_margin_LFL": safety_margin,
            "note": (
                "Feed (11% MeOH, 6% O₂) is within flammability limits. "
                "High N₂ dilution (80%) reduces ignition risk. "
                "Oxygen concentration (6%) is below the LOC (10%), "
                "providing an inherent safety barrier against propagation."
            ),
        }

    # ── SQP Optimisation ────────────────────────────────────────────────────

    def sqp_optimise(
        self,
        L_bounds: tuple[float, float] = (0.5, 3.0),
        T_bounds: tuple[float, float] = (580.0, 680.0),
        target_production_tpd: float = 100.0,
    ) -> dict:
        """
        Sequential Quadratic Programming (SQP) optimisation.

        Maximises HCHO yield subject to:
          - Outlet pressure ≥ 1.1 bar
          - Tube length within [L_min, L_max]
          - Temperature within [T_min, T_max]

        Parameters
        ----------
        L_bounds : tuple
            (min, max) tube length in metres.
        T_bounds : tuple
            (min, max) operating temperature in K.
        target_production_tpd : float
            Target production in tonnes/day (used to cross-check).

        Returns
        -------
        dict with: 'optimal_L', 'optimal_T', 'max_yield',
                   'selectivity', 'P_outlet', 'success'
        """
        def neg_yield(x):
            L, T = x
            try:
                res = self._pfr(T=T).simulate(L=L)
                return -res["yield_HCHO"]
            except Exception:
                return 0.0

        def pressure_constraint(x):
            L, T = x
            try:
                res = self._pfr(T=T).simulate(L=L)
                return res["P"][-1] - 1.1   # must be ≥ 0
            except Exception:
                return -1.0

        constraints = [{"type": "ineq", "fun": pressure_constraint}]
        bounds = [L_bounds, T_bounds]
        x0 = [0.5, 640.0]

        result = minimize(
            neg_yield,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-6, "maxiter": 100},
        )

        if result.success:
            L_opt, T_opt = result.x
            res = self._pfr(T=T_opt).simulate(L=L_opt)
            return {
                "optimal_L":   round(L_opt, 4),
                "optimal_T":   round(T_opt, 1),
                "max_yield":   round(res["yield_HCHO"], 4),
                "selectivity": round(res["selectivity"], 4),
                "conversion":  round(res["conversion"], 4),
                "P_outlet":    round(res["P"][-1], 3),
                "success":     True,
            }
        else:
            return {"success": False, "message": result.message}

    def yield_selectivity_tradeoff(
        self,
        T_range: np.ndarray,
        L: float = 0.5,
    ) -> dict[str, np.ndarray]:
        """
        Compute the yield-selectivity Pareto frontier across temperatures.

        Higher temperatures → higher rates → higher conversion but
        more CO formation → lower selectivity.

        Returns yield and selectivity arrays for plotting.
        """
        result = self.temperature_sensitivity(T_range, L=L)
        return {
            "yield_HCHO":  result["yield_HCHO"],
            "selectivity": result["selectivity"],
            "T":           result["T"],
        }
