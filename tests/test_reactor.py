"""
tests/test_reactor.py
Unit tests for the formaldehyde-reactor package.
Run: python -m pytest tests/ -v
"""

import numpy as np
import pytest
from reactor import PackedBedPFR, ReactorAnalysis
from reactor.kinetics import PowerLawKinetics, LHHWKinetics, arrhenius


# ── Kinetics ──────────────────────────────────────────────────────────────────

class TestArrhenius:
    def test_higher_T_higher_k(self):
        k1 = arrhenius(1e6, 50.0, 600)
        k2 = arrhenius(1e6, 50.0, 700)
        assert k2 > k1

    def test_negative_Ea_decreases_with_T(self):
        """Adsorption constants decrease with temperature (negative Ea)."""
        k1 = arrhenius(1.0, -50.0, 600)
        k2 = arrhenius(1.0, -50.0, 700)
        assert k2 < k1

    def test_zero_Ea_gives_A0(self):
        assert abs(arrhenius(3.0, 0.0, 500) - 3.0) < 1e-9


class TestPowerLawKinetics:
    def setup_method(self):
        self.kin = PowerLawKinetics()
        self.C = {"MeOH": 2.0, "O2": 1.0, "H2O": 0.5, "HCHO": 0.1, "CO": 0.0}

    def test_rates_positive(self):
        r1, r2 = self.kin.rates(self.C, T=640)
        assert r1 >= 0
        assert r2 >= 0

    def test_r1_higher_than_r2(self):
        """Primary reaction should dominate over secondary."""
        r1, r2 = self.kin.rates(self.C, T=640)
        assert r1 > r2

    def test_rate_increases_with_temperature(self):
        r1_low,  _ = self.kin.rates(self.C, T=580)
        r1_high, _ = self.kin.rates(self.C, T=680)
        assert r1_high > r1_low

    def test_zero_concentrations_give_zero_rate(self):
        C_zero = {"MeOH": 0.0, "O2": 0.0, "H2O": 1e-9, "HCHO": 0.0, "CO": 0.0}
        r1, r2 = self.kin.rates(C_zero, T=640)
        assert r1 < 1e-4


class TestLHHWKinetics:
    def setup_method(self):
        self.kin = LHHWKinetics()
        self.P = {"MeOH": 0.18, "O2": 0.096, "H2O": 0.048, "HCHO": 0.001, "CO": 0.0}

    def test_rates_positive(self):
        r1, r2 = self.kin.rates(self.P, T=640)
        assert r1 >= 0
        assert r2 >= 0

    def test_r1_dominates_r2(self):
        r1, r2 = self.kin.rates(self.P, T=640)
        assert r1 > r2

    def test_rate_increases_with_temperature(self):
        r1_low,  _ = self.kin.rates(self.P, T=580)
        r1_high, _ = self.kin.rates(self.P, T=680)
        assert r1_high > r1_low

    def test_adsorption_inhibition_at_high_MeOH(self):
        """High methanol should saturate sites and limit rate growth (LHHW feature)."""
        P_low  = {**self.P, "MeOH": 0.05}
        P_high = {**self.P, "MeOH": 2.00}
        r1_low,  _ = self.kin.rates(P_low, T=640)
        r1_high, _ = self.kin.rates(P_high, T=640)
        # Rate should not scale linearly — the increase should be sub-linear
        assert r1_high / r1_low < 40   # would be ~40x for pure power law

    def test_selectivity_profile_physical(self):
        T_range = np.linspace(550, 750, 10)
        sels = self.kin.selectivity_temperature_sensitivity(T_range, self.P)
        assert all(0 <= s <= 1 for s in sels)


# ── PFR ───────────────────────────────────────────────────────────────────────

class TestPackedBedPFR:
    def test_isothermal_lhhw_runs(self):
        pfr = PackedBedPFR(kinetics="LHHW", mode="isothermal", T_isothermal=640)
        res = pfr.simulate(L=0.5)
        assert "yield_HCHO" in res
        assert "selectivity" in res

    def test_power_law_overpredicts_vs_lhhw(self):
        """Power law should give higher yield than LHHW (per report §3.2)."""
        pfr_lhhw = PackedBedPFR(kinetics="LHHW",       mode="isothermal", T_isothermal=640)
        pfr_pl   = PackedBedPFR(kinetics="power_law",   mode="isothermal", T_isothermal=640)
        res_lhhw = pfr_lhhw.simulate(L=0.5)
        res_pl   = pfr_pl.simulate(L=0.5)
        assert res_pl["yield_HCHO"] > res_lhhw["yield_HCHO"]

    def test_yield_between_0_and_1(self):
        pfr = PackedBedPFR(kinetics="LHHW", mode="isothermal", T_isothermal=640)
        res = pfr.simulate(L=1.0)
        assert 0 <= res["yield_HCHO"] <= 1
        assert 0 <= res["selectivity"] <= 1

    def test_pressure_decreases_along_reactor(self):
        """Ergun equation must produce monotonically decreasing pressure."""
        pfr = PackedBedPFR(kinetics="LHHW", mode="isothermal", T_isothermal=640)
        res = pfr.simulate(L=1.0)
        assert res["P"][-1] < res["P"][0]

    def test_outlet_pressure_above_minimum(self):
        """Outlet pressure must stay above 1.1 atm for downstream unit."""
        pfr = PackedBedPFR(kinetics="LHHW", mode="isothermal", T_isothermal=640)
        res = pfr.simulate(L=1.0)
        assert res["P"][-1] >= 1.1

    def test_isothermal_temperature_constant(self):
        pfr = PackedBedPFR(kinetics="LHHW", mode="isothermal", T_isothermal=640)
        res = pfr.simulate(L=0.5)
        assert np.allclose(res["T"], 640.0, atol=1.0)

    def test_adiabatic_temperature_rises(self):
        """Temperature must increase in exothermic adiabatic reactor."""
        pfr = PackedBedPFR(kinetics="LHHW", mode="adiabatic")
        res = pfr.simulate(L=1.5)
        assert res["T"].max() > res["T"][0]

    def test_methanol_decreases_along_reactor(self):
        pfr = PackedBedPFR(kinetics="LHHW", mode="isothermal", T_isothermal=640)
        res = pfr.simulate(L=1.0)
        assert res["n"][-1, 0] < res["n"][0, 0]   # MeOH consumed

    def test_hcho_increases_along_reactor(self):
        pfr = PackedBedPFR(kinetics="LHHW", mode="isothermal", T_isothermal=640)
        res = pfr.simulate(L=1.0)
        assert res["n"][-1, 3] > res["n"][0, 3]   # HCHO produced

    def test_longer_tube_higher_yield(self):
        pfr = PackedBedPFR(kinetics="LHHW", mode="isothermal", T_isothermal=640)
        res_short = pfr.simulate(L=0.5)
        res_long  = pfr.simulate(L=1.5)
        assert res_long["yield_HCHO"] >= res_short["yield_HCHO"]

    def test_n2_conserved(self):
        """N₂ is inert — molar flow should not change."""
        pfr = PackedBedPFR(kinetics="LHHW", mode="isothermal", T_isothermal=640)
        res = pfr.simulate(L=1.0)
        assert abs(res["n"][-1, 5] - res["n"][0, 5]) < 1e-8


# ── Analysis ──────────────────────────────────────────────────────────────────

class TestReactorAnalysis:
    def setup_method(self):
        self.ra = ReactorAnalysis(kinetics="LHHW")

    def test_temperature_sensitivity_returns_arrays(self):
        T_range = np.linspace(600, 660, 5)
        result = self.ra.temperature_sensitivity(T_range, L=0.5)
        assert len(result["yield_HCHO"]) == 5
        assert len(result["selectivity"]) == 5

    def test_higher_temperature_changes_yield(self):
        T_range = np.array([600.0, 640.0, 680.0])
        result = self.ra.temperature_sensitivity(T_range, L=1.0)
        # Yield should vary with temperature
        assert result["yield_HCHO"].std() > 0

    def test_length_sensitivity_returns_arrays(self):
        L_range = np.array([0.5, 1.0, 1.5])
        result = self.ra.length_sensitivity(L_range, T=640)
        assert len(result["yield_HCHO"]) == 3
        assert len(result["P_outlet"]) == 3

    def test_longer_tube_lower_pressure(self):
        L_range = np.array([0.5, 1.0, 1.5])
        result = self.ra.length_sensitivity(L_range, T=640)
        assert result["P_outlet"][0] > result["P_outlet"][-1]

    def test_flammability_limits_structure(self):
        flamm = self.ra.flammability_limits()
        assert "LFL" in flamm
        assert "LOC" in flamm
        assert "is_flammable" in flamm

    def test_feed_o2_below_loc(self):
        """O₂ in feed (6%) should be below LOC (10%) — inherently safer."""
        flamm = self.ra.flammability_limits()
        assert flamm["y_O2_inlet"] < flamm["LOC"]

    def test_sqp_returns_valid_result(self):
        opt = self.ra.sqp_optimise(
            L_bounds=(0.5, 2.0),
            T_bounds=(600.0, 680.0),
        )
        assert opt["success"]
        assert opt["P_outlet"] >= 1.1
        assert 0 < opt["max_yield"] <= 1

    def test_yield_selectivity_tradeoff_shape(self):
        T_range = np.linspace(600, 680, 6)
        result = self.ra.yield_selectivity_tradeoff(T_range, L=0.5)
        assert len(result["yield_HCHO"]) == 6
        assert len(result["selectivity"]) == 6
