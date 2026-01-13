"""
examples/full_simulation.py
---------------------------
Full PFR simulation for methanol → formaldehyde:
  1. Isothermal LHHW vs power-law comparison (flow profiles)
  2. Adiabatic temperature profile
  3. Temperature sensitivity (yield & selectivity)
  4. SQP optimisation result

Run from project root:
    python examples/full_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from reactor import PackedBedPFR, ReactorAnalysis

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
})

BLUE   = "#1d3557"
RED    = "#e63946"
GREEN  = "#2d6a4f"
ORANGE = "#f4a261"
PURPLE = "#7b2d8b"

SPECIES = ["CH₃OH", "O₂", "H₂O", "HCHO", "CO", "N₂"]


# ════════════════════════════════════════════════════════════════════════════
# 1. ISOTHERMAL FLOW PROFILES — LHHW vs Power Law (640 K, 0.5 m)
# ════════════════════════════════════════════════════════════════════════════
L_OPT = 0.5
T_OPT = 640.0

print("Running isothermal simulations (640 K, 0.5 m)...")

pfr_lhhw = PackedBedPFR(kinetics="LHHW",       mode="isothermal", T_isothermal=T_OPT)
pfr_pl   = PackedBedPFR(kinetics="power_law",   mode="isothermal", T_isothermal=T_OPT)

res_lhhw = pfr_lhhw.simulate(L=L_OPT)
res_pl   = pfr_pl.simulate(L=L_OPT)

print(f"  LHHW  → Yield: {res_lhhw['yield_HCHO']:.3f}  "
      f"Selectivity: {res_lhhw['selectivity']:.3f}  "
      f"Conversion: {res_lhhw['conversion']:.3f}")
print(f"  PL    → Yield: {res_pl['yield_HCHO']:.3f}  "
      f"Selectivity: {res_pl['selectivity']:.3f}  "
      f"Conversion: {res_pl['conversion']:.3f}")


# ════════════════════════════════════════════════════════════════════════════
# 2. ADIABATIC TEMPERATURE PROFILE
# ════════════════════════════════════════════════════════════════════════════
print("\nRunning adiabatic simulation...")

pfr_adiab = PackedBedPFR(kinetics="LHHW", mode="adiabatic")
res_adiab = pfr_adiab.simulate(L=1.5)

print(f"  Adiabatic T_peak: {res_adiab['T'].max():.1f} K  "
      f"(inlet 430 K → peak {res_adiab['T'].max():.0f} K)")


# ════════════════════════════════════════════════════════════════════════════
# 3. TEMPERATURE SENSITIVITY
# ════════════════════════════════════════════════════════════════════════════
print("\nRunning temperature sensitivity sweep...")

ra = ReactorAnalysis(kinetics="LHHW")
T_sweep = np.linspace(560, 700, 30)
sens = ra.temperature_sensitivity(T_sweep, L=L_OPT)


# ════════════════════════════════════════════════════════════════════════════
# 4. SQP OPTIMISATION
# ════════════════════════════════════════════════════════════════════════════
print("\nRunning SQP optimisation...")

opt = ra.sqp_optimise()
if opt["success"]:
    print(f"  Optimal L = {opt['optimal_L']} m, T = {opt['optimal_T']} K")
    print(f"  Max yield = {opt['max_yield']:.4f}, Selectivity = {opt['selectivity']:.4f}")
    print(f"  Outlet pressure = {opt['P_outlet']:.3f} atm")

# Flammability
flamm = ra.flammability_limits()
print(f"\nFlammability: O₂ in feed ({flamm['y_O2_inlet']*100:.0f}%) "
      f"< LOC ({flamm['LOC']*100:.0f}%) → inherently safer operation")


# ════════════════════════════════════════════════════════════════════════════
# 5. VISUALISATION
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating figure...")

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.32)
colours = [RED, ORANGE, BLUE, GREEN, PURPLE, "grey"]

# ── Panel A: Isothermal molar flow profiles (LHHW) ──────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for i, (name, col) in enumerate(zip(SPECIES[:5], colours[:5])):
    ax1.plot(res_lhhw["z"] * 100, res_lhhw["n"][:, i] * 1000,
             color=col, lw=2, label=name)
ax1.set_xlabel("Reactor length (cm)")
ax1.set_ylabel("Molar flow rate (mmol/s)")
ax1.set_title("A  |  Isothermal Flow Profiles (LHHW, 640 K)", fontweight="bold")
ax1.legend(fontsize=8, ncol=2)

# ── Panel B: Adiabatic temperature profile ──────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2_twin = ax2.twinx()
ax2.plot(res_adiab["z"] * 100, res_adiab["T"], color=RED, lw=2.5, label="Temperature")
ax2_twin.plot(res_adiab["z"] * 100, res_adiab["n"][:, 3] * 1000,
              color=GREEN, lw=2, ls="--", label="HCHO flow")
ax2.set_xlabel("Reactor length (cm)")
ax2.set_ylabel("Temperature (K)", color=RED)
ax2_twin.set_ylabel("HCHO flow (mmol/s)", color=GREEN)
ax2.tick_params(axis="y", colors=RED)
ax2_twin.tick_params(axis="y", colors=GREEN)
ax2.set_title("B  |  Adiabatic Temperature Profile (LHHW)", fontweight="bold")
ax2.grid(True, alpha=0.3)
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# ── Panel C: Temperature sensitivity ────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3b = ax3.twinx()
l1, = ax3.plot(sens["T"], sens["yield_HCHO"] * 100,  color=BLUE,  lw=2.5, label="Yield (%)")
l2, = ax3b.plot(sens["T"], sens["selectivity"] * 100, color=ORANGE, lw=2.5, ls="--", label="Selectivity (%)")
ax3.axvline(T_OPT, ls=":", color="grey", lw=1.5, label=f"Optimal T = {T_OPT:.0f} K")
ax3.set_xlabel("Temperature (K)")
ax3.set_ylabel("HCHO Yield (%)", color=BLUE)
ax3b.set_ylabel("Selectivity (%)", color=ORANGE)
ax3.tick_params(axis="y", colors=BLUE)
ax3b.tick_params(axis="y", colors=ORANGE)
ax3.set_title("C  |  Temperature Sensitivity (L = 0.5 m)", fontweight="bold")
ax3.legend(handles=[l1, l2], fontsize=8)

# ── Panel D: Length sensitivity + pressure constraint ───────────────────────
ax4 = fig.add_subplot(gs[1, 1])
L_range = np.linspace(0.3, 2.0, 25)
len_sens = ra.length_sensitivity(L_range, T=T_OPT)

ax4b = ax4.twinx()
ax4.plot(len_sens["L"] * 100, len_sens["yield_HCHO"] * 100, color=BLUE,   lw=2.5, label="Yield (%)")
ax4b.plot(len_sens["L"] * 100, len_sens["P_outlet"],          color=RED,    lw=2,   ls="--", label="P_out (atm)")
ax4b.axhline(1.1, color="black", ls=":", lw=1.5, label="P_min = 1.1 atm")
ax4.axvline(L_OPT * 100, ls=":", color="grey", lw=1.5)

ax4.set_xlabel("Tube length (cm)")
ax4.set_ylabel("HCHO Yield (%)", color=BLUE)
ax4b.set_ylabel("Outlet pressure (atm)", color=RED)
ax4.tick_params(axis="y", colors=BLUE)
ax4b.tick_params(axis="y", colors=RED)
ax4.set_title("D  |  Length Sensitivity + Pressure Constraint", fontweight="bold")
lines = ax4.get_legend_handles_labels()[0] + ax4b.get_legend_handles_labels()[0]
labels = ax4.get_legend_handles_labels()[1] + ax4b.get_legend_handles_labels()[1]
ax4.legend(lines, labels, fontsize=8)

fig.suptitle(
    "Formaldehyde Reactor Simulation — Methanol Partial Oxidation\n"
    f"Optimal: isothermal 640 K · 0.5 m tubes · 23,000 tubes · "
    f"Yield {res_lhhw['yield_HCHO']*100:.1f}% · Selectivity {res_lhhw['selectivity']*100:.1f}%",
    fontsize=12, fontweight="bold", y=1.01,
)

plt.savefig("examples/reactor_simulation_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Figure saved → examples/reactor_simulation_results.png")
