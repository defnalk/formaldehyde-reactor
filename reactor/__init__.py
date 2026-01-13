"""
formaldehyde-reactor
--------------------
Python simulation of a catalytic packed-bed PFR for the partial
oxidation of methanol to formaldehyde.

    CH₃OH + ½O₂  →  HCHO + H₂O    (desired)
    HCHO  + ½O₂  →  CO   + H₂O    (undesired)

Based on the Imperial College London Reaction Engineering group
design project (Group 20, 2025).

Optimal design: isothermal at 640 K, 0.5 m tubes, 23,000 tubes
                → 95.7% yield, 96.5% selectivity

Modules
-------
kinetics      : Power-law and LHHW kinetic models
pfr           : PFR material + energy + pressure-drop ODEs
analysis      : Yield, selectivity, flammability, SQP optimisation
"""

from .kinetics  import PowerLawKinetics, LHHWKinetics
from .pfr       import PackedBedPFR
from .analysis  import ReactorAnalysis

__all__ = ["PowerLawKinetics", "LHHWKinetics", "PackedBedPFR", "ReactorAnalysis"]
