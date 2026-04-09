"""
reactor.cli
-----------
Config-driven entry point for reproducible PFR simulations.

Usage:
    python -m reactor.cli --config config/default.yaml

All knobs that affect numerical results live in the YAML config — no
parameters are hard-coded here. Outputs (figure + metrics.json) are written
to the directory specified by `output.results_dir`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml

from reactor import PackedBedPFR, ReactorAnalysis


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def run(config_path: Path) -> dict:
    cfg = load_config(config_path)

    seed = cfg["output"].get("random_seed", 0)
    random.seed(seed)
    np.random.seed(seed)

    rcfg = cfg["reactor"]
    pfr = PackedBedPFR(
        kinetics=rcfg["kinetics"],
        mode=rcfg["mode"],
        T_isothermal=rcfg["T_isothermal"],
    )
    iso = pfr.simulate(L=rcfg["L"])

    pfr_ad = PackedBedPFR(kinetics=rcfg["kinetics"], mode="adiabatic")
    adiab = pfr_ad.simulate(L=cfg["adiabatic"]["L"])

    ra = ReactorAnalysis(kinetics=rcfg["kinetics"])

    tcfg = cfg["sweeps"]["temperature"]
    T_sweep = np.linspace(tcfg["T_min"], tcfg["T_max"], tcfg["n_points"])
    t_sens = ra.temperature_sensitivity(T_sweep, L=tcfg["L"])

    lcfg = cfg["sweeps"]["length"]
    L_sweep = np.linspace(lcfg["L_min"], lcfg["L_max"], lcfg["n_points"])
    l_sens = ra.length_sensitivity(L_sweep, T=lcfg["T"])

    opt = ra.sqp_optimise() if cfg["optimisation"]["run_sqp"] else None

    metrics = {
        "config": str(config_path),
        "isothermal": {
            "T_K": rcfg["T_isothermal"],
            "L_m": rcfg["L"],
            "yield_HCHO": float(iso["yield_HCHO"]),
            "selectivity": float(iso["selectivity"]),
            "conversion": float(iso["conversion"]),
            "P_outlet_atm": float(iso["P"][-1]),
        },
        "adiabatic": {
            "L_m": cfg["adiabatic"]["L"],
            "T_peak_K": float(adiab["T"].max()),
        },
        "optimisation": (
            {
                "success": bool(opt["success"]),
                "optimal_L_m": float(opt["optimal_L"]),
                "optimal_T_K": float(opt["optimal_T"]),
                "max_yield": float(opt["max_yield"]),
                "selectivity": float(opt["selectivity"]),
                "P_outlet_atm": float(opt["P_outlet"]),
            }
            if opt and opt.get("success")
            else None
        ),
    }

    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / cfg["output"]["metrics_name"]
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    _render_figure(
        cfg=cfg,
        iso=iso,
        adiab=adiab,
        t_sens=t_sens,
        l_sens=l_sens,
        out_path=out_dir / cfg["output"]["figure_name"],
    )

    print(f"✓ metrics → {metrics_path}")
    print(f"✓ figure  → {out_dir / cfg['output']['figure_name']}")
    return metrics


def _render_figure(cfg, iso, adiab, t_sens, l_sens, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    SPECIES = ["CH3OH", "O2", "H2O", "HCHO", "CO"]
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    for i, name in enumerate(SPECIES):
        ax1.plot(iso["z"] * 100, iso["n"][:, i] * 1000, lw=2, label=name)
    ax1.set_xlabel("Reactor length (cm)")
    ax1.set_ylabel("Molar flow (mmol/s)")
    ax1.set_title("A | Isothermal flow profiles")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(adiab["z"] * 100, adiab["T"], lw=2.5)
    ax2.set_xlabel("Reactor length (cm)")
    ax2.set_ylabel("Temperature (K)")
    ax2.set_title("B | Adiabatic temperature profile")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t_sens["T"], t_sens["yield_HCHO"] * 100, lw=2.5, label="Yield (%)")
    ax3.plot(t_sens["T"], t_sens["selectivity"] * 100, lw=2.5, ls="--", label="Selectivity (%)")
    ax3.axvline(cfg["reactor"]["T_isothermal"], ls=":", color="grey")
    ax3.set_xlabel("Temperature (K)")
    ax3.set_ylabel("%")
    ax3.set_title("C | Temperature sensitivity")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(l_sens["L"] * 100, l_sens["yield_HCHO"] * 100, lw=2.5, label="Yield (%)")
    ax4b = ax4.twinx()
    ax4b.plot(l_sens["L"] * 100, l_sens["P_outlet"], lw=2, ls="--", color="red", label="P_out")
    ax4b.axhline(1.1, color="black", ls=":", lw=1.5)
    ax4.set_xlabel("Tube length (cm)")
    ax4.set_ylabel("Yield (%)")
    ax4b.set_ylabel("P_outlet (atm)")
    ax4.set_title("D | Length sensitivity")
    ax4.grid(alpha=0.3)

    fig.suptitle("Formaldehyde reactor — reproducible run", fontweight="bold", y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m reactor.cli",
        description="Config-driven PFR simulation entry point.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to YAML config (default: config/default.yaml)",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
