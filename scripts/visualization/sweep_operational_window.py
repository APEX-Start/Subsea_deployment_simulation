# -*- coding: utf-8 -*-
"""
sweep_operational_window.py — Parameter sweep for operational window

Batch-runs high_fidelity_sim.py across a 10x10 (equivalent regular wave height, v_rel) grid.
For each point, extracts three constraint metrics:
  - max_stress_MPa   (sigma_max)
  - avg_pch_tilt_deg  (theta_PCH)
  - avg_horizontal_offset (R_offset)

Saves results to: results/visualization/sweep_results.npz

Estimated runtime: 50-150 minutes for 100 simulations.
"""

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# --- Ensure package is importable even without `pip install -e .` ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from subsea_deployment_simulation.high_fidelity_sim import DEFAULT_CONFIG, DynamicWireRopeSystem3D


# ==============================================================================
# Grid definition
# ==============================================================================
HS_VALUES = np.linspace(0.5, 5.0, 10)      # equivalent regular wave height H [m], 0.5 to 5.0 step 0.5
VREL_VALUES = np.linspace(0.2, 1.4, 13)    # v_rel [m/s], 0.2 to 1.4 step 0.1
N_TOTAL = len(HS_VALUES) * len(VREL_VALUES)

# Output path
OUTPUT_DIR = PROJECT_ROOT / "results" / "visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "sweep_results.npz"


# ==============================================================================
# Single simulation runner
# ==============================================================================
def run_single(hs: float, v_rel: float) -> dict:
    """
    Run one deployment simulation for given (Hs, v_rel).

    Returns dict with keys:
        sigma_max_MPa, theta_pch_deg, offset_m, success, error_msg
    """
    config = DEFAULT_CONFIG.copy()

    # --- Set wave / speed parameters ---
    config["wave_height"] = hs
    config["wave_amplitude"] = hs / 2.0       # regular wave amplitude A = H / 2
    config["release_speed"] = v_rel
    config["verbose"] = False                  # suppress log spam

    # --- Instantiate and run ---
    system = DynamicWireRopeSystem3D(config)

    per_segment_T = system.release_duration
    t_end = per_segment_T * system.total_segments + 100  # extra settling time

    sol = system.simulate_dynamic_release(t_end=t_end, dt_output=10)

    summary = system.print_deployment_summary(sol)

    return {
        "sigma_max_MPa": summary["max_stress_MPa"],
        "theta_pch_deg": summary["avg_pch_tilt_deg"],
        "offset_m": summary["avg_horizontal_offset"],
        "success": True,
        "error_msg": "",
    }


# ==============================================================================
# Main sweep
# ==============================================================================
def main():
    print("=" * 70)
    print(" Operational Window Parameter Sweep")
    print(f" Grid: {len(HS_VALUES)} H x {len(VREL_VALUES)} v_rel = {N_TOTAL} runs")
    print(f" H range:    [{HS_VALUES[0]:.1f}, {HS_VALUES[-1]:.1f}] m")
    print(f" v_rel range: [{VREL_VALUES[0]:.1f}, {VREL_VALUES[-1]:.1f}] m/s")
    print(f" Output:     {OUTPUT_PATH}")
    print("=" * 70)

    # --- Allocate result arrays ---
    sigma_grid = np.full((len(VREL_VALUES), len(HS_VALUES)), np.nan)
    theta_grid = np.full((len(VREL_VALUES), len(HS_VALUES)), np.nan)
    offset_grid = np.full((len(VREL_VALUES), len(HS_VALUES)), np.nan)

    t_start = time.time()
    run_count = 0
    fail_count = 0

    for j, v_rel in enumerate(VREL_VALUES):
        for i, hs in enumerate(HS_VALUES):
            run_count += 1
            t_run_start = time.time()

            try:
                result = run_single(hs, v_rel)
                if result["success"]:
                    sigma_grid[j, i] = result["sigma_max_MPa"]
                    theta_grid[j, i] = result["theta_pch_deg"]
                    offset_grid[j, i] = result["offset_m"]

                    elapsed = time.time() - t_run_start
                    eta = (time.time() - t_start) / run_count * (N_TOTAL - run_count)
                    print(
                        f"[{run_count:3d}/{N_TOTAL}] "
                        f"Hs={hs:.2f} v_rel={v_rel:.2f}  |  "
                        f"sigma={result['sigma_max_MPa']:.1f} MPa  "
                        f"theta={result['theta_pch_deg']:.2f} deg  "
                        f"R={result['offset_m']:.2f} m  |  "
                        f"{elapsed:.0f}s  ETA {eta/60:.0f}min"
                    )
                else:
                    fail_count += 1
                    print(f"[{run_count:3d}/{N_TOTAL}] Hs={hs:.2f} v_rel={v_rel:.2f}  FAILED: {result['error_msg']}")

            except Exception as e:
                fail_count += 1
                print(f"[{run_count:3d}/{N_TOTAL}] Hs={hs:.2f} v_rel={v_rel:.2f}  EXCEPTION: {e}")
                traceback.print_exc()

            # --- Save intermediate every 10 runs ---
            if run_count % 10 == 0:
                _save_intermediate(sigma_grid, theta_grid, offset_grid)

    # --- Final save ---
    _save_final(sigma_grid, theta_grid, offset_grid)

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f" Sweep complete: {run_count} runs, {fail_count} failed")
    print(f" Total time: {total_time/60:.1f} min")
    print(f" Results saved to: {OUTPUT_PATH}")
    print("=" * 70)


def _save_intermediate(sigma, theta, offset):
    """Save checkpoint to a temporary file."""
    tmp_path = OUTPUT_DIR / "sweep_results_tmp.npz"
    np.savez(
        tmp_path,
        hs_grid=HS_VALUES,
        vrel_grid=VREL_VALUES,
        sigma_grid=sigma,
        theta_grid=theta,
        offset_grid=offset,
        incomplete=True,
    )


def _save_final(sigma, theta, offset):
    """Save final results."""
    np.savez(
        OUTPUT_PATH,
        hs_grid=HS_VALUES,
        vrel_grid=VREL_VALUES,
        sigma_grid=sigma,
        theta_grid=theta,
        offset_grid=offset,
        incomplete=False,
    )
    # Remove temp file if it exists
    tmp_path = OUTPUT_DIR / "sweep_results_tmp.npz"
    if tmp_path.exists():
        tmp_path.unlink()


if __name__ == "__main__":
    main()
