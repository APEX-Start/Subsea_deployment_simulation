# Reproduction Guide

Step-by-step instructions to reproduce all simulation results and paper figures from a clean clone.

## 1. Clone and Install

```bash
git clone https://github.com/APEX-Start/Subsea_deployment_simulation.git
cd Subsea_deployment_simulation
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## 2. Verify Installation

```bash
python -c "import subsea_deployment_simulation; print('Package OK')"
python -m py_compile src/subsea_deployment_simulation/high_fidelity_sim.py
```

## 3. Run Main Simulation

```bash
python -m subsea_deployment_simulation.high_fidelity_sim
```

This generates `results/visualization/visualization_payload.npz` (required for plots 01–10).

**Expected runtime:** ~5–15 minutes depending on hardware.

## 4. Run Operational Window Sweep

```bash
python scripts/visualization/sweep_operational_window.py
```

This generates `results/visualization/sweep_results.npz` (required for plot 11).

**Expected runtime:** ~50–150 minutes for the full 10×13 grid.

## 5. Generate All Paper Figures

```bash
python scripts/visualization/plot_01_upper_boundary_heave_disturbance.py
python scripts/visualization/plot_02_payload_vertical_coordinate_history.py
python scripts/visualization/plot_03_wire_rope_snapshot_envelope.py
python scripts/visualization/plot_04_pch_deployment_trajectory.py
python scripts/visualization/plot_05_pch_and_tool_tilt_histories.py
python scripts/visualization/plot_06_final_hover_stage_zoom_in.py
python scripts/visualization/plot_07_top_tension_history.py
python scripts/visualization/plot_08_maximum_tension_stress_envelope.py
python scripts/visualization/plot_09_wire_rope_stress_time_history.py
python scripts/visualization/plot_10_payload_trajectory_3d_and_displacements.py
python scripts/visualization/plot_11_operational_window.py
python scripts/visualization/plot_12_sea_current.py
```

Output SVGs are saved to `results/figures/`.

## 6. Run Convergence Analysis

```bash
# Static catenary verification (fast)
python scripts/convergence_analysis.py --mode catenary

# Mesh convergence study
python scripts/convergence_analysis.py --mode mesh --quick
```

## 7. Run Sensitivity Analysis

```bash
# Quick test
python scripts/sensitivity_analysis.py --quick

# Full analysis (recommended for paper)
python scripts/sensitivity_analysis.py --samples 100
```

## Expected Output Summary

| Step | Output | Location |
|------|--------|----------|
| Simulation | `visualization_payload.npz` | `results/visualization/` |
| Sweep | `sweep_results.npz` | `results/visualization/` |
| Figures | `fig_01_…svg` – `fig_12_…svg` | `results/figures/` |
| Convergence | CSV + figures | `scripts/results/paper_convergence_validation/` |
| Sensitivity | CSV + figures | `scripts/results/sensitivity_analysis/` |
