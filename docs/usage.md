# Usage Guide

## Environment Setup

```bash
git clone https://github.com/APEX-Start/Subsea_deployment_simulation.git
cd Subsea_deployment_simulation
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # Linux/macOS
python -m pip install --upgrade pip
python -m pip install -e .
```

## Running the Main Simulation

```bash
python -m subsea_deployment_simulation.high_fidelity_sim
```

This runs a complete 3D subsea deployment simulation with default parameters and generates output files.

### Modifying Parameters

All default parameters are in `src/subsea_deployment_simulation/high_fidelity_sim.py`, stored in the `DEFAULT_CONFIG` dictionary. Key sections:

- **Wire rope**: `d` (diameter), `L` (length), `E` (Young's modulus), `rho_c` (density)
- **Payload**: `m_b` (mass), drag coefficients
- **Environment**: `total_water_depth`, wave parameters, current model
- **Deployment**: `release_speed`, `total_segments`, `target_height_above_seabed`

### Output Location

Simulation results are saved to:
- `results/visualization/visualization_payload.npz` — main simulation data for plotting

## Running Convergence Analysis

```bash
# Static catenary verification only
python scripts/convergence_analysis.py --mode catenary

# Full mesh convergence study
python scripts/convergence_analysis.py --mode mesh

# Quick mode (fewer mesh points)
python scripts/convergence_analysis.py --mode mesh --quick
```

## Running Sensitivity Analysis

```bash
# Quick test with few samples
python scripts/sensitivity_analysis.py --quick

# Full analysis
python scripts/sensitivity_analysis.py --samples 100

# Generate only Figure 20 (SRC/SRRC bar chart)
python scripts/sensitivity_analysis.py --fig20-only
```

Output goes to `scripts/results/sensitivity_analysis/` by default.
