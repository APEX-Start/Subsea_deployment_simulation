# Visualization Guide

## Overview

The `scripts/visualization/` directory contains 12 publication-ready figure generators for the subsea deployment simulation paper.

## Prerequisites

Before running visualization scripts, you need simulation output data:

1. Run the main simulation to generate `visualization_payload.npz`:
   ```bash
   python -m subsea_deployment_simulation.high_fidelity_sim
   ```
   This creates `results/visualization/visualization_payload.npz`.

2. For the operational window plot (plot_11), run the parameter sweep:
   ```bash
   python scripts/visualization/sweep_operational_window.py
   ```
   This creates `results/visualization/sweep_results.npz`.

## Script-to-Figure Mapping

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `plot_01_…py` | `visualization_payload.npz` | `fig_01_…svg` | Upper boundary heave disturbance |
| `plot_02_…py` | `visualization_payload.npz` | `fig_02_…svg` | Payload vertical coordinate history |
| `plot_03_…py` | `visualization_payload.npz` | `fig_03_…svg` | Wire rope snapshot envelope |
| `plot_04_…py` | `visualization_payload.npz` | `fig_04_…svg` | PCH deployment trajectory |
| `plot_05_…py` | `visualization_payload.npz` | `fig_05_…svg` | PCH and tool tilt histories |
| `plot_06_…py` | `visualization_payload.npz` | `fig_06_…svg` | Final hover stage zoom-in |
| `plot_07_…py` | `visualization_payload.npz` | `fig_07_…svg` | Top tension history |
| `plot_08_…py` | `visualization_payload.npz` | `fig_08_…svg` | Maximum tension stress envelope |
| `plot_09_…py` | `visualization_payload.npz` | `fig_09_…svg` | Wire rope stress time history |
| `plot_10_…py` | `visualization_payload.npz` | `fig_10_…svg` | Payload trajectory 3D & displacements |
| `plot_11_…py` | `sweep_results.npz` | `fig_11_…svg` | Operational window diagram |
| `plot_12_…py` | (built-in data) | `fig_12_…svg` | Sea current velocity profile |

## Generating All Figures

```bash
# From the project root, after running the simulation:
for i in $(seq -w 1 12); do
    python scripts/visualization/plot_${i}_*.py
done
```

## Output Directory

All figures are saved to `results/figures/` as SVG vector graphics.
