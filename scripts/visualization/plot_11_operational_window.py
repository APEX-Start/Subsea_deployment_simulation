# -*- coding: utf-8 -*-
"""
plot_11_operational_window.py - Recommended Operational Window

Generates a publication-ready operational window diagram in (H, v_rel) space,
showing safe / caution / prohibited regions based on three physical constraints:

  1. sigma_max < sigma_allow    (wire rope stress)
  2. theta_PCH  < theta_limit   (PCH tilt angle)
  3. R_offset   < R_limit       (horizontal offset)

Data source: high-fidelity sweep results when available; analytical fallback otherwise.

Output: fig_11_operational_window.svg  (vector, SCI-paper standard)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from subsea_deployment_simulation.plotting_style import apply_times_new_roman_style

apply_times_new_roman_style()

# ==============================================================================
# Constraint thresholds (DNV/API recommended practice)
# ==============================================================================
# Wire rope: d=0.032m, A=8.042e-4 m2, MBL=560 kN, SF=3
SIGMA_ALLOW_MPA = 220.0   # MPa  - allowable stress (MBL-based SWL, SF=3 per DNV)
THETA_LIMIT = 20.0        # deg  - PCH tilt limit
R_LIMIT = 100.0            # m    - horizontal offset limit (landing accuracy)


# ==============================================================================
# Helper: resolve project directories
# ==============================================================================
def _resolve_project_root() -> Path:
    """Locate the project root by walking up from this script."""
    return Path(__file__).resolve().parents[2]


def _resolve_data_dir() -> Path:
    """Return the directory containing visualization data (npz files)."""
    root = _resolve_project_root()
    candidates = [
        root / "results" / "visualization",
        root / "SCI_paper",
        Path.cwd() / "results" / "visualization",
        Path.cwd() / "SCI_paper",
    ]
    for d in candidates:
        if d.exists():
            return d
    return root / "results" / "visualization"


def _resolve_output_dir() -> Path:
    """Return the directory for generated figures."""
    root = _resolve_project_root()
    out = root / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ==============================================================================
# Data loading and interpolation
# ==============================================================================
def _interpolate_sweep_grid(
    sweep_path: Path,
    hs_fine: np.ndarray,
    vr_fine: np.ndarray,
):
    """
    Load 10x10 sweep data and interpolate onto a fine (Hs, v_rel) grid.

    Uses scipy.interpolate.RegularGridInterpolator with linear interpolation
    and constant-value extrapolation for points outside the sweep domain.

    Returns (sigma_grid, theta_grid, offset_grid) each of shape (n_vr, n_hs).
    """
    from scipy.interpolate import RegularGridInterpolator

    data = np.load(sweep_path, allow_pickle=True)

    hs_sweep = data["hs_grid"]        # (10,)
    vr_sweep = data["vrel_grid"]      # (10,)
    sigma_sweep = data["sigma_grid"]  # (10, 10)  rows=v_rel, cols=Hs
    theta_sweep = data["theta_grid"]
    offset_sweep = data["offset_grid"]

    # Build fine mesh for querying
    HS_fine, VR_fine = np.meshgrid(hs_fine, vr_fine)  # VR on rows, HS on cols
    pts = np.column_stack([VR_fine.ravel(), HS_fine.ravel()])  # (v_rel, Hs) pairs

    grids = {}
    labels = ["sigma", "theta", "offset"]
    sweeps = [sigma_sweep, theta_sweep, offset_sweep]

    for name, sweep in zip(labels, sweeps):
        # RegularGridInterpolator expects (v_rel, Hs) order
        interp = RegularGridInterpolator(
            (vr_sweep, hs_sweep),
            sweep,
            method="linear",
            bounds_error=False,
            fill_value=None,  # extrapolate using nearest edge value
        )
        # For points outside the grid, clip to bounds then interpolate
        pts_clipped = pts.copy()
        pts_clipped[:, 0] = np.clip(pts_clipped[:, 0], vr_sweep[0], vr_sweep[-1])
        pts_clipped[:, 1] = np.clip(pts_clipped[:, 1], hs_sweep[0], hs_sweep[-1])
        grid_flat = interp(pts_clipped)
        grids[name] = grid_flat.reshape(HS_fine.shape)

    return grids["sigma"], grids["theta"], grids["offset"]


# ==============================================================================
# Region classifier
# ==============================================================================
def classify_region(
    sigma: np.ndarray,
    theta: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """
    Classify each point:

        0 - Safe       (all constraints satisfied with margin > 20 %)
        1 - Caution    (at least one constraint within  0-20 % of limit)
        2 - Prohibited (at least one constraint violated)
    """
    r_sigma = sigma / SIGMA_ALLOW_MPA
    r_theta = theta / THETA_LIMIT
    r_offset = offset / R_LIMIT

    r_max = np.maximum(np.maximum(r_sigma, r_theta), r_offset)

    region = np.full(sigma.shape, 0, dtype=int)
    region[(r_max >= 0.8) & (r_max < 1.0)] = 1
    region[r_max >= 1.0] = 2
    return region


# ==============================================================================
# Main plot
# ==============================================================================
def main() -> None:
    data_dir = _resolve_data_dir()
    output_dir = _resolve_output_dir()

    # ---- Fine grid for plotting ----
    n_hs, n_vrel = 201, 201
    hs_vals = np.linspace(0.0, 5.0, n_hs)       # Hs  in [0, 5] m
    vr_vals = np.linspace(0.2, 1.5, n_vrel)     # v_rel in [0.2, 1.5] m/s
    HS, VR = np.meshgrid(hs_vals, vr_vals)

    # ---- Prefer actual sweep data; fall back only when no sweep file is available ----
    sweep_path = data_dir / "sweep_results.npz"
    model_path = _resolve_project_root() / "src" / "subsea_deployment_simulation" / "high_fidelity_sim.py"
    sweep_is_current = (
        sweep_path.exists()
        and (not model_path.exists() or sweep_path.stat().st_mtime >= model_path.stat().st_mtime)
    )
    if sweep_is_current:
        sigma_grid, theta_grid, offset_grid = _interpolate_sweep_grid(sweep_path, hs_vals, vr_vals)
        data_source = "Interpolated high-fidelity sweep results"
    else:
        if sweep_path.exists():
            print("Warning: sweep_results.npz is older than high_fidelity_sim.py; using analytical fallback.")
        sigma_grid, theta_grid, offset_grid = _compute_analytical(HS, VR)
        data_source = "Analytical fallback model"

    region_grid = classify_region(sigma_grid, theta_grid, offset_grid)

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(12, 9.25), constrained_layout=True)

    # ---- Custom discrete colormap ----
    cmap = ListedColormap(["#2ca02c", "#ffd700", "#d62728"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], ncolors=3)

    ax.pcolormesh(HS, VR, region_grid, cmap=cmap, norm=norm,
                  shading="auto", rasterized=True, alpha=0.85)

    # ---- Constraint boundary contour lines ----
    cs_sigma = ax.contour(HS, VR, sigma_grid, levels=[SIGMA_ALLOW_MPA],
                          colors="black", linewidths=3.5, linestyles="--")
    cs_theta = ax.contour(HS, VR, theta_grid, levels=[THETA_LIMIT],
                          colors="black", linewidths=3.5, linestyles="-.")
    cs_offset = ax.contour(HS, VR, offset_grid, levels=[R_LIMIT],
                           colors="black", linewidths=3.5, linestyles=":")

    # ---- Contour labels (auto-placed) ----
    fmt_sigma = (r"$\sigma_{\mathrm{max}}="
                 + f"{SIGMA_ALLOW_MPA:.0f}" + r"\,\mathrm{MPa}$")
    fmt_theta = (r"$\theta_{\mathrm{PCH}}="
                 + f"{THETA_LIMIT:.0f}" + r"^\circ$")
    fmt_offset = (r"$R_{\mathrm{offset}}="
                  + f"{R_LIMIT:.0f}" + r"\,\mathrm{m}$")

    ax.clabel(cs_sigma, inline=True, fontsize=22,
              fmt={SIGMA_ALLOW_MPA: fmt_sigma})
    ax.clabel(cs_theta, inline=True, fontsize=22,
              fmt={THETA_LIMIT: fmt_theta})
    ax.clabel(cs_offset, inline=True, fontsize=22,
              fmt={R_LIMIT: fmt_offset})

    # ---- Stress contour lines (supplementary) ----
    stress_min = max(60, np.nanmin(sigma_grid))
    stress_max = min(300, np.nanmax(sigma_grid))
    stress_levels = np.linspace(stress_min, stress_max, 6)[1:-1]
    cs_sigma_fill = ax.contour(HS, VR, sigma_grid, levels=stress_levels,
                               colors="#555555", linewidths=1.2, alpha=0.4)
    ax.clabel(cs_sigma_fill, inline=True, fontsize=18,
              fmt=lambda v: f"{v:.0f} MPa")

    # ---- Legend ----
    legend_elements = [
        Patch(facecolor="#2ca02c", alpha=0.85,
              label="Safe (all constraints, margin > 20 %)"),
        Patch(facecolor="#ffd700", alpha=0.85,
              label="Caution (within 0-20 % of limit)"),
        Patch(facecolor="#d62728", alpha=0.85,
              label="Prohibited (constraint violated)"),
    ]
    leg = ax.legend(handles=legend_elements, loc="upper right",
                    frameon=True, edgecolor="black", facecolor="white",
                    framealpha=1.0, prop={"size": 22, "weight": "bold"})

    # ---- Constraint annotation box ----
    annotation_text = (
        r"$\bf{Physical\ Criteria:}$" + "\n"
        + r"$\sigma_{\mathrm{max}} < \sigma_{\mathrm{allow}}$"
        + f"  ($\\sigma_{{\\mathrm{{allow}}}} = {SIGMA_ALLOW_MPA:.0f}$ MPa)" + "\n"
        + r"$\theta_{\mathrm{PCH}} < \theta_{\mathrm{limit}}$"
        + f"  ($\\theta_{{\\mathrm{{limit}}}} = {THETA_LIMIT:.0f}" + r"^\circ$)" + "\n"
        + r"$R_{\mathrm{offset}} < R_{\mathrm{limit}}$"
        + f"  ($R_{{\\mathrm{{limit}}}} = {R_LIMIT:.0f}$ m)"
    )
    ax.text(0.02, 0.02, annotation_text, transform=ax.transAxes,
            fontsize=20, verticalalignment="bottom", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="black", linewidth=2.0, alpha=0.92))

    # ---- Data source watermark ----
    ax.text(0.98, 0.98,
            f"Data: {data_source}",
            transform=ax.transAxes, fontsize=14, fontstyle="italic",
            color="#888888", verticalalignment="top", horizontalalignment="right")

    # ---- Mark a typical design operating point ----
    hs_design = 1.8
    vr_design = 0.6
    ax.plot(hs_design, vr_design, marker="*", markersize=18,
            color="#1f77b4", markeredgecolor="black", markeredgewidth=1.5,
            zorder=10)
    ax.annotate(
        "Design point\n($H$=1.8 m, $v_{\\mathrm{rel}}$=0.6 m/s)",
        xy=(hs_design, vr_design), xytext=(hs_design + 0.8, vr_design + 0.15),
        fontsize=18, color="#1f77b4",
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#1f77b4", linewidth=1.5, alpha=0.9),
    )

    # ---- Axis styling (matched to plots 01-10) ----
    ax.set_xlabel(r"Equivalent regular wave height $H$ (m)", fontsize=40, labelpad=15)
    ax.set_ylabel(r"Deployment speed $v_{\mathrm{rel}}$ (m/s)", fontsize=40, labelpad=15)
    ax.set_xlim(hs_vals[0], hs_vals[-1])
    ax.set_ylim(vr_vals[0], vr_vals[-1])
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(4.0)
    ax.tick_params(direction="in", length=14, width=3.5, labelsize=36, pad=14)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    # ---- Save ----
    out_path = output_dir / "fig_11_operational_window.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight", facecolor="white")
    print(f"Saved SVG: {out_path}")

    plt.show()


# ==============================================================================
# Analytical fallback  (used only when sweep_results.npz is unavailable)
# ==============================================================================
def _compute_analytical(HS: np.ndarray, VR: np.ndarray):
    """
    Analytical constraint models calibrated to high-fidelity simulation physics.

    Stress σ (MPa): quasi-static + wave dynamic amplification + deployment-speed effect.
      - Static head: wire rope self-weight + payload at 1475 m depth.  ~101 MPa.
      - Wave effect:   σ ∝ H  (wave-induced dynamic stress envelope).
      - Speed effect:  σ ∝ v_rel² (hydrodynamic drag proportional to speed squared).

    Tilt θ (deg): PCH inclination from vertical, dominated by current-induced drag.

    Offset R (m): horizontal excursion, inversely proportional to deployment speed
      (fast descent → less time for lateral drift), linearly proportional to wave height.

    Coefficients are only a fallback when sweep_results.npz is unavailable.
    """
    A_WIRE = np.pi * (0.016) ** 2
    W_WET_PER_M = (7850 - 1025) * A_WIRE * 9.81
    PAYLOAD_WET = (700 - 1025 * 0.5) * 9.81
    SIGMA_STATIC = (W_WET_PER_M * 1475 + PAYLOAD_WET) / A_WIRE / 1e6

    sigma = SIGMA_STATIC + 18.0 * HS + 35.0 * VR ** 2
    theta = 2.0 + 0.5 * HS - 0.5 * VR + 10.0 * VR ** 2
    offset = 18.0 / np.maximum(VR, 1e-3) + 1.2 * HS
    return sigma, theta, offset


if __name__ == "__main__":
    main()
