from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from subsea_deployment_simulation.plotting_style import apply_times_new_roman_style


apply_times_new_roman_style()


def _resolve_project_root():
    """Locate the project root by walking up from this script."""
    return Path(__file__).resolve().parents[2]


def _resolve_data_dir():
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


def _resolve_output_dir():
    """Return the directory for generated figures."""
    root = _resolve_project_root()
    out = root / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_payload(payload_path=None):
    if payload_path is not None:
        payload_path = Path(payload_path)
        if payload_path.exists():
            return np.load(payload_path, allow_pickle=True)
        raise FileNotFoundError(f"Visualization payload not found: {payload_path}")

    candidates = [
        _resolve_data_dir() / "visualization_payload.npz",
        Path.cwd() / "results" / "visualization" / "visualization_payload.npz",
        Path.cwd() / "visualization_payload.npz",
    ]

    for candidate in candidates:
        if candidate.exists():
            return np.load(candidate, allow_pickle=True)

    tried = "; ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Visualization payload not found. Tried: " + tried +
        ". Run high_fidelity_sim.py first to generate it."
    )


def style_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=40)
    ax.set_ylabel(ylabel, fontsize=40)
    ax.set_title(title, fontsize=1)
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(4)
    ax.tick_params(direction="in", length=15, width=4, labelsize=40, pad=15)


def main():
    payload = load_payload()
    times = payload["times"]
    top_xyz = payload["top_xyz"]
    vessel_xyz = payload["vessel_xyz"]
    wave_period = float(payload["wave_period"])

    duration = max(float(times[-1] - times[0]), 15.0)
    zoom_window = min(max(2.0 * wave_period, 15.0), duration)
    zoom_start = max(float(times[0]), float(times[-1] - zoom_window))
    zoom_mask = times >= zoom_start

    winch_heave = top_xyz[zoom_mask, 2] - np.mean(top_xyz[zoom_mask, 2])
    vessel_heave = vessel_xyz[zoom_mask, 2] - np.mean(vessel_xyz[zoom_mask, 2])

    fig, ax = plt.subplots(figsize=(12, 9.25), constrained_layout=True)
    ax.plot(times[zoom_mask], vessel_heave, color="#56B4E9", linewidth=4, label="Vessel CG heave")
    ax.plot(times[zoom_mask], winch_heave, color="#0072B2", linewidth=4, label="Winch point heave")
    style_axes(ax, "Time (s)", "Vertical heave (m)", "Upper-Boundary Heave Disturbance")
    ax.legend(loc="upper right", frameon=True, edgecolor="black", facecolor="white", 
            framealpha=1.0, prop={'size': 25, 'weight': 'bold'})
    out_path_emf = _resolve_output_dir() / "fig_01_upper_boundary_heave_disturbance.svg"
    fig.savefig(out_path_emf, format='svg', bbox_inches="tight", facecolor="white")
    print(f"Saved SVG: {out_path_emf}")
    plt.show()

if __name__ == "__main__":
    main()



