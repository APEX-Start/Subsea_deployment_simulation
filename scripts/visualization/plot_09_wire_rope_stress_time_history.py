from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from subsea_deployment_simulation.plotting_style import apply_times_new_roman_style


apply_times_new_roman_style()


# Limit plotted samples for cleaner stress curves.
MAX_PLOT_POINTS = 400


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
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)
    # ax.set_title(title, fontsize=38)
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(3.0)
    ax.tick_params(direction="in", length=12, width=3.0, labelsize=25, pad=10)


def sparsify_for_plot(times, *series, max_points=MAX_PLOT_POINTS):
    """Evenly downsample plotted data while preserving overall trend."""
    times = np.asarray(times)
    if times.ndim != 1:
        raise ValueError("times must be a 1D array")

    n = times.size
    if n == 0:
        return (times, *[np.asarray(s) for s in series])
    if n <= max_points:
        return (times, *[np.asarray(s) for s in series])

    idx = np.unique(np.linspace(0, n - 1, max_points, dtype=int))
    sparse_times = times[idx]
    sparse_series = [np.asarray(s)[idx] for s in series]
    return (sparse_times, *sparse_series)


def load_required_series(payload, key, expected_shape):
    if key not in payload.files:
        raise KeyError(
            f"Visualization payload is missing '{key}'. "
            "Run high_fidelity_sim.py again to export filtered stress fields."
        )
    values = np.asarray(payload[key], dtype=float) / 1e6
    if values.shape != expected_shape:
        raise ValueError(f"Payload field '{key}' has shape {values.shape}, expected {expected_shape}.")
    return values


def main():
    payload = load_payload()
    times = np.asarray(payload["times"], dtype=float)
    raw_max_stress = payload["raw_max_stress"] / 1e6
    raw_mean_stress = payload["raw_mean_stress"] / 1e6
    filtered_max_stress = load_required_series(payload, "filtered_max_stress", raw_max_stress.shape)
    filtered_mean_stress = load_required_series(payload, "filtered_mean_stress", raw_mean_stress.shape)

    times, raw_max_stress, filtered_max_stress, raw_mean_stress, filtered_mean_stress = sparsify_for_plot(
        times,
        raw_max_stress,
        filtered_max_stress,
        raw_mean_stress,
        filtered_mean_stress,
        max_points=MAX_PLOT_POINTS,
    )

    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
    ax.plot(times, raw_max_stress, color="#D55E00", linewidth=1.4, alpha=0.38, label="Max stress")
    ax.plot(times, filtered_max_stress, color="#D55E00", linewidth=3.4, label="Max stress-filtered")
    ax.plot(times, raw_mean_stress, color="#0072B2", linewidth=1.4, alpha=0.42, label="Mean stress")
    ax.plot(times, filtered_mean_stress, color="#0072B2", linewidth=3.4, label="Mean stress-filtered")

    style_axes(ax, "Time (s)", "Stress (MPa)", "Wire-Rope Stress Time History")
    ax.legend(
        loc="upper left",
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        prop={"size": 20, "weight": "bold"},
    )

    out_path = _resolve_output_dir() / "fig_09_wire_rope_stress_time_history.svg"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()



