from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

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
    ax.set_xlabel(xlabel, fontsize=38)
    ax.set_ylabel(ylabel, fontsize=38, labelpad=20)
    #ax.set_title(title, fontsize=35)
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(3.5)
    ax.tick_params(direction="in", length=12, width=3.5, labelsize=32, pad=16)


def main():
    payload = load_payload()
    times = payload["times"]
    height_above_seabed = payload["height_above_seabed"]
    target_height = float(payload["target_height"])
    wave_period = float(payload["wave_period"])

    duration = max(float(times[-1] - times[0]), 40.0)
    hover_window = min(max(4.0 * wave_period, 40.0), duration)
    hover_start = max(float(times[0]), float(times[-1] - hover_window))
    hover_mask = times >= hover_start

    t_hover = times[hover_mask]
    h_hover = height_above_seabed[hover_mask]
    h_min_hover = float(np.min(h_hover)) if len(h_hover) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(11.2, 8.0), constrained_layout=True)
    ax.plot(t_hover, h_hover, color="#009E73", linewidth=3.5, label="Height above seabed")
    ax.axhline(target_height, color="red", linestyle="--", linewidth=2.1, label="Target = 25 m")
    ax.axhline(
        h_min_hover,
        color="#D55E00",
        linestyle="--",
        linewidth=2.1,
        label=rf"$\mathbf{{h}}_{{\mathbf{{min}}}}$ = {h_min_hover:.2f} m",#
    )
    ax.fill_between(
        t_hover,
        h_hover,
        target_height,
        where=(h_hover < target_height),
        color="#D55E00",
        alpha=0.18,
    )

    overshoot = max(0.0, target_height - h_min_hover)
    ax.text(
        0.98,
        0.06,
        f"Overshoot = {overshoot:.2f} m",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=1,
        color="#D55E00",
    )

    style_axes(ax, "Time (s)", "Height above seabed (m)", "Final Hover-Stage Zoom-In")
    ax.legend(loc="upper right", frameon=True, edgecolor="black", facecolor="white", framealpha=1.0, prop={"size": 25, "weight": "bold"})

    out_path = _resolve_output_dir() / "fig_06_final_hover_stage_zoom_in.svg"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()



