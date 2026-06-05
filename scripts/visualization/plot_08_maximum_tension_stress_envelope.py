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
    ax.set_ylabel(ylabel, fontsize=38)
    # ax.set_title(title, fontsize=38)
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(3.5)
    ax.tick_params(direction="in", length=12, width=3.5, labelsize=32, pad=16)


def main():
    payload = load_payload()
    depth_centers = payload["depth_centers"]
    max_tension_envelope = payload["max_tension_envelope"]
    max_stress_envelope = payload["max_stress_envelope"]
    seabed_z = float(payload["seabed_z"])

    valid = np.isfinite(max_tension_envelope) & np.isfinite(max_stress_envelope)

    fig, ax = plt.subplots(figsize=(11.2, 9), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.patch.set_linewidth(3.5)
    line_tension, = ax.plot(
        max_tension_envelope[valid] / 1000.0,
        depth_centers[valid],
        color="#0072B2",
        linewidth=3.0,
        label="Max tension envelope",
    )
    ax.axhline(seabed_z, color="#8B4513", linestyle="-", linewidth=2.1)
    style_axes(ax, "Tension (kN)", "Z (m)", "Maximum Tension/Stress Envelope")

    ax_top = ax.twiny()
    line_stress, = ax_top.plot(
        max_stress_envelope[valid] / 1e6,
        depth_centers[valid],
        color="#D55E00",
        linewidth=2.6,
        linestyle="--",
        label="Max stress envelope",
    )
    ax_top.set_xlabel("Stress (MPa)", fontsize=38)
    ax_top.tick_params(direction="in", length=12, width=3.5, labelsize=32, pad=8)
    for spine in ax_top.spines.values():
        spine.set_color("black")
        spine.set_linewidth(3.5)

    ax.legend(
        [line_tension, line_stress],
        ["Max tension envelope", "Max stress envelope"],
        loc="upper left",#微调图例位置以避免与轴标签重叠
        bbox_to_anchor=(0.002, 1.0), #微调图例位置以避免与轴标签重叠
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        prop={"size": 25, "weight": "bold"},
    )

    out_path = _resolve_output_dir() / "fig_08_maximum_tension_stress_envelope.svg"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()



