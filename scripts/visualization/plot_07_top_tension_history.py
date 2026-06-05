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
#    ax.set_title(title, fontsize=38)
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(3.5)
    ax.tick_params(direction="in", length=12, width=3.5, labelsize=32, pad=16)


def _load_optional_series(payload, key, n):
    if key not in payload.files:
        return np.full(n, np.nan, dtype=float)
    arr = np.asarray(payload[key], dtype=float)
    if arr.shape[0] != n:
        return np.full(n, np.nan, dtype=float)
    return arr


def _has_valid(arr):
    return bool(np.any(np.isfinite(arr)))


def main():
    payload = load_payload()
    times = np.asarray(payload["times"], dtype=float)
    n = len(times)

    top_tension = np.asarray(payload["top_tension"], dtype=float)
    static_top_tension = np.asarray(payload["static_top_tension"], dtype=float)

    atc_enabled = False
    if "active_tension_control_enabled" in payload.files:
        atc_enabled = bool(np.asarray(payload["active_tension_control_enabled"]).item())

    top_tension_controlled = _load_optional_series(payload, "top_tension_controlled", n)
    top_tension_open_loop = _load_optional_series(payload, "top_tension_open_loop", n)
    tension_control_target = _load_optional_series(payload, "tension_control_target", n)

    # 向后兼容：老 payload 只有 top_tension 时，根据开关语义映射到对应曲线
    if not _has_valid(top_tension_controlled) and atc_enabled:
        top_tension_controlled = top_tension.copy()
    if not _has_valid(top_tension_open_loop) and (not atc_enabled):
        top_tension_open_loop = top_tension.copy()

    target_reach_time = float(payload["target_reach_time"])

    fig, ax = plt.subplots(figsize=(11.2, 8.2), constrained_layout=True)
    ax.plot(
        times,
        static_top_tension / 1000.0,
        color="#E69F00",
        linewidth=2.8,
        zorder=10,
        label="Static submerged weight",
    )

    if _has_valid(top_tension_open_loop):
        ax.plot(
            times,
            top_tension_open_loop / 1000.0,
            color="#666666",
            linestyle="--",
            linewidth=2.2,
            zorder=3,
            label="Top tension (without ATC)",
        )

    if _has_valid(top_tension_controlled):
        ax.plot(
            times,
            top_tension_controlled / 1000.0,
            color="#000000",
            linewidth=2.5,
            zorder=4,
            label="Top tension (with ATC)",
        )
    else:
        ax.plot(
            times,
            top_tension / 1000.0,
            color="#000000",
            linewidth=2.4,
            zorder=4,
            label="Total dynamic top tension",
        )

    if _has_valid(tension_control_target):
        ax.plot(
            times,
            tension_control_target / 1000.0,
            color="#0072B2",
            linestyle=":",
            linewidth=2.0,
            zorder=5,
            label="ATC target tension",
        )

    if np.isfinite(target_reach_time):
        ax.axvline(target_reach_time, color="#777777", linestyle="--", linewidth=1.8, zorder=1)

    style_axes(ax, "Time (s)", "Top tension (kN)", "Top-Tension History")
    ax.legend(loc="upper left", frameon=True, edgecolor="black", facecolor="white", framealpha=1.0, prop={"size": 25, "weight": "bold"})

    out_path = _resolve_output_dir() / "fig_07_top_tension_history.svg"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()



