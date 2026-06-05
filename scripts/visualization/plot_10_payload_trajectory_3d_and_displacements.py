from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(2.8)
    ax.tick_params(direction="in", length=10, width=2.0)


def main():
    payload = load_payload()
    times = payload["times"]
    payload_ref_xyz = payload["payload_ref_xyz"]
    seabed_z = float(payload["seabed_z"])
    target_z = float(payload["target_z"])

    x = payload_ref_xyz[:, 0]
    y = payload_ref_xyz[:, 1]
    z = payload_ref_xyz[:, 2]

    fig = plt.figure(figsize=(13.5, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d.plot(x, y, z, color="#0072B2", linewidth=2.6)
    ax3d.scatter(x[0], y[0], z[0], color="#009E73", s=80, marker="o", edgecolors="black", linewidths=0.7)
    ax3d.scatter(x[-1], y[-1], z[-1], color="#D55E00", s=90, marker="^", edgecolors="black", linewidths=0.7)

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    x_margin = max((x_max - x_min) * 0.05, 10.0)
    y_margin = max((y_max - y_min) * 0.05, 10.0)
    xx, yy = np.meshgrid([x_min - x_margin, x_max + x_margin], [y_min - y_margin, y_max + y_margin])
    ax3d.plot_surface(xx, yy, np.full_like(xx, seabed_z), alpha=0.13, color="#8B4513", shade=False)
    ax3d.plot_surface(xx, yy, np.full_like(xx, target_z), alpha=0.09, color="#228B22", shade=False)

    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D Payload Trajectory")
    ax3d.view_init(elev=24, azim=48)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("black")
        axis.pane.set_linewidth(2.4)
        if hasattr(axis, "line"):
            axis.line.set_color("black")
            axis.line.set_linewidth(2.4)
    ax3d.tick_params(axis="x", which="major", direction="in", length=10, width=2.0, pad=8)
    ax3d.tick_params(axis="y", which="major", direction="in", length=10, width=2.0, pad=8)
    ax3d.tick_params(axis="z", which="major", direction="in", length=10, width=2.0, pad=8)

    ax_x = fig.add_subplot(gs[0, 1])
    ax_x.plot(times, x, color="#0072B2", linewidth=2.8)
    style_axes(ax_x, "Time (s)", "X (m)", "X Displacement")

    ax_y = fig.add_subplot(gs[1, 0])
    ax_y.plot(times, y, color="#D55E00", linewidth=2.8)
    style_axes(ax_y, "Time (s)", "Y (m)", "Y Displacement")

    ax_z = fig.add_subplot(gs[1, 1])
    ax_z.plot(times, z, color="#009E73", linewidth=2.8)
    ax_z.axhline(seabed_z, color="#8B4513", linestyle="-", linewidth=1.9, label="Seabed")
    ax_z.axhline(target_z, color="red", linestyle="--", linewidth=1.9, label="Target")
    style_axes(ax_z, "Time (s)", "Z (m)", "Z Displacement")
    ax_z.legend(loc="upper right", frameon=True, edgecolor="black", facecolor="white", framealpha=1.0)

    out_path = _resolve_output_dir() / "fig_10_payload_trajectory_3d_and_displacements.svg"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()


