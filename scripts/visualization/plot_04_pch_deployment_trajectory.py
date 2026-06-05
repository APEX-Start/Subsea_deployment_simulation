from pathlib import Path
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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


def _style_projection_axis(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.5)
    ax.tick_params(direction="in", length=4, width=1.6, labelsize=12, pad=3)#子图坐标轴的粗细与长度
    ax.set_box_aspect(1)


def _style_colorbar(cbar, label):
    cbar.set_label(label, fontsize=14)
    cbar.ax.tick_params(direction="in", length=4, width=1.6, labelsize=12, pad =5, zorder=100) #色标刻度的粗细和字体大小
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(1.0)


def main():
    payload = load_payload()
    times = payload["times"]
    payload_ref_xyz = payload["payload_ref_xyz"]
    x = payload_ref_xyz[:, 0]
    y = payload_ref_xyz[:, 1]
    z = payload_ref_xyz[:, 2]

    # Keep point count close to the reference visual density.
    stride = max(1, len(times) // 50)
    xs = x[::stride]
    ys = y[::stride]
    zs = z[::stride]
    ts = times[::stride]

    fig = plt.figure(figsize=(8.0, 6.0), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, wspace=-0.2, hspace=0.175)
    fig.subplots_adjust(left=0.03, right=0.93, bottom=0.055, top=0.985)
    cmap = "RdYlBu_r"

    # Top-left: 3D trajectory.
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    if hasattr(ax3d, "set_computed_zorder"):
        ax3d.set_computed_zorder(False)
    else:
        ax3d.computed_zorder = False

    ax3d.plot(x, y, z, color="#024347", linewidth=1.6, alpha=0.55, zorder=5)
    sc3d = ax3d.scatter(
        xs,
        ys,
        zs,
        c=ts,
        cmap=cmap,
        s=34,
        alpha=0.92,
        edgecolors="black",
        linewidths=0.2,
        depthshade=False,
        zorder=10,
    )
    ax3d.set_xlabel("X (m)", fontsize=12, labelpad=1)#3d坐标轴标签的字体大小和与轴线的距离
    ax3d.set_ylabel("Y (m)", fontsize=12, labelpad=1)
    ax3d.set_zlabel("Z (m)", fontsize=12, labelpad=2)
    ax3d.view_init(elev=24, azim=24)

    # Match the standalone 3D figure's adaptive X/Y tick behavior.
    ax3d.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax3d.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax3d.xaxis.set_minor_locator(ticker.NullLocator())
    ax3d.yaxis.set_minor_locator(ticker.NullLocator())

    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis._axinfo["tick"]["inward_factor"] = -0.05
        axis._axinfo["tick"]["outward_factor"] = 0.3
        axis._axinfo["tick"]["linewidth"][True] = 2.5#3d坐标轴的刻度线粗细

    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("black")
        axis.pane.set_linewidth(1.4)#3d坐标轴的粗细
        if hasattr(axis, "line"):
            axis.line.set_color("black")
            axis.line.set_linewidth(1.4)
    ax3d.tick_params(axis="x", which="major", direction="out", pad=0, labelsize=12, width=0.8) #3d坐标轴的刻度线粗细和标签字体大小
    ax3d.tick_params(axis="y", which="major", direction="out", pad=0, labelsize=12, width=0.8)
    ax3d.tick_params(axis="z", which="major", direction="out", pad=2, labelsize=12, width=0.8)
    ax3d.tick_params(axis="z", which="minor", direction="out", length=15, width=0.1, pad=12)
    ax3d.grid(False)
    if hasattr(ax3d, "set_box_aspect"):
        ax3d.set_box_aspect((1, 1, 1))

    # Start/end markers are drawn last so the final triangle stays on top.
    ax3d.scatter(
        x[0],
        y[0],
        z[0],
        color="#009E73",
        s=54,
        edgecolors="none",
        linewidths=0,
        depthshade=False,
        zorder=100,
    )
    ax3d.scatter(
        x[-1],
        y[-1],
        z[-1],
        color="#D55E00",
        s=60,
        marker="^",
        edgecolors="none",
        linewidths=0,
        depthshade=False,
        zorder=1000,
    )

    # Shift the 3D panel left to match the requested layout.
    shift = 0
    pos3d = ax3d.get_position()
    ax3d.set_position([pos3d.x0 + shift, pos3d.y0, pos3d.width, pos3d.height])

    # Top-right: X-Z view.
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_xz.plot(x, z, color="#0C2091", linewidth=1.8, alpha=0.62, zorder=3)
    ax_xz.scatter(xs, zs, c=ts, cmap=cmap, s=26, alpha=1, edgecolors="#000000", linewidths=0.2, zorder=8)
    _style_projection_axis(ax_xz, "X (m)", "Z (m)")

    # Bottom-left: Y-Z view.
    ax_yz = fig.add_subplot(gs[1, 0])
    ax_yz.plot(y, z, color="#0C2091", linewidth=1.8, alpha=0.62, zorder=3)
    ax_yz.scatter(ys, zs, c=ts, cmap=cmap, s=26, alpha=1, edgecolors="#000000", linewidths=0.2, zorder=8)
    _style_projection_axis(ax_yz, "Y (m)", "Z (m)")

    # Bottom-right: X-Y view.
    ax_xy = fig.add_subplot(gs[1, 1])
    ax_xy.plot(x, y, color="#0C2091", linewidth=1.8, alpha=0.62, zorder=3)
    ax_xy.scatter(xs, ys, c=ts, cmap=cmap, s=26, alpha=1, edgecolors="#000000", linewidths=0.2, zorder=8)
    _style_projection_axis(ax_xy, "X (m)", "Y (m)")

    # Global colorbar on the far right of the full figure, with shorter length.
    cax = fig.add_axes([0.875, 0.22, 0.018, 0.56])# Adjusted to be taller and more centered vertically.
    cbar3d = fig.colorbar(sc3d, cax=cax)
    _style_colorbar(cbar3d, "Time (s)")

    out_path = _resolve_output_dir() / "fig_04_pch_deployment_trajectory.svg"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="black")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()



