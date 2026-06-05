from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import FormatStrFormatter

from subsea_deployment_simulation.plotting_style import apply_times_new_roman_style


apply_times_new_roman_style()


PAYLOAD_COLOR = "#D55E00"
INSET_COLOR = "#E8752A"
CONNECTOR_COLOR = "#002B65"


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
    ax.set_xlabel(xlabel, fontsize=35)
    ax.set_ylabel(ylabel, fontsize=35)
    ax.set_title(title, fontsize=1)
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(4)
    ax.tick_params(direction="in", length=15, width=4, labelsize=35, pad=15)
    ax.legend(
        loc="upper right",
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        prop={"size": 25, "weight": "bold"},
    )


def _style_inset_axes(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(3.2)

    ax.xaxis.set_ticks_position("top")
    ax.tick_params(
        axis="x",
        direction="in",
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        length=8,
        width=3.2,
        labelsize=25,
        pad=5,
    )
    ax.tick_params(
        axis="y",
        direction="in",
        left=True,
        right=False,
        length=8,
        width=3.2,
        labelsize=25,
        pad=5,
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def _padded_ylim(values, pad_fraction=0.14, min_span=0.2):
    ymin = float(np.min(values))
    ymax = float(np.max(values))
    span = max(ymax - ymin, min_span)
    pad = span * pad_fraction
    return ymin - pad, ymax + pad


def _even_meter_ticks(values, step=2.0, max_ticks=2):
    ymin = float(np.min(values))
    ymax = float(np.max(values))
    start = np.ceil(ymin / step) * step
    stop = np.floor(ymax / step) * step
    if start > stop:
        center = round(float(np.mean(values)) / step) * step
        return [center]

    ticks = list(np.arange(start, stop + 0.5 * step, step))
    if len(ticks) > max_ticks:
        indices = np.linspace(0, len(ticks) - 1, max_ticks, dtype=int)
        ticks = [ticks[i] for i in indices]
    return ticks


def _plot_zoomed_payload(inset_ax, times, z_values, xlim, xticks, yticks=None, yfmt=None):
    zoom_mask = (times >= xlim[0]) & (times <= xlim[1])
    t_zoom = times[zoom_mask]
    z_zoom = z_values[zoom_mask]
    if len(t_zoom) == 0:
        raise ValueError(f"No payload samples found in zoom interval {xlim[0]}-{xlim[1]} s.")

    marker_step = max(1, int(round(len(t_zoom) / 32)))
    inset_ax.plot(
        t_zoom,
        z_zoom,
        color=INSET_COLOR,
        linewidth=2.8,
        marker="o",
        markersize=8.5,
        markerfacecolor="white",
        markeredgecolor=INSET_COLOR,
        markeredgewidth=2.4,
        markevery=marker_step,
    )
    inset_ax.set_xlim(*xlim)
    inset_ax.set_xticks(xticks)
    inset_ax.set_ylim(*_padded_ylim(z_zoom))
    inset_ax.set_yticks(yticks if yticks is not None else _even_meter_ticks(z_zoom))
    if yfmt is not None:
        inset_ax.yaxis.set_major_formatter(FormatStrFormatter(yfmt))
    _style_inset_axes(inset_ax)


def _add_zoom_callout(fig, ax, inset_ax, box, inset_corners, box_corners):
    x0, y0, width, height = box
    rect = Rectangle(
        (x0, y0),
        width,
        height,
        fill=False,
        edgecolor=CONNECTOR_COLOR,
        linewidth=3.0,
        linestyle=(0, (1.0, 2.0)),
        zorder=6,
    )
    rect.set_capstyle("round")
    rect.set_joinstyle("round")
    ax.add_patch(rect)

    for inset_corner, box_corner in zip(inset_corners, box_corners):
        connector = ConnectionPatch(
            xyA=inset_corner,
            coordsA=inset_ax.transAxes,
            xyB=box_corner,
            coordsB=ax.transData,
            color=CONNECTOR_COLOR,
            linewidth=3.0,
            linestyle=(0, (1.0, 2.0)),
            zorder=5,
            clip_on=False,
        )
        connector.set_capstyle("round")
        fig.add_artist(connector)


def main():
    payload = load_payload()
    times = payload["times"]
    payload_ref_xyz = payload["payload_ref_xyz"]
    payload_z = payload_ref_xyz[:, 2]
    target_z = float(payload["target_z"])
    seabed_z = float(payload["seabed_z"])
    target_reach_time = float(payload["target_reach_time"])

    fig, ax = plt.subplots(figsize=(11.2, 8.0), constrained_layout=True)
    ax.plot(times, payload_z, color=PAYLOAD_COLOR, linewidth=3.0, label="Payload reference point")
    ax.axhline(target_z, color="red", linestyle="--", linewidth=2.1, label=f"Target Z = {target_z:.0f} m")
    ax.axhline(seabed_z, color="#8B4513", linestyle="-", linewidth=2.1, label="Seabed")
 #   if np.isfinite(target_reach_time):
  #      ax.axvline(target_reach_time, color="#777777", linestyle="--", linewidth=1.8)

    ax.set_xlim(-130, 2600)
    ax.set_ylim(-1540, 0)
    ax.set_xticks([0, 500, 1000, 1500, 2000, 2500])
    ax.set_yticks([0, -500, -1000, -1500])
    style_axes(ax, "Time (s)", "Z (m)", "Payload Vertical Coordinate History")

    mid_inset = ax.inset_axes([0.12, 0.085, 0.43, 0.32])
    _plot_zoomed_payload(mid_inset, times, payload_z, (1000, 1010), [1004, 1008])

    final_inset = ax.inset_axes([0.66, 0.365, 0.30, 0.29])
    _plot_zoomed_payload(
        final_inset,
        times,
        payload_z,
        (2400, 2420),
        [2409, 2419],
        yticks=[target_z - 0.05, target_z - 0.15],
        yfmt="%.2f",
    )

    mid_mask = (times >= 1000) & (times <= 1010)
    final_mask = (times >= 2400) & (times <= 2420)
    mid_center_z = float(np.mean(payload_z[mid_mask]))
    final_center_z = float(np.mean(payload_z[final_mask]))
    mid_box = (920, mid_center_z - 45, 170, 90)
    final_box = (2370, final_center_z - 45, 170, 90)
    _add_zoom_callout(
        fig,
        ax,
        mid_inset,
        mid_box,
        inset_corners=[(0.0, 1.0), (1.0, 1.0)],
        box_corners=[(mid_box[0], mid_box[1]), (mid_box[0] + mid_box[2], mid_box[1])],
    )
    _add_zoom_callout(
        fig,
        ax,
        final_inset,
        final_box,
        inset_corners=[(0.0, 0.0), (1.0, 0.0)],
        box_corners=[
            (final_box[0], final_box[1] + final_box[3]),
            (final_box[0] + final_box[2], final_box[1] + final_box[3]),
        ],
    )

    out_path = _resolve_output_dir() / "fig_02_payload_vertical_coordinate_history.svg"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()



