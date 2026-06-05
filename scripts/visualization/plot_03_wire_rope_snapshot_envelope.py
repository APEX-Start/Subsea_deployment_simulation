from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from subsea_deployment_simulation.plotting_style import apply_times_new_roman_style


apply_times_new_roman_style()


LINE_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
]


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
    ax.set_xlabel(xlabel, fontsize=45)
    ax.set_ylabel(ylabel, fontsize=45)
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(4)
    ax.tick_params(direction="in", length=12, width=4, labelsize=38, pad=10)


def main():
    payload = load_payload()
    times = payload["times"]
    positions_history = payload["positions_history"]
    payload_ref_xyz = payload["payload_ref_xyz"]
    snapshot_indices = payload["snapshot_indices"].astype(int)
    flow_dir_unit = payload["flow_dir_unit"]
    seabed_z = float(payload["seabed_z"])
    target_z = float(payload["target_z"])

    if np.linalg.norm(flow_dir_unit) < 1e-12:
        flow_dir_unit = np.array([1.0, 0.0])

    fig, ax = plt.subplots(figsize=(12, 9.25), constrained_layout=True)

    for i, idx in enumerate(snapshot_indices):
        pos = np.asarray(positions_history[int(idx)], dtype=float)
        s_coord = pos[:, 0] * flow_dir_unit[0] + pos[:, 1] * flow_dir_unit[1]
        payload_s = (
            payload_ref_xyz[int(idx), 0] * flow_dir_unit[0]
            + payload_ref_xyz[int(idx), 1] * flow_dir_unit[1]
        )
        color = LINE_COLORS[i % len(LINE_COLORS)]
        ax.plot(s_coord, pos[:, 2], color=color, linewidth=4, label=f"t = {times[int(idx)]:.0f} s")
        ax.scatter(
            payload_s,
            payload_ref_xyz[int(idx), 2],
            s=40,
            color=color,
            edgecolors="black",
            linewidths=0.7,
            zorder=4,
        )

    ax.axhline(seabed_z, color="#8B4513", linestyle="-", linewidth=2)
    ax.axhline(target_z, color="#009E73", linestyle="--", linewidth=2)
    style_axes(ax, "Along-current offset S (m)", "Z (m)", "Wire-Rope Snapshot Envelope")
    ax.set_ylim(-1550, 50)  # 设置 Z 轴从 -1550 到 0
    ax.legend(
        loc="lower left",
        borderaxespad=1.8,
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        prop={"size": 25, "weight": "bold"},
    )

    out_path = _resolve_output_dir() / "fig_03_wire_rope_snapshot_envelope.svg"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()



