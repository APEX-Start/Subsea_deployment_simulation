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
    ax.set_xlabel(xlabel, fontsize=35)
    ax.set_ylabel(ylabel, fontsize=35, labelpad=10)  # 增加 labelpad 参数调整 y 轴标签与轴线的距离
   # ax.set_title(title, fontsize=35)
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(4)
    ax.tick_params(direction="in", length=12, width=4, labelsize=35)  # 添加 labelsize 参数



def main():
    payload = load_payload()
    times = payload["times"]
    pch_tilt = payload["pch_tilt"]
    tool_tilt = payload["tool_tilt"]

    fig, ax = plt.subplots(figsize=(11.2, 8.5), constrained_layout=True)
    ax.plot(times, pch_tilt, color="#0072B2", linewidth=3.5, label=r"$\theta_{PCH}$")
    #ax.plot(times, tool_tilt, color="#D55E00", linewidth=3, linestyle="--", label=r"$\theta_{Tool}$")

    if np.allclose(pch_tilt, tool_tilt, atol=1e-8, rtol=1e-8):
        ax.text(
            0.98,
            0.05,
            "Current model: nearly rigid coupling",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=1,
            color="#666666",
        )

    style_axes(ax, "Time (s)", "Tilt angle (deg)", "PCH and Tool-String Tilt Histories")
    ax.legend(
        loc="upper right",
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        prop={"size": 30, "weight": "bold"},
    )

    out_path = _resolve_output_dir() / "fig_05_pch_and_tool_tilt_histories.svg"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()



