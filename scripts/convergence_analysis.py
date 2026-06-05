# -*- coding: utf-8 -*-
r"""
论文验证导向的收敛性分析模块。

核心功能：
1. 不同工况下网格收敛性分析对比（下放时间、偏移、应力、求解时间）
2. 静态悬链线（解析）与数值稳态形状对比
3. 自动导出论文图表与 CSV 数据

运行示例（在项目根目录执行）：

    # 语法检查（不运行仿真）
    python -m py_compile scripts/convergence_analysis.py

    # 仅运行静态悬链线验证（快速）
    python scripts/convergence_analysis.py --mode catenary

    # 快速烟测模式（少量网格点）
    python scripts/convergence_analysis.py --mode mesh --quick

    # 完整网格收敛性分析（p95 应力指标，默认）
    python scripts/convergence_analysis.py --mode mesh

    # 使用原始最大应力指标
    python scripts/convergence_analysis.py --mode mesh --stress-metric max

    # 平滑应力 + 自定义参考段数
    python scripts/convergence_analysis.py --mode mesh --stress-metric p95 --smooth-stress --reference-n 80

    # 全部运行（mesh + catenary）
    python scripts/convergence_analysis.py --mode all

    # 自定义输出目录
    python scripts/convergence_analysis.py --mode mesh --output-dir results/convergence

可用参数：
    --mode          执行模式：all | mesh | catenary（默认 all）
    --quick         快速烟测模式（少量网格点，短松弛时长）
    --stress-metric 应力指标：p95（默认）| max
    --smooth-stress 使用平滑应力历史
    --reference-n   参考网格段数 N_ref（默认 40）
    --error-threshold  收敛误差阈值（默认 0.05）
    --relax-time    静态松弛仿真时长/s（默认 900）
    --n-eval        静态采样点数（默认 280）
    --dt-output     动态输出步长/s（默认 1.0）
    --output-dir    输出目录
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

# Ensure the package is importable even without `pip install -e .`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

def setup_publication_style(show_message: bool = True):
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16
    })

from subsea_deployment_simulation.high_fidelity_sim import (
    DEFAULT_CONFIG as HIGH_FIDELITY_DEFAULT_CONFIG,
    DynamicWireRopeSystem3D,
    WireRopeSystem3D,
)

# 复用项目已有出版级样式配置
setup_publication_style(show_message=False)

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = MODULE_DIR / "results" / "paper_convergence_validation"
EPS = 1e-8


@dataclass(frozen=True)
class MeshScenario:
    """动态下放多工况定义。"""

    key: str
    name_en: str
    name_cn: str
    color: str
    marker: str
    release_speed: float
    rope_diameter: float
    winch_x: float
    wave_height: float
    wave_period: float
    current_scale: float


@dataclass(frozen=True)
class StaticCurrentCase:
    """静态悬链线对比用工况定义。"""

    key: str
    name_en: str
    name_cn: str
    color: str
    marker: str
    current_scale: float
    water_depth: float = 600.0
    rope_length: float = 500.0
    node_count: int = 36


class ScaledCurrentDynamicSystem(DynamicWireRopeSystem3D):
    """对 `high_fidelity_sim.py` 动态系统增加海流缩放系数。"""

    def __init__(self, config: Optional[Dict] = None):
        cfg = dict(config or {})
        self.current_scale = float(cfg.pop("current_scale", 1.0))
        super().__init__(cfg)
        self.config["current_scale"] = self.current_scale

    def current_velocity(self, pos):
        return self.current_scale * super().current_velocity(pos)


class ScaledCurrentStaticSystem(WireRopeSystem3D):
    """对基类静态系统增加海流缩放系数。"""

    def __init__(self, config: Optional[Dict] = None):
        cfg = dict(config or {})
        self.current_scale = float(cfg.pop("current_scale", 1.0))
        super().__init__(cfg)
        self.config["current_scale"] = self.current_scale

    def current_velocity(self, pos):
        return self.current_scale * super().current_velocity(pos)


def prepare_output_dirs(output_dir: Optional[str] = None) -> Dict[str, Path]:
    """创建统一输出目录结构。"""
    root = Path(output_dir) if output_dir else DEFAULT_OUTPUT_ROOT
    data_dir = root / "data"
    figures_dir = root / "figures"

    for path in (root, data_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "data": data_dir,
        "figures": figures_dir,
    }


def compute_relative_error(value: float, reference: float, epsilon: float = EPS) -> float:
    """鲁棒相对误差：|v-r| / max(|v|, |r|, epsilon)。"""
    denominator = max(abs(float(value)), abs(float(reference)), float(epsilon))
    return abs(float(value) - float(reference)) / denominator


def estimate_convergence_order(
    n_values: Sequence[float],
    errors: Sequence[float],
) -> Tuple[float, float]:
    """估计误差关于网格数的收敛阶：e ~ N^{-p}，返回 (p, R²)。"""
    n_arr = np.asarray(n_values, dtype=float)
    e_arr = np.asarray(errors, dtype=float)
    valid = (n_arr > 0.0) & np.isfinite(n_arr) & (e_arr > 1e-12) & np.isfinite(e_arr)

    if np.sum(valid) < 2:
        return np.nan, np.nan

    log_n = np.log(n_arr[valid])
    log_e = np.log(e_arr[valid])
    coeff = np.polyfit(log_n, log_e, 1)
    pred = np.polyval(coeff, log_n)

    ss_res = np.sum((log_e - pred) ** 2)
    ss_tot = np.sum((log_e - np.mean(log_e)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    p = -float(coeff[0])
    return p, r2


def project_curve_to_plane(positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 3D 节点投影到“主水平偏移方向-竖向”平面。

    返回：
        x: 沿主水平方向坐标
        z: 竖向坐标（向下为负）
        s: 弧长坐标
    """
    n_nodes = positions.shape[0]
    if n_nodes < 2:
        x = np.zeros(n_nodes)
        z = np.zeros(n_nodes)
        s = np.zeros(n_nodes)
        return x, z, s

    delta = positions - positions[0]
    tail_h = delta[-1, :2]
    norm_h = np.linalg.norm(tail_h)

    if norm_h > 1e-12:
        e_h = tail_h / norm_h
    else:
        e_h = np.array([1.0, 0.0])

    x = delta[:, 0] * e_h[0] + delta[:, 1] * e_h[1]
    z = delta[:, 2]

    s = np.zeros(n_nodes)
    for i in range(1, n_nodes):
        s[i] = s[i - 1] + np.linalg.norm(positions[i] - positions[i - 1])

    return x, z, s


def _solve_straight_line(
    length: float,
    dx: float,
    dz: float,
    wet_weight_per_length: float,
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """近似垂直工况或求解失败时的直线解。"""
    x = np.linspace(0.0, dx, n_points)
    z = np.linspace(0.0, dz, n_points)
    s = np.linspace(0.0, length, n_points)
    tension = wet_weight_per_length * np.maximum(length - s, 0.0)
    return x, z, s, tension


def solve_analytic_catenary(
    length: float,
    dx: float,
    dz: float,
    wet_weight_per_length: float,
    n_points: int = 400,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    求解解析悬链线（二维）。

    坐标约定：
    - 顶端为 (0, 0)
    - x 为水平，z 为竖向（向下为负）
    """
    if length <= 0.0:
        raise ValueError("length 必须为正")

    # Quick straight-line fallback for degenerate cases
    dx_abs = abs(dx)
    if dx_abs < 1e-8 or dx_abs < 1e-6 * max(1.0, length):
        return _solve_straight_line(length, dx, dz, wet_weight_per_length, n_points)

    # Use a robust two-variable least-squares solve for (a, x_center)
    from scipy.optimize import least_squares

    def resid(params: np.ndarray) -> np.ndarray:
        a, x_center = params
        if a <= 0.0 or not np.isfinite(a):
            return np.array([1e6, 1e6])
        z_end = a * (np.cosh((dx - x_center) / a) - np.cosh((-x_center) / a))
        s_total = a * (np.sinh((dx - x_center) / a) - np.sinh((-x_center) / a))
        return np.array([z_end - dz, s_total - length], dtype=float)

    a_init = max(abs(dz) / 2.0, length / 10.0)
    x_center_init = dx / 2.0

    # Multi-start least-squares to improve robustness
    best_res = None
    best_rnorm = float("inf")
    starts = [a_init, max(length / 2.0, 1.0), max(abs(dz) / 2.0, length / 20.0)]
    for a0 in starts:
        try:
            res = least_squares(
                resid,
                x0=np.array([float(a0), float(x_center_init)], dtype=float),
                bounds=([1e-12, -np.inf], [np.inf, np.inf]),
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                max_nfev=2000,
            )
            rnorm = float(np.linalg.norm(res.fun))
            if res.success and rnorm < best_rnorm:
                best_rnorm = rnorm
                best_res = res
        except Exception:
            continue

    if best_res is None:
        return _solve_straight_line(length, dx, dz, wet_weight_per_length, n_points)

    a, x_center = float(best_res.x[0]), float(best_res.x[1])

    # verify residuals are small; otherwise treat as failure
    z_end_calc = a * (np.cosh((dx - x_center) / a) - np.cosh((-x_center) / a))
    s_total_calc = a * (np.sinh((dx - x_center) / a) - np.sinh((-x_center) / a))
    if not (np.isfinite(z_end_calc) and np.isfinite(s_total_calc)):
        return _solve_straight_line(length, dx, dz, wet_weight_per_length, n_points)
    if abs(z_end_calc - dz) > 1e-3 or abs(s_total_calc - length) > 1e-3:
        return _solve_straight_line(length, dx, dz, wet_weight_per_length, n_points)

    x = np.linspace(0.0, dx, int(max(2, n_points)))
    z = a * (np.cosh((x - x_center) / a) - np.cosh((-x_center) / a))

    # Diagnostic info (helpful to compare analytic vs numerical)
    try:
        diag_msg = (
            f"[analytic diag] a={a:.6g}, x_center={x_center:.6g}, z_end_calc={z_end_calc:.6g}, "
            f"s_total_calc={s_total_calc:.6g}"
        )
        print(diag_msg)
    except Exception:
        pass

    s = np.zeros(n_points)
    for i in range(1, n_points):
        s[i] = s[i - 1] + np.hypot(x[i] - x[i - 1], z[i] - z[i - 1])

    dz_dx = np.sinh((x - x_center) / a)
    tension = wet_weight_per_length * a * np.sqrt(1.0 + dz_dx ** 2)

    return x, z, s, tension


def interpolate_curve_by_arc(
    s_source: np.ndarray,
    x_source: np.ndarray,
    z_source: np.ndarray,
    s_target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """将源曲线按归一化弧长插值到目标弧长网格。"""
    s_src = np.asarray(s_source, dtype=float)
    x_src = np.asarray(x_source, dtype=float)
    z_src = np.asarray(z_source, dtype=float)
    s_tgt = np.asarray(s_target, dtype=float)

    if len(s_src) == 0 or len(s_tgt) == 0:
        return np.array([]), np.array([])

    src_norm = s_src / max(float(s_src[-1]), EPS)
    tgt_norm = s_tgt / max(float(s_tgt[-1]), EPS)

    src_norm_unique, unique_idx = np.unique(src_norm, return_index=True)
    x_unique = x_src[unique_idx]
    z_unique = z_src[unique_idx]

    if len(src_norm_unique) < 2:
        return np.full_like(tgt_norm, x_unique[0]), np.full_like(tgt_norm, z_unique[0])

    fx = interp1d(src_norm_unique, x_unique, kind="linear", fill_value="extrapolate")
    fz = interp1d(src_norm_unique, z_unique, kind="linear", fill_value="extrapolate")
    return fx(tgt_norm), fz(tgt_norm)


def _arc_length_from_positions(positions: np.ndarray) -> np.ndarray:
    s = np.zeros(positions.shape[0], dtype=float)
    for i in range(1, positions.shape[0]):
        s[i] = s[i - 1] + float(np.linalg.norm(positions[i] - positions[i - 1]))
    return s


def _payload_wet_weight(system: WireRopeSystem3D) -> float:
    rho_w = float(system.config["rho_w"])
    g = float(system.config["g"])
    return float(
        (system.pch_mass - rho_w * system.pch_volume) * g
        + (system.tool_mass - rho_w * system.tool_volume) * g
    )


def solve_no_current_catenary_reference(
    system: ScaledCurrentStaticSystem,
    n_points: int = 400,
) -> Dict[str, object]:
    """Closed-form vertical catenary limit for the no-current static case."""
    cfg = system.config
    n_calc = int(max(3, n_points))
    length = float(cfg["L"])
    l0 = length / float(n_calc - 1)
    E = float(cfg["E"]) * float(cfg.get("EA_factor", 1.0))
    A = float(system.A)
    wet_weight_per_length = (
        (float(cfg["rho_c"]) - float(cfg["rho_w"])) * A * float(cfg["g"])
    )
    payload_wet = _payload_wet_weight(system)

    positions = np.zeros((n_calc, 3), dtype=float)
    tension = np.empty(n_calc, dtype=float)
    for j in range(n_calc - 1):
        segment_tension = payload_wet + wet_weight_per_length * l0 * (
            (n_calc - 1 - j) - 0.5
        )
        tension[j] = segment_tension
        strain = segment_tension / max(E * A, EPS)
        positions[j + 1, 2] = positions[j, 2] - l0 * (1.0 + strain)

    tension[-1] = payload_wet
    return {
        "positions": positions,
        "s": _arc_length_from_positions(positions),
        "tension": tension,
        "top_tension": payload_wet + wet_weight_per_length * length,
        "iterations": 1,
    }


def solve_quasi_static_drag_reference(
    system: ScaledCurrentStaticSystem,
    n_points: int = 400,
    max_iter: int = 1200,
    tol: float = 1e-6,
    relax: float = 0.12,
) -> Dict[str, object]:
    """
    Quasi-static drag equilibrium used by the paper for steady-current validation.

    The rope and payload forces are accumulated from the payload end upward. Segment
    directions are then updated from the cumulative tension vector, which is the
    discrete counterpart of the bottom-up equilibrium integration in the manuscript.
    """
    cfg = system.config
    n_calc = int(max(3, n_points))
    length = float(cfg["L"])
    l0 = length / float(n_calc - 1)
    E = float(cfg["E"]) * float(cfg.get("EA_factor", 1.0))
    A = float(system.A)
    rho_w = float(cfg["rho_w"])
    d = float(cfg["d"])
    c_t = float(cfg.get("C_t", 0.02))
    c_n = float(cfg.get("C_n", 1.0))
    wet_weight_per_length = (float(cfg["rho_c"]) - rho_w) * A * float(cfg["g"])
    payload_wet = _payload_wet_weight(system)

    positions = np.zeros((n_calc, 3), dtype=float)
    positions[:, 2] = -np.linspace(0.0, length, n_calc)
    segment_forces = np.zeros((n_calc - 1, 3), dtype=float)
    top_tension = np.nan

    for iteration in range(int(max_iter)):
        previous = positions.copy()

        last_vec = positions[-1] - positions[-2]
        last_norm = float(np.linalg.norm(last_vec))
        tau_bottom = last_vec / last_norm if last_norm > EPS else np.array([0.0, 0.0, -1.0])

        payload_force = np.array([0.0, 0.0, -payload_wet], dtype=float)
        payload_force += system._compute_payload_drag(
            positions[-1],
            np.zeros(3, dtype=float),
            tau_bottom,
            0.0,
        )

        cumulative_force = payload_force.copy()

        for j in range(n_calc - 2, -1, -1):
            r1 = positions[j]
            r2 = positions[j + 1]
            seg_vec = r2 - r1
            seg_norm = float(np.linalg.norm(seg_vec))
            tau = seg_vec / seg_norm if seg_norm > EPS else np.array([0.0, 0.0, -1.0])
            r_mid = 0.5 * (r1 + r2)

            c_mid, a_mid = system.fluid_kinematics(r_mid, 0.0)
            v_rel = -c_mid
            v_t_mag = float(np.dot(v_rel, tau))
            v_t_vec = v_t_mag * tau
            v_n_vec = v_rel - v_t_vec
            v_n_mag = float(np.linalg.norm(v_n_vec))

            f_drag_t = -0.5 * rho_w * c_t * (np.pi * d) * l0 * abs(v_t_mag) * v_t_vec
            f_drag_n = -0.5 * rho_w * c_n * d * l0 * v_n_mag * v_n_vec

            a_tangent = float(np.dot(a_mid, tau)) * tau
            a_normal = a_mid - a_tangent
            cm_n = 1.0 + float(cfg.get("Ca_rope_normal", 1.0))
            cm_t = 1.0 + float(cfg.get("Ca_rope_tangent", 0.05))
            f_inertia = rho_w * A * l0 * (cm_n * a_normal + cm_t * a_tangent)

            f_weight = np.array([0.0, 0.0, -wet_weight_per_length * l0], dtype=float)
            segment_force = f_drag_t + f_drag_n + f_inertia + f_weight

            segment_forces[j] = cumulative_force + 0.5 * segment_force
            cumulative_force += segment_force

        top_tension = float(np.linalg.norm(cumulative_force))

        updated = np.zeros_like(positions)
        for j in range(n_calc - 1):
            force = segment_forces[j]
            force_norm = float(np.linalg.norm(force))
            direction = force / force_norm if force_norm > EPS else np.array([0.0, 0.0, -1.0])
            strain = force_norm / max(E * A, EPS)
            updated[j + 1] = updated[j] + l0 * (1.0 + strain) * direction

        positions = (1.0 - relax) * positions + relax * updated
        if float(np.max(np.linalg.norm(positions - previous, axis=1))) < tol:
            break

    s = _arc_length_from_positions(positions)
    tension = np.linalg.norm(segment_forces, axis=1)
    tension_nodes = np.empty(n_calc, dtype=float)
    tension_nodes[:-1] = tension
    tension_nodes[-1] = float(np.linalg.norm(payload_force))

    return {
        "positions": positions,
        "s": s,
        "tension": tension_nodes,
        "top_tension": top_tension,
        "iterations": iteration + 1,
    }


def extract_dynamic_metrics(system: DynamicWireRopeSystem3D, solution) -> Dict[str, float]:
    """统一提取动态下放核心指标。"""
    times = solution.t
    states = solution.y
    n_of_t = getattr(solution, "N_of_t", np.full(times.shape, system.config["N"], dtype=int))

    final_n = int(n_of_t[-1])
    final_pos = states[: 3 * final_n, -1].reshape(final_n, 3)

    payload_ref = system.compute_payload_reference_point(final_pos)
    final_depth = -float(payload_ref[2])
    final_height_above_seabed = float(system.compute_height_above_seabed(final_pos))

    lookback = max(30.0, 3.0 * float(system.config.get("wave_period", 10.0)))
    avg_offset, avg_pch_tilt, avg_tool_tilt = system.get_convergent_offset(solution, lookback_time=lookback)

    max_stress_history, _, _, _, _ = system._compute_stress_strain_history(times, states, n_of_t)
    if len(max_stress_history) > 0:
        max_stress_mpa = float(np.max(max_stress_history) / 1e6)
        stress_95pct_mpa = float(np.percentile(max_stress_history, 95) / 1e6)
    else:
        max_stress_mpa = np.nan
        stress_95pct_mpa = np.nan

    deployment_time = float(system.target_reach_time) if system.target_reach_time is not None else float(times[-1])
    solve_time = float(getattr(solution, "total_solve_time", np.nan))

    return {
        "deployment_time_s": deployment_time,
        "horizontal_offset_m": float(avg_offset),
        "pch_tilt_deg": float(avg_pch_tilt),
        "tool_tilt_deg": float(avg_tool_tilt),
        "max_stress_mpa": max_stress_mpa,
        "stress_95pct_mpa": stress_95pct_mpa,
        "final_depth_m": final_depth,
        "final_height_above_seabed_m": final_height_above_seabed,
        "solve_time_s": solve_time,
    }


class PaperVerificationRunner:
    """论文验证收敛分析执行器。"""

    def __init__(self, output_dir: Optional[str] = None, smooth_stress: bool = False):
        self.stress_smoothing_enabled = bool(smooth_stress)
        self.output_paths = prepare_output_dirs(output_dir)
        self.mesh_scenarios = self._build_default_mesh_scenarios()
        self.static_cases = self._build_default_static_cases()
        self.base_dynamic_config = self._build_dynamic_base_config()
        self.base_dynamic_config["visualization_smooth_enabled"] = self.stress_smoothing_enabled
        self.base_static_config = self._build_static_base_config()

    @staticmethod
    def _build_default_mesh_scenarios() -> List[MeshScenario]:
        return [
            MeshScenario(
                key="case_a_slow_small_wave",
                name_en="A: Slow, small wave",
                name_cn="A: 慢速小波",
                color="#2ca02c",
                marker="o",
                release_speed=0.3,
                rope_diameter=0.025,
                winch_x=0.0,
                wave_height=1.0,
                wave_period=8.0,
                current_scale=0.70,
            ),
            MeshScenario(
                key="case_b_general",
                name_en="B: General",
                name_cn="B: 一般工况",
                color="#1f77b4",
                marker="^",
                release_speed=0.6,
                rope_diameter=0.032,
                winch_x=0.5,
                wave_height=1.8,
                wave_period=11.0,
                current_scale=1.00,
            ),
            MeshScenario(
                key="case_c_fast_large_wave",
                name_en="C: Fast, large wave",
                name_cn="C: 快速大波",
                color="#d62728",
                marker="D",
                release_speed=1.0,
                rope_diameter=0.04,
                winch_x=2.0,
                wave_height=2.5,
                wave_period=12.0,
                current_scale=1.30,
            ),
        ]

    @staticmethod
    def _build_default_static_cases() -> List[StaticCurrentCase]:
        return [
            StaticCurrentCase(
                key="no_current",
                name_en="No current",
                name_cn="No current",
                color="#4c4c4c",
                marker="s",
                current_scale=0.0,
                water_depth=600.0,
                rope_length=500.0,
                node_count=36,
            ),
            StaticCurrentCase(
                key="slow_current",
                name_en="Low current",
                name_cn="慢速海流",
                color="#2ca02c",
                marker="o",
                current_scale=0.70,
                water_depth=600.0,
                rope_length=500.0,
                node_count=36,
            ),
            StaticCurrentCase(
                key="medium_current",
                name_en="Moderate current",
                name_cn="中等海流",
                color="#1f77b4",
                marker="^",
                current_scale=1.00,
                water_depth=600.0,
                rope_length=500.0,
                node_count=36,
            ),
            StaticCurrentCase(
                key="fast_current",
                name_en="Strong current",
                name_cn="快速海流",
                color="#d62728",
                marker="D",
                current_scale=1.30,
                water_depth=600.0,
                rope_length=500.0,
                node_count=36,
            ),
        ]

    @staticmethod
    def _build_dynamic_base_config() -> Dict:
        cfg = HIGH_FIDELITY_DEFAULT_CONFIG.copy()
        cfg.update(
            {
                "verbose": False,
                "active_tension_control_enabled": False,
                "solver_profile": "engineering",
                "use_sparse_jacobian": True,
                "event_detection_enabled": True,
                "initialization_mode": "current_biased",
                "payload_motion_mode": "rope_aligned",
                "summary_safety_factor_basis": "raw",
                "total_depth": 1500.0,
                "total_segments": 25,
                "growth_initial_length": 0.05,
                "target_height_above_seabed": 25.0,
                "extra_sim_time_after_target": 0.0,
                "target_search_time_after_release": 600.0,
            }
        )
        return cfg

    @staticmethod
    def _build_static_base_config() -> Dict:
        cfg = HIGH_FIDELITY_DEFAULT_CONFIG.copy()
        cfg.update(
            {
                "verbose": False,
                "wave_amplitude": 0.0,
                "heave_RAO": 0.0,
                "roll_RAO": 0.0,
                "pitch_RAO": 0.0,
                "surge_RAO": 0.0,
                "sway_RAO": 0.0,
                "yaw_RAO": 0.0,
                "vessel_speed": 0.0,
                "active_tension_control_enabled": False,
                "solver_profile": "engineering",
                "use_sparse_jacobian": True,
                "initialization_mode": "vertical",
                "payload_motion_mode": "vertical_hanging",
                "xi_axial": 0.25,
                "xi_transverse": 0.05,
            }
        )
        return cfg

    @staticmethod
    def _rope_wet_weight_per_length(config: Dict) -> float:
        area = np.pi * (float(config["d"]) / 2.0) ** 2
        return (float(config["rho_c"]) - float(config["rho_w"])) * area * float(config["g"])

    def _build_dynamic_case_config(self, scenario: MeshScenario, n_segments: int) -> Dict:
        cfg = copy.deepcopy(self.base_dynamic_config)
        cfg["total_segments"] = int(n_segments)
        cfg["release_speed"] = float(scenario.release_speed)
        cfg["d"] = float(scenario.rope_diameter)
        cfg["wave_height"] = float(scenario.wave_height)
        cfg["wave_amplitude"] = 0.5 * float(scenario.wave_height)
        cfg["wave_period"] = float(scenario.wave_period)
        cfg["current_scale"] = float(scenario.current_scale)
        winch_offset = list(cfg.get("winch_offset", [0.0, 0.0, 3.10]))
        while len(winch_offset) < 3:
            winch_offset.append(0.0)
        winch_offset[0] = float(scenario.winch_x)
        cfg["winch_offset"] = winch_offset
        return cfg

    @staticmethod
    def _estimate_dynamic_t_end(config: Dict) -> float:
        release_speed = max(float(config.get("release_speed", 1.0)), 1e-6)
        total_depth = float(config.get("total_depth", 1500.0))
        total_segments = max(int(config.get("total_segments", 1)), 1)
        growth_initial_length = float(config.get("growth_initial_length", 0.0))
        segment_length = (
            total_depth + max(total_segments - 1, 0) * growth_initial_length
        ) / total_segments
        release_completion_time = (
            max(total_segments - 1, 0) * segment_length / release_speed
        )
        extra = float(config.get("extra_sim_time_after_target", 0.0))
        target_search_time = float(config.get("target_search_time_after_release", 600.0))
        nominal_deployment_time = max(total_depth / release_speed, release_completion_time)
        return nominal_deployment_time + max(20.0, extra, target_search_time)

    def _run_single_dynamic_case(
        self,
        scenario: MeshScenario,
        n_segments: int,
        dt_output: float,
    ) -> Dict[str, float]:
        cfg = self._build_dynamic_case_config(scenario, n_segments)

        wall_start = time.time()
        system = ScaledCurrentDynamicSystem(cfg)
        t_end = self._estimate_dynamic_t_end(cfg)
        solution = system.simulate_dynamic_release(t_end=t_end, dt_output=dt_output)
        metrics = extract_dynamic_metrics(system, solution)
        metrics["wall_clock_s"] = float(time.time() - wall_start)
        metrics["target_reached"] = bool(system.target_reach_time is not None)
        if not metrics["target_reached"]:
            raise RuntimeError(
                "Target height was not reached; convergence result would be invalid. "
                f"scenario={scenario.name_en}, N={n_segments}, "
                f"final_height_above_seabed={metrics['final_height_above_seabed_m']:.3f} m, "
                f"t_end={t_end:.3f} s"
            )
        return metrics

    def run_mesh_convergence(
        self,
        n_values: Optional[Sequence[int]] = None,
        reference_n: Optional[int] = 60,
        reference_extra: Optional[int] = None,
        error_threshold: float = 0.05,
        dt_output: float = 1.0,
        stress_metric: str = "p95",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """执行多工况网格收敛分析。"""
        if n_values is None:
            n_values = [10, 20, 30, 40, 50, 60, 70, 80]

        n_values = [int(n) for n in n_values]
        stress_metric = str(stress_metric).lower()
        if stress_metric not in {"p95", "max"}:
            raise ValueError("stress_metric must be 'p95' or 'max'")

        stress_col = "stress_95pct_mpa" if stress_metric == "p95" else "max_stress_mpa"
        stress_label = "sigma95" if stress_metric == "p95" else "sigma_max_raw"
        stress_postprocess = "smoothed" if self.stress_smoothing_enabled else "raw"
        records: List[Dict] = []
        reference_rows: List[Dict] = []
        summary_rows: List[Dict] = []

        print("\n" + "=" * 80)
        print("多工况网格收敛性分析")
        print("=" * 80)
        print(f"Stress postprocess: {stress_postprocess}")

        for scenario in self.mesh_scenarios:
            print(f"\n[工况] {scenario.name_cn}")
            n_ref = (
                int(reference_n)
                if reference_n is not None
                else max(n_values) + int(reference_extra if reference_extra is not None else 4)
            )
            print(f"  参考网格: N_ref = {n_ref}")

            reference = self._run_single_dynamic_case(scenario, n_ref, dt_output=dt_output)
            reference["scenario_key"] = scenario.key
            reference["scenario_name"] = scenario.name_en
            reference["scenario_name_cn"] = scenario.name_cn
            reference["N_ref"] = n_ref
            reference["release_speed_mps"] = scenario.release_speed
            reference["rope_diameter_m"] = scenario.rope_diameter
            reference["winch_x_m"] = scenario.winch_x
            reference["wave_height_m"] = scenario.wave_height
            reference["wave_period_s"] = scenario.wave_period
            reference["stress_metric"] = stress_label
            reference["stress_postprocess"] = stress_postprocess
            reference["stress_smoothing_enabled"] = self.stress_smoothing_enabled
            reference["stress_metric_mpa"] = reference[stress_col]
            reference_rows.append(reference)

            scenario_records: List[Dict] = []
            for idx, n_seg in enumerate(n_values, start=1):
                print(f"  [{idx}/{len(n_values)}] N = {n_seg} ...", end=" ")
                one = self._run_single_dynamic_case(scenario, n_seg, dt_output=dt_output)

                row = {
                    "scenario_key": scenario.key,
                    "scenario_name": scenario.name_en,
                    "scenario_name_cn": scenario.name_cn,
                    "N": int(n_seg),
                    "release_speed_mps": scenario.release_speed,
                    "rope_diameter_m": scenario.rope_diameter,
                    "winch_x_m": scenario.winch_x,
                    "wave_height_m": scenario.wave_height,
                    "wave_period_s": scenario.wave_period,
                    "deployment_time_s": one["deployment_time_s"],
                    "horizontal_offset_m": one["horizontal_offset_m"],
                    "max_stress_mpa": one["max_stress_mpa"],
                    "stress_95pct_mpa": one["stress_95pct_mpa"],
                    "stress_metric": stress_label,
                    "stress_postprocess": stress_postprocess,
                    "stress_smoothing_enabled": self.stress_smoothing_enabled,
                    "stress_metric_mpa": one[stress_col],
                    "pch_tilt_deg": one["pch_tilt_deg"],
                    "tool_tilt_deg": one["tool_tilt_deg"],
                    "final_depth_m": one["final_depth_m"],
                    "final_height_above_seabed_m": one["final_height_above_seabed_m"],
                    "solve_time_s": one["solve_time_s"],
                    "wall_clock_s": one["wall_clock_s"],
                    "target_reached": one["target_reached"],
                    "reference_deployment_time_s": reference["deployment_time_s"],
                    "reference_horizontal_offset_m": reference["horizontal_offset_m"],
                    "reference_max_stress_mpa": reference["max_stress_mpa"],
                    "reference_stress_95pct_mpa": reference["stress_95pct_mpa"],
                    "reference_stress_metric_mpa": reference[stress_col],
                }

                row["error_deployment_time"] = compute_relative_error(
                    row["deployment_time_s"],
                    row["reference_deployment_time_s"],
                )
                row["error_horizontal_offset"] = compute_relative_error(
                    row["horizontal_offset_m"],
                    row["reference_horizontal_offset_m"],
                )
                row["error_stress_metric"] = compute_relative_error(
                    row["stress_metric_mpa"],
                    row["reference_stress_metric_mpa"],
                )
                row["error_max_stress"] = row["error_stress_metric"]

                scenario_records.append(row)
                print(
                    "完成 | "
                    f"时间误差={row['error_deployment_time'] * 100:.2f}%, "
                    f"偏移误差={row['error_horizontal_offset'] * 100:.2f}%, "
                    f"应力误差({stress_label})={row['error_stress_metric'] * 100:.2f}%"
                )

            records.extend(scenario_records)
            sdf = pd.DataFrame(scenario_records).sort_values("N")

            valid = sdf[
                (sdf["error_deployment_time"] <= error_threshold)
                & (sdf["error_horizontal_offset"] <= error_threshold)
                & (sdf["error_stress_metric"] <= error_threshold)
            ]
            recommended_n = int(valid["N"].iloc[0]) if not valid.empty else int(sdf["N"].iloc[-1])

            p_time, r2_time = estimate_convergence_order(sdf["N"], sdf["error_deployment_time"])
            p_offset, r2_offset = estimate_convergence_order(sdf["N"], sdf["error_horizontal_offset"])
            p_stress, r2_stress = estimate_convergence_order(sdf["N"], sdf["error_stress_metric"])

            summary_rows.append(
                {
                    "scenario_key": scenario.key,
                    "scenario_name": scenario.name_en,
                    "scenario_name_cn": scenario.name_cn,
                    "N_ref": n_ref,
                    "stress_metric": stress_label,
                    "stress_postprocess": stress_postprocess,
                    "stress_smoothing_enabled": self.stress_smoothing_enabled,
                    "recommended_N": recommended_n,
                    "meet_threshold": bool(not valid.empty),
                    "convergence_order_time": p_time,
                    "convergence_order_time_r2": r2_time,
                    "convergence_order_offset": p_offset,
                    "convergence_order_offset_r2": r2_offset,
                    "convergence_order_stress": p_stress,
                    "convergence_order_stress_r2": r2_stress,
                }
            )

        df = pd.DataFrame(records).sort_values(["scenario_name", "N"]).reset_index(drop=True)
        df_ref = pd.DataFrame(reference_rows)
        df_summary = pd.DataFrame(summary_rows)

        mesh_csv = self.output_paths["data"] / "mesh_convergence_detail.csv"
        ref_csv = self.output_paths["data"] / "mesh_convergence_reference.csv"
        summary_csv = self.output_paths["data"] / "mesh_convergence_summary.csv"
        order_summary_csv = self.output_paths["data"] / "mesh_convergence_order_summary.csv"
        df.to_excel(str(mesh_csv).replace(".csv", ".xlsx"), index=False)
        df_ref.to_excel(str(ref_csv).replace(".csv", ".xlsx"), index=False)
        df_summary.to_excel(str(summary_csv).replace(".csv", ".xlsx"), index=False)

        order_summary_df = df_summary[
            [
                "scenario_key",
                "scenario_name",
                "stress_metric",
                "stress_postprocess",
                "recommended_N",
                "meet_threshold",
                "convergence_order_time",
                "convergence_order_time_r2",
                "convergence_order_offset",
                "convergence_order_offset_r2",
                "convergence_order_stress",
                "convergence_order_stress_r2",
            ]
        ].rename(
            columns={
                "convergence_order_time": "order_time",
                "convergence_order_time_r2": "order_time_r2",
                "convergence_order_offset": "order_offset",
                "convergence_order_offset_r2": "order_offset_r2",
                "convergence_order_stress": "order_stress",
                "convergence_order_stress_r2": "order_stress_r2",
            }
        )
        order_summary_df.to_excel(str(order_summary_csv).replace(".csv", ".xlsx"), index=False)

        print("\nConvergence-order summary by scenario:")
        for _, row in order_summary_df.iterrows():
            print(
                "  "
                f"{row['scenario_name']}: "
                f"p_time={row['order_time']:.3f} (R2={row['order_time_r2']:.3f}), "
                f"p_offset={row['order_offset']:.3f} (R2={row['order_offset_r2']:.3f}), "
                f"p_stress[{row['stress_metric']}, {row['stress_postprocess']}]={row['order_stress']:.3f} (R2={row['order_stress_r2']:.3f}), "
                f"recommended_N={int(row['recommended_N'])}"
            )

        self.plot_mesh_convergence(
            df,
            n_values=n_values,
            error_threshold=error_threshold,
            stress_metric=stress_metric,
            save_path=self.output_paths["figures"] / "mesh_convergence_comparison_3x2.png",
        )
        self.plot_mesh_solver_cost(
            df,
            save_path=self.output_paths["figures"] / "mesh_solver_time_bar.png",
        )

        print("\n网格收敛分析完成。")
        print(f"  数据: {mesh_csv}")
        print(f"  参考: {ref_csv}")
        print(f"  汇总: {summary_csv}")
        print(f"  收敛阶汇总: {order_summary_csv}")
        return df, df_ref, df_summary

    def plot_mesh_convergence(
        self,
        df: pd.DataFrame,
        n_values: Sequence[int],
        error_threshold: float,
        stress_metric: str = "p95",
        save_path: Optional[Path] = None,
    ):
        """绘制 3x2 多工况网格收敛对比图。"""
        fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex="col")
        stress_metric = str(stress_metric).lower()
        stress_label = "Stress sigma95 (MPa)" if stress_metric == "p95" else "Raw max stress (MPa)"
        if "stress_postprocess" in df.columns and (df["stress_postprocess"] == "smoothed").any():
            stress_label = stress_label.replace(" (MPa)", " smoothed (MPa)")

        panel_defs = [
            ("deployment_time_s", "error_deployment_time", "Deployment time (s)", "(a)", "(b)"),
            ("horizontal_offset_m", "error_horizontal_offset", "Horizontal offset (m)", "(c)", "(d)"),
            ("stress_metric_mpa", "error_stress_metric", stress_label, "(e)", "(f)"),
        ]

        for row_idx, (metric, metric_err, y_label_left, left_tag, right_tag) in enumerate(panel_defs):
            ax_l = axes[row_idx, 0]
            ax_r = axes[row_idx, 1]

            for scenario in self.mesh_scenarios:
                sdf = df[df["scenario_key"] == scenario.key].sort_values("N")
                if sdf.empty:
                    continue

                ax_l.plot(
                    sdf["N"],
                    sdf[metric],
                    marker=scenario.marker,
                    color=scenario.color,
                    linewidth=1.5,
                    markersize=6,
                    label=scenario.name_en,
                )
                ax_r.plot(
                    sdf["N"],
                    sdf[metric_err] * 100.0,
                    marker=scenario.marker,
                    color=scenario.color,
                    linewidth=1.5,
                    markersize=6,
                    label=scenario.name_en,
                )

            ax_l.set_ylabel(y_label_left)
            ax_r.set_ylabel("Relative error (%)")
            ax_l.grid(True, alpha=0.3)
            ax_r.grid(True, alpha=0.3)
            ax_r.axhline(y=error_threshold * 100.0, color="gray", linestyle="--", linewidth=1.0)

            ax_l.text(0.06, 0.90, left_tag, transform=ax_l.transAxes, fontweight="bold")
            ax_r.text(0.06, 0.90, right_tag, transform=ax_r.transAxes, fontweight="bold")

        axes[2, 0].set_xlabel("Number of rope segments N")
        axes[2, 1].set_xlabel("Number of rope segments N")
        axes[0, 0].legend(loc="best", frameon=False)
        axes[0, 1].legend(loc="best", frameon=False)

        for ax in axes.ravel():
            ax.set_xticks(list(n_values))

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved: {save_path}")
        plt.close(fig)

    def plot_mesh_solver_cost(self, df: pd.DataFrame, save_path: Optional[Path] = None):
        """绘制分组柱状图：不同工况下的求解时间对比。"""
        fig, ax = plt.subplots(figsize=(8, 5))

        pivot = df.pivot_table(index="N", columns="scenario_name", values="solve_time_s", aggfunc="mean")
        n_values = sorted(pivot.index.tolist())
        x = np.arange(len(n_values), dtype=float)

        n_scen = len(self.mesh_scenarios)
        bar_width = 0.8 / max(1, n_scen)

        for i, scenario in enumerate(self.mesh_scenarios):
            values = []
            for n_val in n_values:
                if scenario.name_en in pivot.columns:
                    values.append(float(pivot.loc[n_val, scenario.name_en]))
                else:
                    values.append(np.nan)

            offset = (i - (n_scen - 1) / 2.0) * bar_width
            ax.bar(
                x + offset,
                values,
                width=bar_width,
                color=scenario.color,
                alpha=0.9,
                label=scenario.name_en,
            )

        ax.set_xlabel("Number of rope segments N")
        ax.set_ylabel("Solver time (s)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in n_values])
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper left", frameon=False)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved: {save_path}")
        plt.close(fig)

    def _build_static_case_config(self, case: StaticCurrentCase) -> Dict:
        cfg = copy.deepcopy(self.base_static_config)
        cfg["N"] = int(case.node_count)
        cfg["L"] = float(case.rope_length)
        cfg["total_water_depth"] = float(case.water_depth)
        cfg["seabed_depth"] = -float(case.water_depth)
        cfg["current_scale"] = float(case.current_scale)
        return cfg

    def _run_single_static_case(
        self,
        case: StaticCurrentCase,
        relax_time: float,
        n_eval: int,
    ) -> Tuple[Dict, Dict]:
        cfg = self._build_static_case_config(case)
        system = ScaledCurrentStaticSystem(cfg)

        t_eval = np.linspace(0.0, float(relax_time), int(max(2, n_eval)))
        solution = system.solve(t_span=(0.0, float(relax_time)), t_eval=t_eval)
        if not solution.success:
            raise RuntimeError(f"Static solve failed: {case.name_en}")

        positions = solution.y[: 3 * cfg["N"], -1].reshape(cfg["N"], 3)
        x_num, z_num, s_num = project_curve_to_plane(positions)

        if abs(float(case.current_scale)) < EPS:
            reference = solve_no_current_catenary_reference(system, n_points=400)
            tension_reference = solve_no_current_catenary_reference(system, n_points=cfg["N"])
        else:
            reference = solve_quasi_static_drag_reference(system, n_points=400)
            tension_reference = solve_quasi_static_drag_reference(system, n_points=cfg["N"])
        reference_positions = np.asarray(reference["positions"], dtype=float)
        x_ana, z_ana, s_ana = project_curve_to_plane(reference_positions)
        t_ana = np.asarray(reference["tension"], dtype=float)
        t_ana_for_numeric_grid = np.asarray(tension_reference["tension"], dtype=float)

        x_num_interp, z_num_interp = interpolate_curve_by_arc(s_num, x_num, z_num, s_ana)
        shape_error = np.sqrt((x_num_interp - x_ana) ** 2 + (z_num_interp - z_ana) ** 2)
        rmse_shape = float(np.sqrt(np.mean(shape_error ** 2))) if len(shape_error) > 0 else np.nan
        max_shape_error = float(np.max(shape_error)) if len(shape_error) > 0 else np.nan

        l, l_unit, l_norm = system._compute_segment_vectors(positions)
        tension_num = system._compute_tension(l, l_unit, l_norm, float(solution.t[-1]))
        top_tension_num = float(np.linalg.norm(tension_num[0])) if tension_num.size > 0 else np.nan
        top_tension_ana = (
            float(t_ana_for_numeric_grid[0]) if len(t_ana_for_numeric_grid) > 0 else np.nan
        )

        top_tension_error = (
            compute_relative_error(top_tension_num, top_tension_ana)
            if np.isfinite(top_tension_num) and np.isfinite(top_tension_ana)
            else np.nan
        )

        reference_method = (
            "analytic_catenary"
            if abs(float(case.current_scale)) < EPS
            else "bottom_up_quasi_static_drag"
        )

        result_row = {
            "case_key": case.key,
            "case_name": case.name_en,
            "case_name_cn": case.name_cn,
            "reference_method": reference_method,
            "current_scale": case.current_scale,
            "water_depth_m": case.water_depth,
            "rope_length_m": case.rope_length,
            "node_count": case.node_count,
            "reference_iterations": int(reference["iterations"]),
            "solve_time_s": float(getattr(solution, "solve_time", np.nan)),
            "bottom_offset_m": float(x_num[-1]),
            "bottom_depth_m": float(z_num[-1]),
            "shape_rmse_m": rmse_shape,
            "shape_max_error_m": max_shape_error,
            "top_tension_num_kN": top_tension_num / 1000.0 if np.isfinite(top_tension_num) else np.nan,
            "top_tension_ana_kN": top_tension_ana / 1000.0 if np.isfinite(top_tension_ana) else np.nan,
            "top_tension_error_pct": top_tension_error * 100.0 if np.isfinite(top_tension_error) else np.nan,
        }

        curve_data = {
            "name": case.name_en,
            "color": case.color,
            "marker": case.marker,
            "x_num": x_num,
            "z_num": z_num,
            "s_num": s_num,
            "x_ana": x_ana,
            "z_ana": z_ana,
            "s_ana": s_ana,
            "t_ana": t_ana,
            "reference_method": reference_method,
        }
        return result_row, curve_data

    def run_static_catenary_comparison(
        self,
        relax_time: float = 240.0,
        n_eval: int = 280,
    ) -> pd.DataFrame:
        """执行静态悬链线对比。"""
        print("\n" + "=" * 80)
        print("Static Equilibrium Comparison (Reference vs Numerical)")
        print("=" * 80)

        rows: List[Dict] = []
        curves: List[Dict] = []

        for case in self.static_cases:
            print(f"\n[Static Case] {case.name_en} (current_scale={case.current_scale:.2f})")
            row, curve = self._run_single_static_case(case, relax_time=relax_time, n_eval=n_eval)
            rows.append(row)
            
            # Save node coordinates to Excel
            # Pad arrays to the same length with NaNs to put in single DataFrame
            max_len = max(len(curve["x_num"]), len(curve["x_ana"]))
            def pad(arr, length):
                return np.pad(arr, (0, length - len(arr)), constant_values=np.nan)
            
            curve_df = pd.DataFrame({
                "x_num_m": pad(curve["x_num"], max_len), 
                "z_num_m": pad(curve["z_num"], max_len), 
                "s_num_m": pad(curve["s_num"], max_len),
                "x_ana_m": pad(curve["x_ana"], max_len), 
                "z_ana_m": pad(curve["z_ana"], max_len), 
                "s_ana_m": pad(curve["s_ana"], max_len),
                "tension_ana_kN": pad(curve["t_ana"] / 1000.0, max_len)
            })
            curve_out_xlsx = self.output_paths["data"] / f"static_catenary_nodes_{case.key}.xlsx"
            try:
                # try to remove existing file to avoid permission issues when possible
                if curve_out_xlsx.exists():
                    try:
                        curve_out_xlsx.unlink()
                    except PermissionError:
                        # fall back to unique filename
                        import time

                        curve_out_xlsx = self.output_paths["data"] / (
                            f"static_catenary_nodes_{case.key}_{int(time.time())}.xlsx"
                        )
                curve_df.to_excel(curve_out_xlsx, index=False)
            except PermissionError:
                import time

                curve_out_xlsx = self.output_paths["data"] / (
                    f"static_catenary_nodes_{case.key}_{int(time.time())}.xlsx"
                )
                curve_df.to_excel(curve_out_xlsx, index=False)
            curves.append(curve)
            print(
                "  "
                f"RMSE={row['shape_rmse_m']:.3f}m, "
                f"MaxErr={row['shape_max_error_m']:.3f}m, "
                f"TopTensionErr={row['top_tension_error_pct']:.2f}%"
            )

        df = pd.DataFrame(rows)
        out_xlsx = self.output_paths["data"] / "static_catenary_comparison.csv".replace(".csv", ".xlsx")
        try:
            if out_xlsx.exists():
                try:
                    out_xlsx.unlink()
                except PermissionError:
                    import time

                    out_xlsx = self.output_paths["data"] / (f"static_catenary_comparison_{int(time.time())}.xlsx")
            df.to_excel(out_xlsx, index=False)
        except PermissionError:
            import time

            out_xlsx = self.output_paths["data"] / (f"static_catenary_comparison_{int(time.time())}.xlsx")
            df.to_excel(out_xlsx, index=False)

        self.plot_static_catenary_comparison(
            curves,
            save_path=self.output_paths["figures"] / "static_catenary_comparison.png",
        )

        print("\nStatic equilibrium comparison completed.")
        print(f"  Data: {out_xlsx}")
        return df

    @staticmethod
    def plot_static_catenary_comparison(curves: List[Dict], save_path: Optional[Path] = None):
        """绘制静态形状对比图：x-深度。"""
        fig, ax = plt.subplots(figsize=(8, 5.5))

        all_z = []
        for curve in curves:
            name = curve["name"]
            color = curve["color"]
            marker = curve["marker"]

            x_ana = np.asarray(curve["x_ana"], dtype=float)
            z_ana = np.asarray(curve["z_ana"], dtype=float)
            x_num = np.asarray(curve["x_num"], dtype=float)
            z_num = np.asarray(curve["z_num"], dtype=float)

            markevery = max(1, len(x_num) // 14)
            ax.plot(x_ana, z_ana, "-", color=color, linewidth=1.8, label=f"{name} (reference)")
            ax.plot(
                x_num,
                z_num,
                "--",
                color=color,
                linewidth=1.2,
                marker=marker,
                markersize=4,
                markevery=markevery,
                label=f"{name} (numerical)",
            )

            all_z.extend(z_ana.tolist())
            all_z.extend(z_num.tolist())

        ax.set_xlabel("Horizontal offset (m)")
        ax.set_ylabel("Depth (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=False, ncol=2)

        if all_z:
            min_z = min(all_z)
            ax.set_ylim(min_z * 1.02, 5.0)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved: {save_path}")
        plt.close(fig)

    def run_all(
        self,
        mesh_n_values: Optional[Sequence[int]] = None,
        reference_n: Optional[int] = 40,
        reference_extra: Optional[int] = None,
        error_threshold: float = 0.05,
        dt_output: float = 1.0,
        stress_metric: str = "p95",
        static_relax_time: float = 900.0,
        static_n_eval: int = 280,
    ) -> Dict[str, pd.DataFrame]:
        """一键执行“网格收敛 + 静态悬链线对比”。"""
        mesh_df, mesh_ref_df, mesh_summary_df = self.run_mesh_convergence(
            n_values=mesh_n_values,
            reference_n=reference_n,
            reference_extra=reference_extra,
            error_threshold=error_threshold,
            dt_output=dt_output,
            stress_metric=stress_metric,
        )
        static_df = self.run_static_catenary_comparison(
            relax_time=static_relax_time,
            n_eval=static_n_eval,
        )

        overview_rows = []
        for _, row in mesh_summary_df.iterrows():
            overview_rows.append(
                {
                    "section": "mesh_convergence",
                    "scenario": row["scenario_name"],
                    "metric": "recommended_N",
                    "value": row["recommended_N"],
                }
            )
            overview_rows.append(
                {
                    "section": "mesh_convergence",
                    "scenario": row["scenario_name"],
                    "metric": f"order_stress_{row['stress_metric']}",
                    "value": row["convergence_order_stress"],
                }
            )

        for _, row in static_df.iterrows():
            overview_rows.append(
                {
                    "section": "static_catenary",
                    "scenario": row["case_name"],
                    "metric": "shape_rmse_m",
                    "value": row["shape_rmse_m"],
                }
            )
            overview_rows.append(
                {
                    "section": "static_catenary",
                    "scenario": row["case_name"],
                    "metric": "top_tension_error_pct",
                    "value": row["top_tension_error_pct"],
                }
            )

        overview_df = pd.DataFrame(overview_rows)
        overview_csv = self.output_paths["data"] / "verification_overview.csv"
        overview_df.to_excel(str(overview_csv).replace(".csv", ".xlsx"), index=False)

        print("\n" + "=" * 80)
        print("论文验证流程执行完成")
        print("=" * 80)
        print(f"输出根目录: {self.output_paths['root']}")
        print(f"总览文件: {overview_csv}")

        return {
            "mesh_detail": mesh_df,
            "mesh_reference": mesh_ref_df,
            "mesh_summary": mesh_summary_df,
            "static_catenary": static_df,
            "overview": overview_df,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="论文验证导向收敛性分析")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "mesh", "catenary"],
        help="执行模式：all(默认) | mesh | catenary",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--quick", action="store_true", help="快速烟测模式")
    parser.add_argument("--dt-output", type=float, default=1.0, help="动态输出步长(s)")
    parser.add_argument("--reference-n", type=int, default=40, help="动态收敛性分析的参考段数 N_ref")
    parser.add_argument("--reference-extra", type=int, default=None, help="兼容旧参数：N_ref = max(N) + reference_extra")
    parser.add_argument(
        "--stress-metric",
        type=str,
        default="p95",
        choices=["p95", "max"],
        help="动态收敛应力指标：p95=95%%分位应力(默认)，max=原始最大应力",
    )
    parser.add_argument(
        "--smooth-stress",
        action="store_true",
        help="使用 high_fidelity_sim 的平滑应力历史计算收敛指标",
    )
    parser.add_argument("--error-threshold", type=float, default=0.05, help="收敛误差阈值（默认5%%）")
    parser.add_argument("--relax-time", type=float, default=900.0, help="静态松弛仿真时长(s)")
    parser.add_argument("--n-eval", type=int, default=280, help="静态求解采样点数")
    return parser.parse_args()


def main():
    args = parse_args()

    runner = PaperVerificationRunner(output_dir=args.output_dir, smooth_stress=args.smooth_stress)

    if args.quick:
        mesh_n_values = [5, 10, 15, 20]
        static_relax_time = min(args.relax_time, 120.0)
        static_n_eval = min(args.n_eval, 120)
    else:
        mesh_n_values = [10, 20, 30, 40, 50, 60, 70, 80]
        static_relax_time = args.relax_time
        static_n_eval = args.n_eval

    if args.mode == "mesh":
        runner.run_mesh_convergence(
            n_values=mesh_n_values,
            reference_n=args.reference_n,
            reference_extra=args.reference_extra,
            error_threshold=args.error_threshold,
            dt_output=args.dt_output,
            stress_metric=args.stress_metric,
        )
    elif args.mode == "catenary":
        runner.run_static_catenary_comparison(
            relax_time=static_relax_time,
            n_eval=static_n_eval,
        )
    else:
        runner.run_all(
            mesh_n_values=mesh_n_values,
            reference_n=args.reference_n,
            reference_extra=args.reference_extra,
            error_threshold=args.error_threshold,
            dt_output=args.dt_output,
            stress_metric=args.stress_metric,
            static_relax_time=static_relax_time,
            static_n_eval=static_n_eval,
        )


if __name__ == "__main__":
    main()
