# -*- coding: utf-8 -*-
"""
敏感性分析模块 - LHS采样 + 高保真仿真 + SRC/SRRC敏感性指标

基于 methodology.md 设计：
- 拉丁超立方采样（LHS）
- 批量调用高保真仿真
- 标准化回归系数（SRC）和秩相关系数（SRRC）计算
- 结果可直接作为后续 Kriging 代理模型的初始训练集
& C:\\Users\\syc\\anaconda3\\envs\\deploy\\python.exe sensitivity_convergence\\sensitivity_analysis.py --fig20-only
作者：自动生成
日期：2025-12-11
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from scipy.stats.qmc import LatinHypercube
import time
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入高保真仿真模块
from high_fidelity_sim import (
    DEFAULT_CONFIG as HIGH_FIDELITY_DEFAULT_CONFIG,
    DynamicWireRopeSystem3D,
)
from high_fidelity_1_opt import setup_publication_style

DEFAULT_OUTPUT_ROOT = MODULE_DIR / 'results' / 'sensitivity_analysis'


def prepare_output_dirs(output_dir: Optional[str] = None) -> Dict[str, str]:
    """Create the default organized output tree for sensitivity analysis."""
    root = Path(output_dir) if output_dir else DEFAULT_OUTPUT_ROOT
    data_dir = root / 'data'
    figures_dir = root / 'figures'
    checkpoints_dir = data_dir / 'checkpoints'

    for path in (root, data_dir, figures_dir, checkpoints_dir):
        path.mkdir(parents=True, exist_ok=True)

    return {
        'root': str(root),
        'data': str(data_dir),
        'figures': str(figures_dir),
        'checkpoints': str(checkpoints_dir),
    }

# ==============================================================================
# 参数配置
# ==============================================================================

# 敏感性分析的输入参数及其范围（7个因子）
PARAM_BOUNDS = {
    'release_speed': (0.1, 1.5),       # 下放速度 (m/s)
    'd': (0.015, 0.05),                  # 钢丝绳直径 (m)
    'm_b': (300, 1500),                 # 重物质量 (kg)
    'wave_amplitude': (0.2, 4.0),       # 波浪幅值 (m)
    'wave_period': (6.0, 16.0),         # 波浪周期 (s)
    'wave_direction': (-90.0, 90.0),    # 波浪入射方向 (deg)，相对船首
    'winch_offset_x': (-5.0, 5.0),      # 放缆点X偏移 (m)
    'winch_offset_y': (-5.0, 5.0),      # 放缆点Y偏移 (m)
}

# 圆周变量列表（需要用sin/cos变换处理）
CIRCULAR_PARAMS = ['wave_direction']

# 参数名称（用于显示）
PARAM_NAMES = list(PARAM_BOUNDS.keys())
PARAM_LABELS = {
    'release_speed': r'$v_{rel}$',
    'd': r'$d$',
    'm_b': r'$m_b$',
    'wave_amplitude': r'$A$',
    'wave_period': r'$T_w$',
    'wave_direction': r'$\theta_w$',
    'winch_offset_x': r'$x_f$',
    'winch_offset_y': r'$y_f$',
}

# 输出指标名称
OUTPUT_NAMES = ['T', 'R_offset', 'sigma_max']
OUTPUT_LABELS = {
    'T': r'Deployment Time $T$ (s)',
    'R_offset': r'Horizontal Offset $R_{offset}$ (m)',
    'sigma_max': r'Max Stress $\sigma_{max}$ (MPa)',
}


# ==============================================================================
# LHS 采样模块
# ==============================================================================

def generate_lhs_samples(n_samples: int, param_bounds: Dict[str, Tuple[float, float]], 
                          seed: int = 42) -> np.ndarray:
    """
    生成拉丁超立方采样样本
    
    参数:
        n_samples: 样本数量（建议 10~15 × 因子数）
        param_bounds: 参数边界字典 {参数名: (min, max)}
        seed: 随机种子
    
    返回:
        samples: (n_samples, n_params) 的样本矩阵
    """
    n_params = len(param_bounds)
    param_names = list(param_bounds.keys())
    
    # 使用 scipy 的 LHS 采样器
    sampler = LatinHypercube(d=n_params, seed=seed)
    X_unit = sampler.random(n=n_samples)  # [0, 1] 范围内的样本
    
    # 按参数范围缩放
    samples = np.zeros((n_samples, n_params))
    for i, name in enumerate(param_names):
        low, high = param_bounds[name]
        samples[:, i] = low + (high - low) * X_unit[:, i]
    
    return samples


def samples_to_dataframe(samples: np.ndarray, param_names: List[str]) -> pd.DataFrame:
    """将样本矩阵转换为 DataFrame"""
    return pd.DataFrame(samples, columns=param_names)


# ==============================================================================
# 高保真仿真批量运行模块
# ==============================================================================

def create_config_from_sample(sample: np.ndarray, param_names: List[str], 
                               base_config: Optional[Dict] = None) -> Dict:
    """
    从单个样本创建仿真配置
    
    参数:
        sample: 单个样本向量
        param_names: 参数名称列表
        base_config: 基础配置（可选）
    
    返回:
        config: 完整的仿真配置字典
    """
    # 默认基础配置改为复用工程主版本 `high_fidelity_sim.py`，并在此覆盖敏感性分析固定项
    config = HIGH_FIDELITY_DEFAULT_CONFIG.copy()
    config.update({
        'N': 2,                  # 初始2个节点（1段）
        'L': 100,                # 初始钢丝绳长度 100m
        'total_depth': 1500,
        'total_water_depth': 1500,
        'seabed_depth': -1500,
        'target_height_above_seabed': 25.0,
        'segment_length': 100.0,
        'total_segments': 15,    # 15段 × 100m = 1500m
        'winch_offset': [0.0, 0.0, 15.0],
        'active_tension_control_enabled': False,
        'extra_sim_time_after_target': 50.0,
        # 采样参数会在后续覆盖，这里仅保留一个显式默认值
        'wave_amplitude': 1.5,
        'wave_period': 8.0,
    })
    
    if base_config is not None:
        config.update(base_config)
    
    # 用样本值覆盖对应参数
    winch_offset_z = config.get('winch_offset', [0.0, 0.0, 15.0])[2]  # 保留原z值
    winch_x = 0.0
    winch_y = 0.0
    
    for i, name in enumerate(param_names):
        if name == 'winch_offset_x':
            winch_x = sample[i]
        elif name == 'winch_offset_y':
            winch_y = sample[i]
        else:
            config[name] = sample[i]
    
    # 组合 winch_offset（如果有水平分量参数）
    if 'winch_offset_x' in param_names or 'winch_offset_y' in param_names:
        config['winch_offset'] = [winch_x, winch_y, winch_offset_z]
    
    return config


def run_single_simulation(config: Dict, t_end: float = None, 
                           dt_output: float = 2.0, verbose: bool = False) -> Dict:
    """
    运行单次高保真仿真
    
    参数:
        config: 仿真配置
        t_end: 仿真结束时间 (s)，如果为 None 则根据下放速度自动计算
        dt_output: 输出时间步长 (s)
        verbose: 是否打印详细信息
    
    返回:
        result: 包含关键输出指标的字典
    """
    # 动态计算仿真时间：确保有足够时间到达目标深度 + 额外50s稳定期
    if t_end is None:
        segment_length = config.get('segment_length', 100.0)
        total_segments = config.get('total_segments', 15)
        release_speed = config.get('release_speed', 2.0)
        extra_time = config.get('extra_sim_time_after_target', 50.0)
        # 计算完全下放所需时间 + 额外稳定时间 + 安全余量
        t_end = (segment_length / release_speed) * total_segments + extra_time + 100.0
    try:
        # 抑制仿真过程中的打印信息
        if not verbose:
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
        
        # 创建仿真系统
        system = DynamicWireRopeSystem3D(config)
        
        # 运行仿真
        sol = system.simulate_dynamic_release(t_end=t_end, dt_output=dt_output)
        
        # 获取摘要信息
        summary = system.print_deployment_summary(sol)
        
        if not verbose:
            sys.stdout = old_stdout
        
        # 提取关键输出
        final_pos = summary['final_position']
        
        # 计算落点偏移（假设目标点在 (0, 0)）
        R_offset = np.sqrt(final_pos[0]**2 + final_pos[1]**2)
        
        result = {
            'T': summary['deployment_complete_time'],
            'R_offset': R_offset,
            'sigma_max': summary['max_stress_MPa'],
            'success': True,
            'target_reached': summary['target_reached'],
            'final_height': summary['final_height_above_seabed'],
            'safety_factor': summary['safety_factor'],
        }
        
    except Exception as e:
        if not verbose:
            sys.stdout = old_stdout
        print(f"仿真失败: {e}")
        result = {
            'T': np.nan,
            'R_offset': np.nan,
            'sigma_max': np.nan,
            'success': False,
            'error': str(e),
        }
    
    return result


def run_batch_simulations(samples: np.ndarray, param_names: List[str],
                           base_config: Optional[Dict] = None,
                           t_end: float = None, dt_output: float = 2.0,
                           save_interval: int = 5, save_path: str = None,
                           checkpoint_dir: str = None) -> pd.DataFrame:
    """
    批量运行高保真仿真
    
    参数:
        samples: (N, n_params) 样本矩阵
        param_names: 参数名称列表
        base_config: 基础配置
        t_end: 仿真结束时间
        dt_output: 输出时间步长
        save_interval: 中间保存间隔
        save_path: 结果保存路径
        checkpoint_dir: 检查点保存目录
    
    返回:
        results_df: 包含输入参数和输出指标的 DataFrame
    """
    n_samples = samples.shape[0]
    print(f"\n{'='*70}")
    print(f"开始批量仿真：共 {n_samples} 个样本")
    print(f"{'='*70}\n")
    
    # 初始化结果存储
    results = []
    start_time = time.time()
    
    for i in range(n_samples):
        sample = samples[i]
        config = create_config_from_sample(sample, param_names, base_config)
        
        print(f"[{i+1}/{n_samples}] 运行仿真...")
        print(f"  参数: {dict(zip(param_names, sample))}")
        
        sim_start = time.time()
        result = run_single_simulation(config, t_end=t_end, dt_output=dt_output, verbose=False)
        sim_time = time.time() - sim_start
        
        # 合并输入参数和输出结果
        row = {name: sample[j] for j, name in enumerate(param_names)}
        row.update(result)
        row['sim_time'] = sim_time
        results.append(row)
        
        if result['success']:
            print(f"  结果: T={result['T']:.2f}s, R_offset={result['R_offset']:.2f}m, "
                  f"σ_max={result['sigma_max']:.2f}MPa")
        else:
            print(f"  失败: {result.get('error', 'Unknown error')}")
        print(f"  耗时: {sim_time:.1f}s")
        print()
        
        # 中间保存
        if save_path and (i + 1) % save_interval == 0:
            temp_df = pd.DataFrame(results)
            if checkpoint_dir:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f'simulation_results_checkpoint_{i+1}.csv'
                )
            else:
                checkpoint_path = save_path.replace('.csv', f'_checkpoint_{i+1}.csv')
            temp_df.to_csv(checkpoint_path, index=False)
            print(f"  [检查点] 已保存 {i+1} 个样本结果: {checkpoint_path}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"批量仿真完成：总耗时 {total_time/60:.1f} 分钟")
    print(f"{'='*70}\n")
    
    # 创建结果 DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存最终结果
    if save_path:
        results_df.to_csv(save_path, index=False)
        print(f"结果已保存至: {save_path}")
    
    return results_df


# ==============================================================================
# 敏感性指标计算模块
# ==============================================================================

def compute_sensitivity_indices(X: np.ndarray, Y: np.ndarray, 
                                  param_names: List[str]) -> Dict:
    """
    计算标准化回归系数（SRC）和秩相关系数（SRRC）
    
    参数:
        X: (N, n_params) 输入样本矩阵
        Y: (N,) 单个输出响应向量
        param_names: 参数名称列表
    
    返回:
        indices: 包含 SRC 和 SRRC 的字典
    """
    # 移除 NaN 值
    valid_mask = ~np.isnan(Y)
    X_valid = X[valid_mask]
    Y_valid = Y[valid_mask]
    
    if len(Y_valid) < 3:
        return {name: {'SRC': np.nan, 'SRRC': np.nan} for name in param_names}
    
    # 标准化输入
    X_mean = X_valid.mean(axis=0)
    X_std = X_valid.std(axis=0)
    X_std[X_std < 1e-12] = 1.0  # 避免除零
    X_stdzd = (X_valid - X_mean) / X_std
    
    # 标准化输出
    Y_mean = Y_valid.mean()
    Y_std = Y_valid.std()
    if Y_std < 1e-12:
        Y_std = 1.0
    Y_stdzd = (Y_valid - Y_mean) / Y_std
    
    # 线性回归获取 SRC
    reg = LinearRegression(fit_intercept=True).fit(X_stdzd, Y_stdzd)
    beta = reg.coef_
    
    # R² 值
    r2 = reg.score(X_stdzd, Y_stdzd)
    
    # Spearman 秩相关系数
    rho = []
    for j in range(X_valid.shape[1]):
        r, _ = spearmanr(X_valid[:, j], Y_valid)
        rho.append(r)
    rho = np.array(rho)
    
    # 组织结果
    indices = {}
    for i, name in enumerate(param_names):
        indices[name] = {
            'SRC': beta[i],
            'SRRC': rho[i],
            'abs_SRC': abs(beta[i]),
            'abs_SRRC': abs(rho[i]),
        }
    indices['R2'] = r2
    
    return indices


def compute_all_sensitivity(results_df: pd.DataFrame, param_names: List[str],
                             output_names: List[str],
                             circular_params: List[str] = None) -> Dict:
    """
    计算所有输出指标的敏感性
    
    参数:
        results_df: 仿真结果 DataFrame
        param_names: 输入参数名称列表
        output_names: 输出指标名称列表
        circular_params: 圆周变量列表（需要sin/cos变换）
    
    返回:
        all_indices: 嵌套字典 {output_name: {param_name: {SRC, SRRC}}}
    """
    if circular_params is None:
        circular_params = []
    
    # 构建输入矩阵，处理圆周变量
    X_cols = []
    expanded_param_names = []
    
    for name in param_names:
        if name in circular_params:
            # 圆周变量用 sin/cos 替代（角度转弧度）
            theta_rad = np.deg2rad(results_df[name].values)
            X_cols.append(np.sin(theta_rad))
            X_cols.append(np.cos(theta_rad))
            expanded_param_names.append(f'{name}_sin')
            expanded_param_names.append(f'{name}_cos')
        else:
            X_cols.append(results_df[name].values)
            expanded_param_names.append(name)
    
    X = np.column_stack(X_cols)
    
    all_indices = {}
    for output_name in output_names:
        Y = results_df[output_name].values
        indices = compute_sensitivity_indices(X, Y, expanded_param_names)
        
        # 对圆周变量，合并 sin/cos 的敏感性为单一指标
        for circ_name in circular_params:
            if circ_name in param_names:
                sin_key = f'{circ_name}_sin'
                cos_key = f'{circ_name}_cos'
                if sin_key in indices and cos_key in indices:
                    # 使用 sqrt(SRC_sin^2 + SRC_cos^2) 作为综合敏感性
                    src_sin = indices[sin_key]['SRC']
                    src_cos = indices[cos_key]['SRC']
                    srrc_sin = indices[sin_key]['SRRC']
                    srrc_cos = indices[cos_key]['SRRC']
                    
                    combined_src = np.sqrt(src_sin**2 + src_cos**2)
                    combined_srrc = np.sqrt(srrc_sin**2 + srrc_cos**2)
                    
                    indices[circ_name] = {
                        'SRC': combined_src,
                        'SRRC': combined_srrc,
                        'abs_SRC': combined_src,
                        'abs_SRRC': combined_srrc,
                        'SRC_sin': src_sin,
                        'SRC_cos': src_cos,
                        'SRRC_sin': srrc_sin,
                        'SRRC_cos': srrc_cos,
                    }
        
        all_indices[output_name] = indices
    
    return all_indices


def create_sensitivity_table(all_indices: Dict, param_names: List[str],
                              output_names: List[str]) -> pd.DataFrame:
    """
    创建敏感性汇总表格
    
    返回:
        table_df: 敏感性汇总 DataFrame
    """
    rows = []
    for param in param_names:
        row = {'Parameter': param}
        for output in output_names:
            row[f'{output}_SRC'] = all_indices[output][param]['SRC']
            row[f'{output}_SRRC'] = all_indices[output][param]['SRRC']
        rows.append(row)
    
    table_df = pd.DataFrame(rows)
    return table_df


# ==============================================================================
# 可视化模块
# ==============================================================================

def plot_sensitivity_heatmap(all_indices: Dict, param_names: List[str],
                              output_names: List[str], index_type: str = 'SRC',
                              save_path: str = None):
    """
    绘制敏感性热力图（小图：6×8 英寸，适合论文单栏）
    
    参数:
        all_indices: 敏感性指标字典
        param_names: 参数名称列表
        output_names: 输出名称列表
        index_type: 'SRC' 或 'SRRC'
        save_path: 保存路径
    """
    setup_publication_style()
    
    # 构建数据矩阵
    data = np.zeros((len(param_names), len(output_names)))
    for i, param in enumerate(param_names):
        for j, output in enumerate(output_names):
            data[i, j] = all_indices[output][param][index_type]
    
    # 小图尺寸：宽6英寸，高8英寸
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # 绘制热力图
    im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # 设置刻度标签
    param_labels = [PARAM_LABELS.get(p, p) for p in param_names]
    output_labels = [OUTPUT_LABELS.get(o, o).split('$')[1].split('$')[0] if '$' in OUTPUT_LABELS.get(o, o) else o for o in output_names]
    
    ax.set_xticks(range(len(output_names)))
    ax.set_xticklabels(output_labels, fontsize=11, fontweight='medium')
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(param_labels, fontsize=11)
    
    # 添加数值标注
    for i in range(len(param_names)):
        for j in range(len(output_names)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha='center', va='center', fontsize=10, fontweight='medium',
                          color='white' if abs(data[i, j]) > 0.5 else 'black')
    
    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(f'{index_type}', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_title(f'Sensitivity Analysis ({index_type})', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Output Response', fontsize=12, fontweight='medium')
    ax.set_ylabel('Input Parameter', fontsize=12, fontweight='medium')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"热力图已保存至: {save_path}")
    
    plt.close(fig)


def plot_sensitivity_bar(all_indices: Dict, param_names: List[str],
                          output_names: List[str], save_dir: str = None):
    """
    绘制敏感性条形图（每个输出响应单独一张小图：宽8cm×高6cm）
    
    参数:
        all_indices: 敏感性指标字典
        param_names: 参数名称列表
        output_names: 输出名称列表
        save_dir: 保存目录（每个输出生成一个文件）
    """
    setup_publication_style()
    
    param_labels = [PARAM_LABELS.get(p, p) for p in param_names]
    colors = ['#0072B2', '#D55E00']  # 色盲友好配色
    
    # 单位转换：cm -> 英寸 (1英寸 = 2.54cm)
    cm_to_inch = 1 / 2.54
    fig_width = 8 * cm_to_inch   # 8cm 宽
    fig_height = 6 * cm_to_inch  # 6cm 高
    
    for output in output_names:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        src_values = [all_indices[output][p]['SRC'] for p in param_names]
        srrc_values = [all_indices[output][p]['SRRC'] for p in param_names]
        
        x = np.arange(len(param_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, src_values, width, label='SRC', color=colors[0], edgecolor='white', linewidth=0.3)
        bars2 = ax.bar(x + width/2, srrc_values, width, label='SRRC', color=colors[1], edgecolor='white', linewidth=0.3)
        
        # 动态计算坐标轴范围
        all_values = src_values + srrc_values
        y_max = max(max(all_values), 0.1)
        y_min = min(min(all_values), -0.1)
        y_margin = max(abs(y_max), abs(y_min)) * 0.15
        y_lim = max(abs(y_max) + y_margin, abs(y_min) + y_margin)
        y_lim = min(y_lim, 1.1)  # 上限不超过1.1
        
        ax.set_xlabel('Parameter', fontsize=8, fontweight='medium')
        ax.set_ylabel('Sensitivity Index', fontsize=8, fontweight='medium')
        ax.set_title(OUTPUT_LABELS.get(output, output), fontsize=9, fontweight='bold', pad=5)
        ax.set_xticks(x)
        ax.set_xticklabels(param_labels, fontsize=7, rotation=45, ha='right')
        ax.legend(fontsize=6, loc='best', framealpha=0.9)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_ylim(-y_lim, y_lim)
        ax.tick_params(axis='both', labelsize=7)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, f'sensitivity_bar_{output}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"条形图已保存至: {save_path}")
        
        plt.show()
        plt.close(fig)


def plot_sensitivity_radar(all_indices: Dict, param_names: List[str],
                            output_names: List[str], save_path: str = None):
    """
    绘制敏感性雷达图（小图：6×8 英寸）
    展示各参数对不同输出的综合敏感性
    
    参数:
        all_indices: 敏感性指标字典
        param_names: 参数名称列表
        output_names: 输出名称列表
        save_path: 保存路径
    """
    setup_publication_style()
    
    # 小图尺寸：6×8 英寸
    fig, ax = plt.subplots(figsize=(6, 8), subplot_kw=dict(projection='polar'))
    
    param_labels = [PARAM_LABELS.get(p, p) for p in param_names]
    n_params = len(param_names)
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, n_params, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    # 配色方案
    colors = ['#0072B2', '#D55E00', '#009E73']  # 色盲友好
    markers = ['o', 's', '^']
    
    for idx, output in enumerate(output_names):
        # 使用绝对值 SRC
        values = [abs(all_indices[output][p]['SRC']) for p in param_names]
        values += values[:1]  # 闭合
        
        ax.plot(angles, values, 'o-', linewidth=2, label=output, 
                color=colors[idx % len(colors)], marker=markers[idx % len(markers)], markersize=6)
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(param_labels, fontsize=11)
    
    # 设置径向范围
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    
    ax.set_title('Sensitivity Radar Chart (|SRC|)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"雷达图已保存至: {save_path}")
    
    plt.close(fig)


def plot_bar_with_boxplot(results_df: pd.DataFrame, all_indices: Dict, 
                           param_names: List[str], output_names: List[str],
                           save_dir: str = None):
    """
    绘制条形图+箱线图组合（每个输出响应单独一张大图：16×6 英寸）
    上半部分：敏感性条形图
    下半部分：参数分布箱线图
    
    参数:
        results_df: 仿真结果 DataFrame
        all_indices: 敏感性指标字典
        param_names: 参数名称列表
        output_names: 输出名称列表
        save_dir: 保存目录
    """
    setup_publication_style()
    
    param_labels = [PARAM_LABELS.get(p, p) for p in param_names]
    n_params = len(param_names)
    
    for output in output_names:
        # 大图尺寸：宽16英寸，高6英寸
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6), height_ratios=[1, 1])
        
        # === 上半部分：敏感性条形图 ===
        src_values = [all_indices[output][p]['SRC'] for p in param_names]
        srrc_values = [all_indices[output][p]['SRRC'] for p in param_names]
        
        x = np.arange(n_params)
        width = 0.35
        colors = ['#0072B2', '#D55E00']
        
        bars1 = ax1.bar(x - width/2, src_values, width, label='SRC', color=colors[0], edgecolor='white', linewidth=0.5)
        bars2 = ax1.bar(x + width/2, srrc_values, width, label='SRRC', color=colors[1], edgecolor='white', linewidth=0.5)
        
        # 添加数值标签
        for bar, val in zip(bars1, src_values):
            if abs(val) > 0.05:
                offset = 0.03 if val > 0 else -0.03
                ax1.text(bar.get_x() + bar.get_width()/2, val + offset, 
                        f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=9)
        
        ax1.set_ylabel('Sensitivity Index', fontsize=11, fontweight='medium')
        ax1.set_title(f'{OUTPUT_LABELS.get(output, output)} - Sensitivity & Parameter Distribution', 
                     fontsize=13, fontweight='bold', pad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels([])
        ax1.legend(fontsize=10, loc='upper right', framealpha=0.9)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(-1.15, 1.15)
        ax1.tick_params(axis='both', labelsize=10)
        
        # === 下半部分：参数分布箱线图 ===
        # 对每个参数的数据进行标准化以便在同一坐标系显示
        box_data = []
        for param in param_names:
            data = results_df[param].dropna().values
            # 标准化到 [0, 1] 范围
            if data.max() > data.min():
                normalized = (data - data.min()) / (data.max() - data.min())
            else:
                normalized = np.zeros_like(data)
            box_data.append(normalized)
        
        bp = ax2.boxplot(box_data, positions=x, widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor='#E8E8E8', color='#333333'),
                        medianprops=dict(color='#D55E00', linewidth=2),
                        whiskerprops=dict(color='#333333'),
                        capprops=dict(color='#333333'),
                        flierprops=dict(marker='o', markerfacecolor='#999999', markersize=4, alpha=0.6))
        
        # 根据敏感性大小设置箱线图颜色
        abs_src = [abs(v) for v in src_values]
        max_src = max(abs_src) if max(abs_src) > 0 else 1
        for patch, src in zip(bp['boxes'], abs_src):
            intensity = src / max_src
            color = plt.cm.Blues(0.3 + 0.6 * intensity)
            patch.set_facecolor(color)
        
        ax2.set_xlabel('Parameter', fontsize=11, fontweight='medium')
        ax2.set_ylabel('Normalized Value', fontsize=11, fontweight='medium')
        ax2.set_xticks(x)
        ax2.set_xticklabels(param_labels, fontsize=10, rotation=45, ha='right')
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.tick_params(axis='both', labelsize=10)
        
        # 添加说明文字
        ax2.text(0.02, 0.95, 'Box color intensity \u221d |SRC|', transform=ax2.transAxes,
                fontsize=9, va='top', style='italic', color='#666666')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, f'sensitivity_bar_box_{output}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"条形图+箱线图已保存至: {save_path}")
        
        plt.close(fig)


def plot_scatter_matrix(results_df: pd.DataFrame, param_names: List[str],
                         output_names: List[str], save_path: str = None):
    """
    绘制输入-输出散点图矩阵（大图：16×6 英寸）
    """
    setup_publication_style()
    
    n_params = len(param_names)
    n_outputs = len(output_names)
    
    # 大图尺寸：宽16英寸，高6英寸
    fig, axes = plt.subplots(n_outputs, n_params, figsize=(16, 6))
    
    for i, output in enumerate(output_names):
        for j, param in enumerate(param_names):
            ax = axes[i, j] if n_outputs > 1 else axes[j]
            
            x = results_df[param].values
            y = results_df[output].values
            
            ax.scatter(x, y, alpha=0.6, s=25, c='#0072B2', edgecolors='white', linewidth=0.3)
            
            # 拟合线性趋势
            valid = ~(np.isnan(x) | np.isnan(y))
            if np.sum(valid) > 2:
                z = np.polyfit(x[valid], y[valid], 1)
                p = np.poly1d(z)
                x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
                ax.plot(x_line, p(x_line), 'r-', linewidth=1.5, alpha=0.8)
                
                # 添加 R² 标注
                y_pred = p(x[valid])
                ss_res = np.sum((y[valid] - y_pred) ** 2)
                ss_tot = np.sum((y[valid] - np.mean(y[valid])) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                ax.text(0.95, 0.05, f'$R^2$={r2:.2f}', transform=ax.transAxes, 
                       fontsize=8, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
            
            if i == n_outputs - 1:
                ax.set_xlabel(PARAM_LABELS.get(param, param), fontsize=9, fontweight='medium')
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(OUTPUT_LABELS.get(output, output).split('$')[0].strip() if '$' in OUTPUT_LABELS.get(output, output) else output, 
                             fontsize=9, fontweight='medium')
            
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"散点图矩阵已保存至: {save_path}")
    
    plt.close(fig)


def _fit_polynomial_with_r2(x: np.ndarray, y: np.ndarray, degree: int):
    """Fit a polynomial trend and return the model and in-sample R2."""
    valid = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid]
    y_valid = y[valid]

    if len(x_valid) <= degree or len(np.unique(x_valid)) <= degree:
        return None, np.nan

    coeff = np.polyfit(x_valid, y_valid, degree)
    model = np.poly1d(coeff)
    y_pred = model(x_valid)
    ss_res = np.sum((y_valid - y_pred) ** 2)
    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return model, r2


def _binned_median_trend(x: np.ndarray, y: np.ndarray, n_bins: int = 6):
    """Compute binned medians for a robust visual trend line."""
    valid = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid]
    y_valid = y[valid]

    if len(x_valid) < 3 or np.nanmax(x_valid) <= np.nanmin(x_valid):
        return np.array([]), np.array([])

    edges = np.linspace(np.nanmin(x_valid), np.nanmax(x_valid), n_bins + 1)
    centers, medians = [], []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (x_valid >= edges[i]) & (x_valid <= edges[i + 1])
        else:
            mask = (x_valid >= edges[i]) & (x_valid < edges[i + 1])

        if np.sum(mask) == 0:
            continue

        centers.append(np.median(x_valid[mask]))
        medians.append(np.median(y_valid[mask]))

    return np.asarray(centers), np.asarray(medians)


def plot_sigma_max_nonlinear_trends(results_df: pd.DataFrame,
                                    key_params: List[str] = None,
                                    save_path: str = None):
    """
    绘制 sigma_max 与关键参数之间的非线性散点趋势图。

    该图用于补充 SRC/SRRC 分析，直观展示 sigma_max 在线性 R2 较低时
    仍可能存在局部非线性、离散性和瞬态峰值响应。
    """
    setup_publication_style()

    if key_params is None:
        key_params = ['wave_amplitude', 'd', 'release_speed', 'wave_period']

    key_params = [p for p in key_params if p in results_df.columns]
    if 'sigma_max' not in results_df.columns or not key_params:
        print("跳过 sigma_max 非线性趋势图：缺少 sigma_max 或关键参数列。")
        return

    n_params = len(key_params)
    n_cols = 2
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2, 5.6), squeeze=False)

    scatter_color = '#0072B2'
    quadratic_color = '#D55E00'
    linear_color = '#666666'
    median_color = '#111111'

    for idx, param in enumerate(key_params):
        ax = axes[idx // n_cols, idx % n_cols]
        x = pd.to_numeric(results_df[param], errors='coerce').to_numpy(dtype=float)
        y = pd.to_numeric(results_df['sigma_max'], errors='coerce').to_numpy(dtype=float)
        valid = ~(np.isnan(x) | np.isnan(y))

        ax.scatter(
            x[valid], y[valid],
            alpha=0.68, s=28, c=scatter_color,
            edgecolors='white', linewidth=0.35, label='LHS samples'
        )

        if np.sum(valid) > 3:
            x_line = np.linspace(np.nanmin(x[valid]), np.nanmax(x[valid]), 200)

            linear_model, r2_linear = _fit_polynomial_with_r2(x, y, degree=1)
            if linear_model is not None:
                ax.plot(
                    x_line, linear_model(x_line),
                    linestyle='--', linewidth=1.2, color=linear_color,
                    alpha=0.85, label='Linear fit'
                )

            quadratic_model, r2_quadratic = _fit_polynomial_with_r2(x, y, degree=2)
            if quadratic_model is not None:
                ax.plot(
                    x_line, quadratic_model(x_line),
                    linestyle='-', linewidth=1.8, color=quadratic_color,
                    label='Quadratic trend'
                )

            bin_x, bin_y = _binned_median_trend(x, y, n_bins=6)
            if len(bin_x) > 1:
                ax.plot(
                    bin_x, bin_y,
                    linestyle='-', linewidth=1.2, color=median_color,
                    marker='o', markersize=4, markerfacecolor='white',
                    label='Binned median'
                )

            label_lines = []
            if not np.isnan(r2_linear):
                label_lines.append(rf'$R^2_{{lin}}$={r2_linear:.2f}')
            if not np.isnan(r2_quadratic):
                label_lines.append(rf'$R^2_{{quad}}$={r2_quadratic:.2f}')
            if label_lines:
                ax.text(
                    0.04, 0.96, '\n'.join(label_lines),
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.82, edgecolor='none')
                )

        ax.set_xlabel(PARAM_LABELS.get(param, param), fontsize=9, fontweight='medium')
        ax.set_ylabel(OUTPUT_LABELS['sigma_max'], fontsize=9, fontweight='medium')
        ax.grid(alpha=0.25, linestyle='--', linewidth=0.6)
        ax.tick_params(axis='both', labelsize=8)
        ax.text(
            0.5, -0.28, f'({chr(97 + idx)})',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=9, fontweight='medium'
        )

    for idx in range(n_params, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis('off')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center', ncol=4, frameon=False,
        bbox_to_anchor=(0.5, 1.02), fontsize=8
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"sigma_max 非线性趋势图已保存至: {save_path}")

        root, ext = os.path.splitext(save_path)
        if ext.lower() != '.svg':
            svg_path = f'{root}.svg'
            plt.savefig(svg_path, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"sigma_max 非线性趋势图 SVG 已保存至: {svg_path}")

    plt.close(fig)


# ==============================================================================
# 主程序入口
# ==============================================================================

def run_sensitivity_analysis(n_samples: int = 50, 
                              param_bounds: Dict = None,
                              base_config: Dict = None,
                              t_end: float = None,
                              output_dir: str = None,
                              seed: int = 42,
                              circular_params: List[str] = None):
    """
    运行完整的敏感性分析流程
    
    参数:
        n_samples: 样本数量（建议 10~15 × 因子数）
        param_bounds: 参数边界字典
        base_config: 基础仿真配置
        t_end: 仿真结束时间
        output_dir: 输出目录
        seed: 随机种子
        circular_params: 圆周变量列表（需要sin/cos变换处理）
    
    返回:
        results_df: 仿真结果 DataFrame
        sensitivity_table: 敏感性汇总表
    """
    if param_bounds is None:
        param_bounds = PARAM_BOUNDS
    
    if circular_params is None:
        circular_params = CIRCULAR_PARAMS
    
    param_names = list(param_bounds.keys())
    
    # 创建输出目录
    output_paths = prepare_output_dirs(output_dir)
    output_root = output_paths['root']
    data_dir = output_paths['data']
    figures_dir = output_paths['figures']
    checkpoints_dir = output_paths['checkpoints']
    
    print("="*70)
    print("        敏感性分析：LHS + 高保真仿真 + SRC/SRRC")
    print("="*70)
    print(f"\n样本数量: {n_samples}")
    print(f"分析参数: {param_names}")
    print(f"输出目录: {output_root}")
    print()
    
    # Step 1: LHS 采样
    print("Step 1: 生成 LHS 采样...")
    samples = generate_lhs_samples(n_samples, param_bounds, seed=seed)
    samples_df = samples_to_dataframe(samples, param_names)
    lhs_samples_path = os.path.join(data_dir, 'lhs_samples.csv')
    samples_df.to_csv(lhs_samples_path, index=False)
    print(f"  采样完成，共 {n_samples} 个样本")
    print(f"  样本已保存至: {lhs_samples_path}")
    print()
    
    # Step 2: 批量仿真
    print("Step 2: 批量运行高保真仿真...")
    results_path = os.path.join(data_dir, 'simulation_results.csv')
    results_df = run_batch_simulations(
        samples, param_names, base_config=base_config,
        t_end=t_end, dt_output=2.0,
        save_interval=5, save_path=results_path, checkpoint_dir=checkpoints_dir
    )
    
    # 统计成功率
    success_rate = results_df['success'].mean() * 100
    print(f"\n仿真成功率: {success_rate:.1f}%")
    
    # Step 3: 计算敏感性指标
    print("\nStep 3: 计算敏感性指标 (SRC/SRRC)...")
    if circular_params:
        print(f"  圆周变量处理: {circular_params} → sin/cos 变换")
    all_indices = compute_all_sensitivity(results_df, param_names, OUTPUT_NAMES, 
                                           circular_params=circular_params)
    
    # 打印结果
    print("\n" + "-"*70)
    print("敏感性分析结果汇总")
    print("-"*70)
    
    for output in OUTPUT_NAMES:
        print(f"\n【{OUTPUT_LABELS.get(output, output)}】 (R² = {all_indices[output]['R2']:.3f})")
        print(f"{'参数':<20} {'SRC':>10} {'SRRC':>10}")
        print("-"*42)
        for param in param_names:
            src = all_indices[output][param]['SRC']
            srrc = all_indices[output][param]['SRRC']
            print(f"{param:<20} {src:>+10.3f} {srrc:>+10.3f}")
    
    # 创建汇总表格
    sensitivity_table = create_sensitivity_table(all_indices, param_names, OUTPUT_NAMES)
    sensitivity_indices_path = os.path.join(data_dir, 'sensitivity_indices.csv')
    sensitivity_table.to_csv(sensitivity_indices_path, index=False)
    print(f"\n敏感性指标已保存至: {sensitivity_indices_path}")
    
    # Step 4: 可视化
    print("\nStep 4: 生成可视化图表...")
    
    # 4.1 热力图（小图 6×8）
    plot_sensitivity_heatmap(all_indices, param_names, OUTPUT_NAMES, 
                             index_type='SRC',
                             save_path=os.path.join(figures_dir, 'sensitivity_heatmap_SRC.png'))
    
    plot_sensitivity_heatmap(all_indices, param_names, OUTPUT_NAMES,
                             index_type='SRRC',
                             save_path=os.path.join(figures_dir, 'sensitivity_heatmap_SRRC.png'))
    
    # 4.2 柱状图（每个输出单独一张小图 6×8）
    plot_sensitivity_bar(all_indices, param_names, OUTPUT_NAMES,
                         save_dir=figures_dir)
    
    # 4.3 雷达图（小图 6×8）
    plot_sensitivity_radar(all_indices, param_names, OUTPUT_NAMES,
                           save_path=os.path.join(figures_dir, 'sensitivity_radar.png'))
    
    # 4.4 条形图+箱线图组合（每个输出单独一张大图 16×6）
    plot_bar_with_boxplot(results_df, all_indices, param_names, OUTPUT_NAMES,
                          save_dir=figures_dir)
    
    # 4.5 散点图矩阵（大图 16×6）
    plot_scatter_matrix(results_df, param_names, OUTPUT_NAMES,
                        save_path=os.path.join(figures_dir, 'scatter_matrix.png'))

    # 4.6 图20：sigma_max 关键参数非线性趋势图
    plot_sigma_max_nonlinear_trends(
        results_df,
        key_params=['wave_amplitude', 'd', 'release_speed', 'wave_period'],
        save_path=os.path.join(figures_dir, 'fig_20_sigma_max_nonlinear_trends.png')
    )
    
    # 识别主导参数
    print("\n" + "="*70)
    print("主导参数识别")
    print("="*70)
    
    for output in OUTPUT_NAMES:
        sorted_params = sorted(param_names, 
                               key=lambda p: abs(all_indices[output][p]['SRC']), 
                               reverse=True)
        print(f"\n{output} 主导参数（按 |SRC| 排序）:")
        for i, param in enumerate(sorted_params[:3], 1):
            src = all_indices[output][param]['SRC']
            print(f"  {i}. {param}: SRC = {src:+.3f}")
    
    print("\n" + "="*70)
    print("敏感性分析完成！")
    print("="*70)
    print(f"\n所有结果已保存至: {output_root}")
    print("该数据可直接作为 Kriging 代理模型的初始训练集。")
    
    return results_df, sensitivity_table, all_indices


def redraw_fig20_from_saved_results(results_csv: str = None, output_dir: str = None):
    """Redraw Fig. 20 from the saved sensitivity simulation table."""
    output_paths = prepare_output_dirs(output_dir)
    data_dir = output_paths['data']
    figures_dir = output_paths['figures']

    if results_csv is None:
        results_csv = os.path.join(data_dir, 'simulation_results.csv')

    if not os.path.exists(results_csv):
        raise FileNotFoundError(
            f"未找到图20原始数据: {results_csv}\n"
            "请先运行完整敏感性分析，或用 --results-csv 指定已有 simulation_results.csv。"
        )

    results_df = pd.read_csv(results_csv)
    fig20_columns = ['wave_amplitude', 'd', 'release_speed', 'wave_period', 'sigma_max']
    missing = [col for col in fig20_columns if col not in results_df.columns]
    if missing:
        raise ValueError(f"simulation_results.csv 缺少图20所需列: {missing}")

    fig20_data_path = os.path.join(data_dir, 'fig_20_sigma_max_nonlinear_trends_data.csv')
    results_df[fig20_columns].to_csv(fig20_data_path, index=False)
    print(f"图20原始数据子集已保存至: {fig20_data_path}")

    plot_sigma_max_nonlinear_trends(
        results_df,
        key_params=['wave_amplitude', 'd', 'release_speed', 'wave_period'],
        save_path=os.path.join(figures_dir, 'fig_20_sigma_max_nonlinear_trends.png')
    )


# ==============================================================================
# 脚本入口
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sensitivity analysis and figure generation.')
    parser.add_argument(
        '--fig20-only',
        action='store_true',
        help='Only redraw Fig. 20 from saved simulation_results.csv; do not rerun simulations.'
    )
    parser.add_argument(
        '--results-csv',
        default=None,
        help='Path to an existing simulation_results.csv used with --fig20-only.'
    )
    parser.add_argument(
        '--output-dir',
        default=str(DEFAULT_OUTPUT_ROOT),
        help='Output directory root. Default: sensitivity_convergence/results/sensitivity_analysis.'
    )
    args = parser.parse_args()

    if args.fig20_only:
        redraw_fig20_from_saved_results(results_csv=args.results_csv, output_dir=args.output_dir)
        raise SystemExit(0)

    # 设置参数（可根据需要调整）
    N_SAMPLES = 80  # 样本数：8个因子 × 10 = 80（7个原始因子 + wave_direction的sin/cos扩展）
    
    # 基础配置：在 `high_fidelity_sim.py` 默认配置基础上覆盖敏感性分析固定项
    BASE_CONFIG = {
        # 初始离散设置
        'N': 2,                  # 初始2个节点（1段）
        'L': 100,                # 初始长度 100m
        
        # 环境参数
        'total_water_depth': 1500,
        'seabed_depth': -1500,
        'target_height_above_seabed': 25.0,
        
        # 下放参数
        'total_segments': 15,    # 15段 × 100m
        'segment_length': 100.0,
        'total_depth': 1500,
        'active_tension_control_enabled': False,
        'extra_sim_time_after_target': 50.0,
    }
    
    # 运行敏感性分析
    # t_end=None 表示根据每个样本的下放速度自动计算仿真时间
    # 确保装备下放至目标深度后继续模拟50s
    # circular_params 用于处理波浪方向的圆周变量特性
    results_df, sensitivity_table, all_indices = run_sensitivity_analysis(
        n_samples=N_SAMPLES,
        param_bounds=PARAM_BOUNDS,
        base_config=BASE_CONFIG,
        t_end=None,  # 自动计算，确保到达目标深度 + 50s
        output_dir=str(DEFAULT_OUTPUT_ROOT),
        seed=42,
        circular_params=CIRCULAR_PARAMS  # wave_direction 用 sin/cos 变换
    )
