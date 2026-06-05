# Subsea Deployment Simulation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-fidelity simulation and analysis tools for 3D subsea wire-rope deployment dynamics.

**[English](#english) | [中文](#chinese)**

---

<a id="english"></a>

## English

### Overview

This repository provides a physics-based simulation framework for subsea payload deployment (lowering) systems. It models the complete deployment dynamics including:

- **6-DOF vessel motion** with wave response (RAO)
- **Active Tension Control (ATC)** for extreme load suppression
- **Wire rope dynamics** — catenary shape, axial elasticity, hydrodynamic drag
- **Ocean current profiles** — exponential decay model (South China Sea parameters)
- **Payload hydrodynamics** — added mass, drag, and buoyancy effects

The simulation engine uses `scipy.integrate.solve_ivp` to solve the coupled nonlinear ODE system.

### Project Structure

```
Subsea_deployment_simulation/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── src/
│   └── subsea_deployment_simulation/
│       ├── __init__.py
│       ├── high_fidelity_sim.py      # Core simulation model
│       └── plotting_style.py          # Publication-style plotting helpers
├── scripts/
│   ├── convergence_analysis.py        # Mesh/solver convergence study
│   ├── sensitivity_analysis.py        # LHS-based parameter sensitivity
│   └── visualization/                 # Paper figure generators (plot_01 … plot_12)
├── notebooks/
│   └── sea_current.ipynb              # Interactive ocean current exploration
├── data/                              # Small example data (if any)
├── results/                           # Generated outputs (not tracked)
│   ├── visualization/                 # Simulation payloads (npz)
│   └── figures/                       # Generated SVG figures
├── docs/
│   ├── usage.md
│   ├── visualization.md
│   └── reproduction.md
└── tests/
    └── test_imports.py
```

### Installation

```bash
git clone https://github.com/APEX-Start/Subsea_deployment_simulation.git
cd Subsea_deployment_simulation
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### Quick Start

**Run the main simulation:**

```bash
python -m subsea_deployment_simulation.high_fidelity_sim
```

**Run convergence analysis:**

```bash
python scripts/convergence_analysis.py --mode catenary
python scripts/convergence_analysis.py --mode mesh --quick
```

**Run sensitivity analysis:**

```bash
python scripts/sensitivity_analysis.py --quick
python scripts/sensitivity_analysis.py --samples 50
```

**Generate paper figures** (requires simulation output first):

```bash
python scripts/visualization/plot_01_upper_boundary_heave_disturbance.py
python scripts/visualization/plot_11_operational_window.py
```

### Configuration

All default simulation parameters are in `src/subsea_deployment_simulation/high_fidelity_sim.py` as the `DEFAULT_CONFIG` dictionary. Key parameters include:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | 2 | Number of wire rope segments |
| `L` | 100 m | Wire rope length |
| `d` | 0.032 m | Wire rope diameter |
| `m_b` | 500 kg | Payload mass |
| `total_water_depth` | 1500 m | Water depth |
| `release_speed` | 0.6 m/s | Lowering speed |

---

<a id="chinese"></a>

## 中文

### 项目简介

本项目是一个三维钢丝绳水下放物（吊放）系统的高保真仿真框架，涵盖母船六自由度运动、主动张力控制（ATC）、钢丝绳动力学、海流扰动和载荷水动力等完整物理过程。核心求解器基于 `scipy.integrate.solve_ivp`。

### 核心功能

- **`high_fidelity_sim.py`** — 主仿真模块，集成母船运动、波浪激励、钢丝绳非线性动力学
- **`convergence_analysis.py`** — 网格/步长收敛性验证与静态悬链线对比
- **`sensitivity_analysis.py`** — 拉丁超立方采样（LHS）参数敏感性分析
- **`scripts/visualization/`** — 论文图表生成脚本（12 张图）
- **`sea_current.ipynb`** — 海流流速剖面交互分析

### 安装与使用

```bash
git clone https://github.com/APEX-Start/Subsea_deployment_simulation.git
cd Subsea_deployment_simulation
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -e .
```

运行主仿真：

```bash
python -m subsea_deployment_simulation.high_fidelity_sim
```

参数修改请编辑 `src/subsea_deployment_simulation/high_fidelity_sim.py` 顶部的 `DEFAULT_CONFIG` 字典。

### 注释说明

主仿真代码内含详细中文工程参数释义。更多使用说明请参阅 `docs/` 目录。
