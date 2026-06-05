# Subsea Deployment Simulation / 水下放物系统仿真工程

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[English](#english) | [中文](#chinese)**

<a id="english"></a>
## English

### Overview
This repository contains a high-fidelity engineering simulation program for subsea wire rope deployment (lowering) systems, along with scripts for parameter sensitivity and convergence analyses. It primarily models a 3D subsea payload deployment scenario considering comprehensive environmental and physical factors such as vessel heave compensation, wave excitation, subsea currents, and wire rope dynamics.

### Key Features
- **High-Fidelity Simulation (`high_fidelity_sim.py`)**: Subsea lowering dynamics modeled via `scipy.integrate.solve_ivp`. It encompasses:
  - Six degrees of freedom (6-DOF) vessel motion and wave response.
  - Active Tension Control (ATC) algorithms.
  - Subsea current modeling and dynamic responses of the payload and wire rope.
- **Sensitivity Analysis (`sensitivity_analysis.py`)**: Uses Latin Hypercube Sampling (LHS) and polynomial fitting to analyze the impact of various physical parameters (e.g., stiffness, payload mass, wave conditions) on peak tension.
- **Convergence Analysis (`convergence_analysis.py`)**: Evaluates the computational convergence of the ODE solver and the physical reliability of the results under various step sizes and mesh configurations.
- **Ocean Current Analysis (`sea_current.ipynb`)**: Jupyter notebook for interactive visualization and analysis of sea current profiles.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/APEX-Start/Subsea_deployment_simulation.git
   cd Subsea_deployment_simulation
   ```

2. Install dependency packages:
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start
- Run a single high-fidelity simulation and visualize the summary results:
  ```bash
  python high_fidelity_sim.py
  ```
- Run the parameter sensitivity analysis (requires `high_fidelity_sim.py` configuration):
  ```bash
  python sensitivity_analysis.py
  ```
- Interact with the ocean current dataset or models:
  ```bash
  jupyter notebook sea_current.ipynb
  ```

---

<a id="chinese"></a>
## 中文 / Chinese

### 项目简介
本项目包含一个三维钢丝绳水下放物（吊放）系统的高保真工程仿真程序，以及配套的参数敏感性分析和网格/步长收敛性分析代码。项目主要用于求解和分析深水或海洋工程情况下的装备下放动力学问题，涵盖了母船补偿、波浪激励、海流扰动及管线/钢丝绳动态响应。

### 核心功能与文件结构
- **`high_fidelity_sim.py` (主仿真模块)**: 
  使用 `solve_ivp` 进行非线性常微分方程组求解，集成了以下特性：
  - 具备母船六自由度（6-DOF）运动模型与波浪响应（RAO）。
  - 支持主动张力控制（ATC），有效抑制极值载荷。
  - 考虑波浪流体力、复杂海流剖面以及钢丝绳与负载在水流中的非线性阻力。
- **`sensitivity_analysis.py` (敏感性分析)**:
  利用拉丁超立方抽样 (LHS) 在多元空间进行样本生成，通过评估大量工况分析输入参数（刚度系数、负载、波要素等）对结果（如最大缆绳张力）的敏感性响应。
- **`convergence_analysis.py` (收敛性分析)**:
  测试不同离散段数、求解器相对/绝对容差下收敛性，并出具对比分析结果。
- **`sea_current.ipynb`**:
  用于海流流场探索与数据分析的 Jupyter Notebook 交互文件。

### 安装依赖

1. 获取代码:
   ```bash
   git clone https://github.com/APEX-Start/Subsea_deployment_simulation.git
   cd Subsea_deployment_simulation
   ```

2. 安装所需 Python 运行库:
   ```bash
   pip install -r requirements.txt
   ```

### 快速使用
- **直接运行完整下放仿真**，查看控制台输出及自动生成的结果图表：
  ```bash
  python high_fidelity_sim.py
  ```
- **执行敏感性分析批处理**：
  ```bash
  python sensitivity_analysis.py
  ```
- **查看海流环境探索**：
  ```bash
  jupyter notebook sea_current.ipynb
  ```

### 注释及开发说明
主逻辑代码 `high_fidelity_sim.py` 内含详细的中文工程参数释义。如需修改放缆深度、水文环境或仿真时长，可以直接调整该文件顶部的 `DEFAULT_CONFIG` 字典。
