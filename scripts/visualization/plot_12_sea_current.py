from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from subsea_deployment_simulation.plotting_style import apply_times_new_roman_style


apply_times_new_roman_style()

# ==========================================
# 1. 顶刊级全局样式配置 (参考 Pareto 绘图风格)
# ==========================================


# ==========================================
# 2. 数据准备与拟合
# ==========================================
observed_depths = np.array([0, 25, 247, 411, 576, 740, 904, 1069, 1233, 1398])
observed_u = np.array([1.37, 1.32, 0.95, 0.59, 0.58, 0.57, 0.56, 0.56, 0.56, 0.56])
observed_v = observed_u * 0.2  

H = 1500.0  
full_depths = np.linspace(0, H, 500)

def exp_decay_model(z, a, b, c):
    return a + b * np.exp(-z / c)

# 拟合参数
popt_u, _ = curve_fit(exp_decay_model, observed_depths, observed_u, p0=[0.56, 0.8, 200.0])
popt_v, _ = curve_fit(exp_decay_model, observed_depths, observed_v, p0=[0.11, 0.2, 200.0])

fitted_u = exp_decay_model(full_depths, *popt_u)
fitted_v = exp_decay_model(full_depths, *popt_v)

# ==========================================
# 3. 高级可视化绘图
# ==========================================
fig, ax = plt.subplots(figsize=(12, 9.25))

# 绘制拟合曲线：使用加粗实线 (参考 Pareto 的稳重感)
ax.plot(fitted_u, -full_depths, label='Fitted $u$ (East)', color='#0072B2', linewidth=6, zorder=2)
ax.plot(fitted_v, -full_depths, label='Fitted $v$ (North)', color='#D55E00', linewidth=6, zorder=2)

# 绘制观测点：使用带边框的大型散点 (与 Pareto 散点风格统一)
ax.scatter(observed_u, -observed_depths, color='#0072B2', s=180, marker='o', 
           edgecolors='black', linewidth=1.5, label='Observed $u$', zorder=3)
ax.scatter(observed_v, -observed_depths, color='#D55E00', s=180, marker='s', 
           edgecolors='black', linewidth=1.5, label='Observed $v$', zorder=3)

# 坐标轴精细化调整
ax.set_xlabel('Current Velocity (m/s)', labelpad=5, fontweight='normal', fontsize=35)
ax.set_ylabel('Water Depth below Surface (m)', labelpad=5, fontweight='normal', fontsize=35)

# 刻度设置：向内、加粗、纯黑 (核心风格点)
ax.tick_params(axis='both', which='major', direction='in', length=12, width=4, pad=15, labelsize=35)
ax.tick_params(axis='both', which='minor', direction='in', length=6, width=3, labelsize=35)

# 边框加粗
for spine in ax.spines.values():
    spine.set_linewidth(4)
    spine.set_edgecolor('black')

# 网格线优化：轻微、虚线
ax.grid(False)

# 倒置 Y 轴并设置范围
ax.set_ylim(-H, 0)
ax.set_xlim(0, 1.6)

# 图例与布局
ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1, borderpad=0.8, 
          fontsize=30).get_frame().set_linewidth(1.5)
plt.tight_layout()

# 保存图像
_output_dir = Path(__file__).resolve().parents[2] / "results" / "figures"
_output_dir.mkdir(parents=True, exist_ok=True)
_out_path = _output_dir / 'fig_12_sea_current.svg'
plt.savefig(_out_path, format='svg', bbox_inches='tight', facecolor='white')
print(f'Saved SVG to: {_out_path}')
plt.show()