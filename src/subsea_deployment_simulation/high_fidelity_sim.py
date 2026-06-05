# -*- coding: utf-8 -*-
"""
三维钢丝绳水下放物系统 - 工程主版本

工程约定:
- 设计判据默认优先看原始峰值应力，滤波值只用于趋势分析和汇报展示。
- 25 m 停止条件默认使用 ODE 事件精确触发；若关闭事件检测，则回退到离散检查。
- 所有摘要、导出和注释都应与当前实现保持一致，避免"代码改了、注释没改"。
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
import os
import sys
import time
import warnings
from types import SimpleNamespace
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ==============================================================================
# 默认仿真配置
# ==============================================================================
# 这里放的是工程主版本的统一参数来源。
# 建议优先在此维护算例参数，避免多处覆盖带来的歧义。
# 单一修改入口：下放完毕(到达目标高度)后继续模拟的额外时长（秒）
EXTRA_SIM_TIME_AFTER_TARGET = 100.0

DEFAULT_CONFIG = {
    # 基本物理参数
    'g': 9.81,
    'rho_w': 1025,
    'N': 2,
    'L': 100,
    
    # 钢丝绳参数
    'E': 2.1e11,
    'd': 0.032,              # 钢丝绳直径 
    'rho_c': 7850,
    'm_b': 500,
    # 缆系参数理论依据：
    # 1. EA_factor=0.5: 根据 API RP 2SK 及 DNVGL 规范，螺旋缠绕钢丝绳的有效轴向刚度(Effective Axial Stiffness)约为名义截面刚度的 40% ~ 60%。取0.5。
    'EA_factor': 0.5,        # 有效刚度折减系数
    # 2. Breaking Load (MBL): 替代原来无根据的 0.2% 应变许用值。对于常用船用 32mm 高强度钢丝绳，DNV 规范推荐 MBL 约为 560 kN。
    'breaking_load': 560000.0,  # [N] 最小破断力(MBL)
    
    # 阻力系数
    'C_b_axial': 0.6,
    'A_b_axial': np.pi * (0.25**2),
    'C_b_radial': 1.0,
    'A_b_radial': 0.375,
    'V_b': 0.5,
    'C_t': 0.02,
    'C_n': 1.0,
    'xi_axial': 0.15,
    'xi_transverse': 0.03,
    'rayleigh_alpha_M': 0.02,

    # 附加质量系数
    'Ca_rope_normal': 1.0,
    'Ca_rope_tangent': 0.05,
    'Ca_payload': 0.5,
    
    # 水深和目标参数
    'total_water_depth': 1500,
    'seabed_depth': -1500,
    'target_height_above_seabed': 25.0,
    
    # 下放参数
    'total_depth': 1500,
    'total_segments': 30,
    'release_speed': 0.6,
    'extra_sim_time_after_target': EXTRA_SIM_TIME_AFTER_TARGET,
    'growth_initial_length': 1.0,        # 工程默认取 1.0 m；可切回更小的历史值

    # 主动张力控制（ATC）参数：通过调整放缆速度抑制顶端过张力
    'active_tension_control_enabled': False, # ATC总开关：默认关闭
    'tension_control_target_scale': 1.08,
    'tension_control_kp': 0.35,
    'tension_control_ki': 0.02,
    'tension_control_deadband_ratio': 0.02,
    'tension_control_speed_min': 0.25,
    'tension_control_speed_max': 1.40,
    'tension_control_rate_limit': 0.003,
    'tension_control_integral_limit': 80.0,
    
    # heave 补偿
    'heave_ratio': 0.35,
    
    # PCH参数
    'pch_upper_cyl_diameter': 0.1,
    'pch_upper_cyl_height': 1.5,
    'pch_cone_upper_diameter': 0.1,
    'pch_cone_lower_diameter': 0.5,
    'pch_cone_height': 0.5,
    'pch_lower_cyl_diameter': 0.5,
    'pch_lower_cyl_height': 0.5,
    'pch_mass': 300.0,
    'pch_Cd_axial': 0.8,
    'pch_Cd_radial': 1.2,
    'pch_Ca': 0.5,
    
    # 工具串参数
    'tool_segment_count': 4,
    'tool_segment_length': 5.0,
    'tool_diameter': 0.1,
    'tool_mass': 400.0,
    'tool_Cd': 1.0,
    'tool_Ca': 1.0,
    
    # 波浪参数
    'wave_height': 1.0,
    'wave_amplitude': 0.5,
    'wave_period': 11.0,
    'wave_direction': 0.0,  # deg，相对船艏入射角；0=迎浪，90=横浪
    'vessel_heading': 0.0,
    'vessel_speed': 0.0,
    'wave_excitation_enabled': True,    # 波浪激励开关（流体波浪 + 船体波浪响应）
    'vessel_rao_enabled': True,         # 船体RAO响应开关
    'wave_compensation_enabled': True,  # 波浪补偿（heave补偿）开关
    'current_model_enabled': True,      # 海流模型开关
    # 海流剖面参数：与论文 2.3 节保持一致，u/v = background + surface_excess * exp(-z / decay_depth)
    'current_u_background': 0.3432,
    'current_u_surface_excess': 0.7354,
    'current_v_background': 0.0686,
    'current_v_surface_excess': 0.1471,
    'current_decay_depth': 242.5,
    # 单位：heave/surge/sway 为 m/m；roll/pitch/yaw 为 deg/m。
    # VesselMotion 会按相对浪向分配纵荡/横荡/横摇/纵摇，避免迎浪时仍产生横摇。
    'heave_RAO': 0.45,
    'roll_RAO': 0.8,
    'pitch_RAO': 0.35,
    'surge_RAO': 0.12,
    'sway_RAO': 0.08,
    'yaw_RAO': 0.15,
    # 默认假定通过中部/近中心线入水点作业；z 为主甲板相对水线的近似高度:
    # depth_to_main_deck - summer_loadline = 10.80 - 7.70 = 3.10 m。
    # 若实际为舷侧/艉部吊放，应改为相对船体重心的真实 x/y/z。
    'winch_offset': [0.0, 0.0, 3.10],

    # 顶端边界条件模式
    # 'rigid'   : 原始刚性强制位移边界（向后兼容）
    # 'filtered': 对顶端heave速度做一阶低通滤波，截止频率远低于波浪频率，消除高频应力冲击
    'top_bc_mode': 'rigid',
    'top_bc_filter_tau': 30.0,      # 低通滤波时间常数(s)，建议 3~5 倍波浪周期（默认30s≈2.7×Tw=11s）

    # 仿真/求解开关
    'verbose': True,
    'solver_profile': 'engineering',     # 'engineering' | 'strict'
    'use_sparse_jacobian': True,
    'event_detection_enabled': True,
    'initialization_mode': 'current_biased',  # 'current_biased' | 'vertical'
    'payload_motion_mode': 'rope_aligned',       # 'rope_aligned' | 'vertical_hanging'
    'summary_safety_factor_basis': 'raw',        # 设计校核固定使用原始峰值；滤波仅用于趋势展示
    # 保持 False：_compute_stress_strain_history 返回真正原始时程，避免污染 raw_* 和设计校核。
    'visualization_smooth_enabled': False,
    # 仅控制 Excel 是否附加 *_Smoothed 列；npz 总是导出 raw_* 与 filtered_* 两套字段。
    'stress_filter_enabled': True,
    # 工程滤波参数：先做约 1.5 s 中值滤波去尖峰，再做零相位低通保留波浪尺度响应。
    'stress_filter_median_window_s': 1.5,
    'stress_filter_cutoff_wave_multiple': 4.0,
    'stress_filter_cutoff_hz': 0.45,
}



class VesselMotion:
    """
    船体六自由度运动模型
    """

    def __init__(self, config=None):
        self.config = DEFAULT_CONFIG.copy()
        if config is not None:
            self.config.update(config)

        # 动力学开关
        self.wave_excitation_enabled = bool(self.config['wave_excitation_enabled'])
        self.vessel_rao_enabled = bool(self.config['vessel_rao_enabled'])

        # 波浪参数
        self.wave_amplitude = float(self.config['wave_amplitude'])  # m
        self.wave_period = float(self.config['wave_period'])  # s
        self.wave_direction = np.deg2rad(float(self.config['wave_direction']))  # rad
        self.omega = 2 * np.pi / self.wave_period

        # 船体参数
        self.vessel_heading = np.deg2rad(float(self.config['vessel_heading']))  # rad
        self.vessel_speed = float(self.config['vessel_speed'])  # m/s

        # RAO
        self.heave_RAO = float(self.config['heave_RAO'])
        self.roll_RAO = np.deg2rad(float(self.config['roll_RAO']))
        self.pitch_RAO = np.deg2rad(float(self.config['pitch_RAO']))
        self.surge_RAO = float(self.config['surge_RAO'])
        self.sway_RAO = float(self.config['sway_RAO'])
        self.yaw_RAO = np.deg2rad(float(self.config['yaw_RAO']))

    def _relative_wave_components(self):
        """返回相对浪向在船体纵向和横向上的响应分量。"""
        # wave_direction 在配置中按相对船艏入射角解释。
        relative_heading = self.wave_direction
        return np.cos(relative_heading), np.sin(relative_heading)

    def get_vessel_position(self, t):
        """船体质心位置"""
        wave_amp = self.wave_amplitude if self.wave_excitation_enabled else 0.0
        if self.vessel_rao_enabled:
            long_comp, trans_comp = self._relative_wave_components()
            phase = self.omega * t
            transverse_phase = phase + np.pi / 2
            surge = self.surge_RAO * wave_amp * long_comp * np.sin(phase) + self.vessel_speed * t
            sway = self.sway_RAO * wave_amp * trans_comp * np.sin(transverse_phase)
            heave = self.heave_RAO * wave_amp * np.sin(self.omega * t)
        else:
            surge = self.vessel_speed * t
            sway = 0.0
            heave = 0.0
        return np.array([surge, sway, heave])

    def get_vessel_velocity(self, t):
        """船体质心速度"""
        wave_amp = self.wave_amplitude if self.wave_excitation_enabled else 0.0
        if self.vessel_rao_enabled:
            long_comp, trans_comp = self._relative_wave_components()
            phase = self.omega * t
            transverse_phase = phase + np.pi / 2
            vx = self.surge_RAO * wave_amp * long_comp * self.omega * np.cos(phase) + self.vessel_speed
            vy = self.sway_RAO * wave_amp * trans_comp * self.omega * np.cos(transverse_phase)
            vz = self.heave_RAO * wave_amp * self.omega * np.cos(self.omega * t)
        else:
            vx = self.vessel_speed
            vy = 0.0
            vz = 0.0
        return np.array([vx, vy, vz])

    def get_vessel_orientation(self, t):
        """船体姿态角"""
        wave_amp = self.wave_amplitude if self.wave_excitation_enabled else 0.0
        if self.vessel_rao_enabled:
            long_comp, trans_comp = self._relative_wave_components()
            phase = self.omega * t
            transverse_phase = phase + np.pi / 2
            roll = self.roll_RAO * wave_amp * trans_comp * np.sin(transverse_phase)
            pitch = self.pitch_RAO * wave_amp * long_comp * np.sin(phase)
            yaw = self.vessel_heading + self.yaw_RAO * wave_amp * trans_comp * np.sin(transverse_phase)
        else:
            roll = 0.0
            pitch = 0.0
            yaw = self.vessel_heading
        return np.array([roll, pitch, yaw])

    def get_vessel_angular_velocity(self, t):
        """船体角速度"""
        wave_amp = self.wave_amplitude if self.wave_excitation_enabled else 0.0
        if self.vessel_rao_enabled:
            long_comp, trans_comp = self._relative_wave_components()
            phase = self.omega * t
            transverse_phase = phase + np.pi / 2
            wx = self.roll_RAO * wave_amp * trans_comp * self.omega * np.cos(transverse_phase)
            wy = self.pitch_RAO * wave_amp * long_comp * self.omega * np.cos(phase)
            wz = self.yaw_RAO * wave_amp * trans_comp * self.omega * np.cos(transverse_phase)
        else:
            wx = 0.0
            wy = 0.0
            wz = 0.0
        return np.array([wx, wy, wz])

    def get_rotation_matrix(self, t):
        """旋转矩阵"""
        roll, pitch, yaw = self.get_vessel_orientation(t)
        r = Rotation.from_euler('zyx', [yaw, pitch, roll])
        return r.as_matrix()


class WireRopeSystem3D:
    """
    三维钢丝绳系统基类。

    该类负责:
    - 读取和归一化配置；
    - 建立钢丝绳、PCH 和工具串的几何/质量参数；
    - 提供力学项、阻力项、质量项和 ODE 求解基础设施；
    - 为动态下放子类提供可覆写的增长段接口。
    """

    def __init__(self, config=None):
        # 使用全局默认配置
        self.config = DEFAULT_CONFIG.copy()
        
        # 用户配置覆盖默认值
        if config is not None:
            self.config.update(config)

        # 应用有效刚度折减
        self.config['E'] *= self.config.get('EA_factor', 1.0)
        self.verbose = bool(self.config.get('verbose', True))

        # 段原始长度数组 - 只用于钢丝绳
        self.l0 = np.full(self.config['N'] - 1, self.config['L'] / (self.config['N'] - 1))
        self._segment_stiffness_reference_lengths = {}
        self.A = np.pi * (self.config['d'] / 2) ** 2
        self.m_c = self.config['rho_c'] * self.A * self.l0[0]

        # 时间偏置
        self.time_offset = 0.0

        # 初始化船体
        self.vessel = VesselMotion(self.config)

        # ===== 重物单元初始化（PCH + 工具串作为整体）=====
        # 重物单元从初始时刻就浸没在水中，不需要通过节点数激活
        self._init_payload_unit()

        # 初始化状态
        self.init_state = self._initialize_system()
        
        # === 修复1：基类属性初始化 ===
        # 增长段信息（基类默认无增长段）
        # 这些属性在 system_equations 中被引用，必须在基类中初始化
        self.growing_segment_idx = None
        self.growth_start_time_abs = None
        self.growth_target_length = self.l0[0] if len(self.l0) > 0 else 1.0

        # 顶端边界条件辅助状态
        # 方案A（filtered）：一阶低通滤波器状态，分别跟踪x/y/z三个分量
        self._top_vel_filtered = None   # 滤波后的顶端速度，首次调用时初始化
        self._top_vel_filter_t = None   # 上次滤波更新的绝对时间
        # 方案B（spring）：节点0作为自由节点，其加速度由弹簧力决定，无需额外状态

    def _log(self, message):
        """统一的过程日志入口，便于通过 `verbose` 开关关闭中间输出。"""
        if self.verbose:
            print(message)

    def _get_solver_settings(self):
        """
        根据 `solver_profile` 返回求解器参数。

        `engineering`:
            当前工程主版本默认设置，优先兼顾稳定性与效率。
        `strict`:
            吸收 `high_fidelity_1_opt.py` 中更严格的容差与更小的步长，
            适合做关键工况复核与数值敏感性检查。
        """
        profile = self.config.get('solver_profile', 'engineering')

        if profile == 'engineering':
            return {
                'primary_method': 'Radau',
                'primary_atol': 1e-4,
                'primary_rtol': 1e-3,
                'primary_max_step': 0.5,
                'fallback_method': 'BDF',
                'fallback_atol': 1e-4,
                'fallback_rtol': 1e-3,
                'fallback_max_step': 0.5,
                'rescue_method': 'LSODA',
                'rescue_atol': 1e-3,
                'rescue_rtol': 1e-2,
                'rescue_max_step': 1.0,
            }

        if profile == 'strict':
            return {
                'primary_method': 'Radau',
                'primary_atol': 1e-6,
                'primary_rtol': 1e-4,
                'primary_max_step': 0.2,
                'fallback_method': 'BDF',
                'fallback_atol': 1e-5,
                'fallback_rtol': 1e-3,
                'fallback_max_step': 0.2,
                'rescue_method': 'LSODA',
                'rescue_atol': 1e-4,
                'rescue_rtol': 1e-2,
                'rescue_max_step': 0.5,
            }

        raise ValueError(f"未知 solver_profile: {profile}")
    
    def _init_payload_unit(self):
        """
        初始化重物单元（PCH + 工具串）
        
        重物单元作为独立单元处理，不参与钢丝绳节点计数。
        
        PCH结构（从上到下，刚性连接为整体）：
        1. 上圆柱：直径0.1m，高1.5m（连接钢丝绳）
        2. 圆台：上直径0.1m，下直径0.5m，高0.5m
        3. 下圆柱：直径0.5m，高0.5m
        
        工具串：4段圆柱，直径0.1m，每段5m，总长20m
        
        重物单元从初始时刻就浸没在水中，始终受到浮力和水动力作用
        """
        cfg = self.config
        
        # ===== PCH几何参数（三段结构）=====
        # 上圆柱
        self.pch_upper_cyl_d = cfg['pch_upper_cyl_diameter']
        self.pch_upper_cyl_h = cfg['pch_upper_cyl_height']
        # 圆台
        self.pch_cone_d_upper = cfg['pch_cone_upper_diameter']
        self.pch_cone_d_lower = cfg['pch_cone_lower_diameter']
        self.pch_cone_h = cfg['pch_cone_height']
        # 下圆柱
        self.pch_lower_cyl_d = cfg['pch_lower_cyl_diameter']
        self.pch_lower_cyl_h = cfg['pch_lower_cyl_height']
        
        # PCH总高度
        self.pch_height = self.pch_upper_cyl_h + self.pch_cone_h + self.pch_lower_cyl_h
        
        # PCH物理参数
        self.pch_mass = cfg['pch_mass']
        self.pch_Cd_axial = cfg['pch_Cd_axial']
        self.pch_Cd_radial = cfg['pch_Cd_radial']
        self.pch_Ca = cfg['pch_Ca']
        
        # ===== 计算PCH各部分体积 =====
        # 上圆柱体积
        r_upper = self.pch_upper_cyl_d / 2
        self.pch_upper_cyl_volume = np.pi * r_upper**2 * self.pch_upper_cyl_h
        
        # 圆台体积 V = (1/3) * π * h * (r1² + r1*r2 + r2²)
        r_cone_upper = self.pch_cone_d_upper / 2
        r_cone_lower = self.pch_cone_d_lower / 2
        self.pch_cone_volume = (1/3) * np.pi * self.pch_cone_h * (
            r_cone_upper**2 + r_cone_upper * r_cone_lower + r_cone_lower**2
        )
        
        # 下圆柱体积
        r_lower = self.pch_lower_cyl_d / 2
        self.pch_lower_cyl_volume = np.pi * r_lower**2 * self.pch_lower_cyl_h
        
        # PCH总体积
        self.pch_volume = self.pch_upper_cyl_volume + self.pch_cone_volume + self.pch_lower_cyl_volume
        
        # ===== 计算PCH各部分投影面积（用于阻力计算）=====
        # 轴向投影面积（向下运动时的迎流面积 = 下圆柱底面）
        self.pch_A_axial_down = np.pi * r_lower**2
        # 轴向投影面积（向上运动时的迎流面积 = 上圆柱顶面）
        self.pch_A_axial_up = np.pi * r_upper**2
        
        # 径向投影面积（侧面投影）
        # 上圆柱侧面投影
        A_radial_upper = self.pch_upper_cyl_d * self.pch_upper_cyl_h
        # 圆台侧面投影（梯形）
        A_radial_cone = (self.pch_cone_d_upper + self.pch_cone_d_lower) / 2 * self.pch_cone_h
        # 下圆柱侧面投影
        A_radial_lower = self.pch_lower_cyl_d * self.pch_lower_cyl_h
        self.pch_A_radial = A_radial_upper + A_radial_cone + A_radial_lower
        
        # ===== 工具串几何和物理参数 =====
        self.tool_segment_count = cfg['tool_segment_count']
        self.tool_segment_length = cfg['tool_segment_length']
        self.tool_diameter = cfg['tool_diameter']
        self.tool_length = self.tool_segment_count * self.tool_segment_length  # 总长度
        self.tool_mass = cfg['tool_mass']
        self.tool_Cd = cfg['tool_Cd']
        self.tool_Ca = cfg['tool_Ca']
        
        # 工具串排水体积
        self.tool_volume = np.pi * (self.tool_diameter / 2)**2 * self.tool_length
        
        # 工具串投影面积
        self.tool_A_axial = np.pi * (self.tool_diameter / 2)**2  # 端面
        self.tool_A_radial = self.tool_diameter * self.tool_length  # 侧面
        
        # ===== 重物单元总参数 =====
        self.payload_total_mass = self.pch_mass + self.tool_mass
        self.payload_total_volume = self.pch_volume + self.tool_volume
        self.payload_total_length = self.pch_height + self.tool_length
        
        # 初始化信息只在 verbose 模式下打印，避免批量参数扫描时刷屏。
        self._log("重物单元初始化完成（细化PCH模型）:")
        self._log("  PCH结构（从上到下）:")
        self._log(
            f"    上圆柱: d={self.pch_upper_cyl_d}m, h={self.pch_upper_cyl_h}m, "
            f"V={self.pch_upper_cyl_volume:.4f}m^3"
        )
        self._log(
            f"    圆台:   d_up={self.pch_cone_d_upper}m, d_low={self.pch_cone_d_lower}m, "
            f"h={self.pch_cone_h}m, V={self.pch_cone_volume:.4f}m^3"
        )
        self._log(
            f"    下圆柱: d={self.pch_lower_cyl_d}m, h={self.pch_lower_cyl_h}m, "
            f"V={self.pch_lower_cyl_volume:.4f}m^3"
        )
        self._log(
            f"    PCH总计: 质量={self.pch_mass}kg, 体积={self.pch_volume:.4f}m^3, "
            f"高度={self.pch_height}m"
        )
        self._log(f"  工具串: {self.tool_segment_count}段x{self.tool_segment_length}m, d={self.tool_diameter}m")
        self._log(
            f"    工具串总计: 质量={self.tool_mass}kg, 体积={self.tool_volume:.4f}m^3, "
            f"长度={self.tool_length}m"
        )
        self._log(
            f"  重物单元总计: 质量={self.payload_total_mass}kg, "
            f"体积={self.payload_total_volume:.4f}m^3, 长度={self.payload_total_length}m"
        )

    def _initialize_system(self):
        """
        初始化系统状态。

        该方法支持两种初始构型:
        - `current_biased`:
          工程主版本默认。根据海流给出近似偏移和初始速度，减少起算瞬态。
        - `vertical`:
          历史版本/对照版本常用。钢丝绳初始为竖直直线，便于做基线比较。
        """
        N = self.config['N']
        r = np.zeros((N, 3))
        v = np.zeros((N, 3))
        dz = self.l0[0]
        r[0] = self.top_position(0)

        init_mode = self.config.get('initialization_mode', 'current_biased')
        if init_mode == 'vertical':
            for i in range(1, N):
                r[i] = r[0] + np.array([0.0, 0.0, -i * dz])
            return np.concatenate([r.flatten(), v.flatten()])

        if init_mode != 'current_biased':
            raise ValueError(f"未知 initialization_mode: {init_mode}")

        # 估算海流引起的水平偏移。
        # 这是工程初始化技巧，不代表严格静力解，只用于减少刚起算时的数值冲击。
        for i in range(1, N):
            depth = i * dz
            z_pos = r[0, 2] - depth

            current = self.current_velocity(np.array([0, 0, z_pos]))

            # 偏移量使用经验系数控制，目的是给出合理初值，而非替代稳态求解。
            offset_factor = 0.02
            cumulative_offset_x = offset_factor * current[0] * depth
            cumulative_offset_y = offset_factor * current[1] * depth

            r[i] = r[0] + np.array([cumulative_offset_x, cumulative_offset_y, -depth])

            # 让初始速度部分贴近海流，进一步减弱启动瞬态。
            v[i] = 0.5 * current
        
        return np.concatenate([r.flatten(), v.flatten()])

    def fluid_kinematics(self, pos, t_abs):
        """
        综合流体运动学 (海流 + 线性 Airy 波浪)
        返回: (u_fluid, a_fluid)
        u_fluid: 3D流体速度
        a_fluid: 3D流体加速度
        """
        # 稳态海流
        u_current = self.current_velocity(pos)

        if not bool(self.config.get('wave_excitation_enabled', True)):
            return u_current, np.zeros(3)
        
        # 波浪运动学 (Linear Airy Wave)
        z = pos[2]  # z <= 0 (表面为0，向下为负)
        H = self.config.get('total_water_depth', 1500)
        
        if z > 0 or z < -H:
            return u_current, np.zeros(3)
            
        A = self.vessel.wave_amplitude
        omega = self.vessel.omega
        theta = self.vessel.wave_direction
        g = self.config.get('g', 9.81)
        
        # 求解波数 k (深水近似 k = omega^2 / g)
        k = omega**2 / g
        
        # 相位 Phi = k(x*cos(theta) + y*sin(theta)) - omega*t
        # 这里为了简化，可以用全深度的精确系数:
        # 为了防止 sinh(kH) 溢出（深水区 kH 很大），可以做近似替换
        # cosh(k(z+H)) / sinh(kH)  exp(kz)
        # sinh(k(z+H)) / sinh(kH)  exp(kz)
        # 这是一个极好的且无溢出风险的深水近似（当 kH > 3 时足够精确）
        
        # 对 kH 很小的情况我们用精确的 tanh
        if k * H < 3.0:
            # 简单固定点迭代求 k (k = omega^2 / (g * tanh(kH)))
            for _ in range(5):
                k = omega**2 / (g * np.tanh(k * H))
            cosh_kH = np.cosh(k * H)
            sinh_kH = np.sinh(k * H)
            f_x = np.cosh(k * (z + H)) / sinh_kH
            f_z = np.sinh(k * (z + H)) / sinh_kH
        else:
            f_x = np.exp(k * z)
            f_z = np.exp(k * z)
            
        Phi = k * (pos[0] * np.cos(theta) + pos[1] * np.sin(theta)) - omega * t_abs
        
        # 速度
        u_wave = A * omega * f_x * np.cos(theta) * np.cos(Phi)
        v_wave = A * omega * f_x * np.sin(theta) * np.cos(Phi)
        w_wave = A * omega * f_z * np.sin(Phi)
        
        # 加速度
        ax_wave = A * omega**2 * f_x * np.cos(theta) * np.sin(Phi)
        ay_wave = A * omega**2 * f_x * np.sin(theta) * np.sin(Phi)
        az_wave = -A * omega**2 * f_z * np.cos(Phi)
        
        u_fluid = u_current + np.array([u_wave, v_wave, w_wave])
        a_fluid = np.array([ax_wave, ay_wave, az_wave])
        
        return u_fluid, a_fluid

    def current_velocity(self, pos):
        """三维海流速度：使用论文 2.3 节的指数衰减剖面。"""
        if not bool(self.config.get('current_model_enabled', True)):
            return np.array([0.0, 0.0, 0.0])

        z = -pos[2]  # 深度
        H = self.config['total_water_depth']
        if z < 0 or z > H:
            return np.array([0.0, 0.0, 0.0])
        decay_depth = float(self.config.get('current_decay_depth', 242.5))
        u = (
            float(self.config.get('current_u_background', 0.536))
            + float(self.config.get('current_u_surface_excess', 0.858)) * np.exp(-z / decay_depth)
        )
        v = (
            float(self.config.get('current_v_background', 0.107))
            + float(self.config.get('current_v_surface_excess', 0.172)) * np.exp(-z / decay_depth)
        )
        w = 0.0
        return np.array([u, v, w])

    def _compute_segment_vectors(self, r):
        """计算段向量"""
        N = r.shape[0]
        l = np.zeros((N - 1, 3))
        l_unit = np.zeros((N - 1, 3))
        l_norm = np.zeros(N - 1)
        for j in range(N - 1):
            l[j] = r[j + 1] - r[j]
            l_norm[j] = np.linalg.norm(l[j])
            l_unit[j] = l[j] / l_norm[j] if l_norm[j] > 1e-12 else np.array([0.0, 0.0, -1.0])
        return l, l_unit, l_norm

    def effective_rest_length(self, j, t_abs):
        """
        返回第j段在t_abs时刻的等效原始长度
        基类：恒等于self.l0[j]
        子类会覆写此方法实现动态增长
        """
        return self.l0[min(j, len(self.l0) - 1)]

    def _is_growing_segment(self, j):
        """判断索引 j 是否属于当前增长段（兼容单值和列表）。"""
        idx = getattr(self, 'growing_segment_idx', None)
        if idx is None:
            return False
        if isinstance(idx, (list, tuple, set, np.ndarray)):
            return j in idx
        return j == idx

    def _segment_stiffness_reference_length(self, j, l0_eff):
        """
        返回用于轴向刚度 EA/L 的参考长度。

        普通段使用自身原长；增长段和目标高度中断后的冻结短段使用完整目标段长，
        从而避免短段因 L 过小产生非物理刚度放大。
        """
        ref_lengths = getattr(self, '_segment_stiffness_reference_lengths', {})
        ref_len = ref_lengths.get(int(j))
        if ref_len is not None:
            return max(float(ref_len), 1e-6)

        if self._is_growing_segment(j):
            return max(float(getattr(self, 'growth_target_length', l0_eff)), 1e-6)

        return max(float(l0_eff), 1e-6)

    def _segment_effective_modulus(self, j, l0_eff):
        """按参考长度缩放 E，使短段/增长段保持目标轴向刚度 EA/L_ref。"""
        E_nominal = self.config.get('E', 1e11)
        ref_len = self._segment_stiffness_reference_length(j, l0_eff)
        scale_factor = max(float(l0_eff) / ref_len, 1e-4)
        return E_nominal * scale_factor

    def _compute_tension(self, l, l_unit, l_norm, t_abs):
        N = l.shape[0] + 1
        T = np.zeros((N - 1, 3))

        for j in range(N - 1):
            l0_eff = self.effective_rest_length(j, t_abs)
            denom = max(l0_eff, 1e-6)
            strain = (l_norm[j] - l0_eff) / denom

            if strain > 0:
                E_current = self._segment_effective_modulus(j, l0_eff)
                T[j] = E_current * self.A * strain * l_unit[j]
        return T

    def _compute_strain(self, l_norm, t_abs):
        """计算应变"""
        N = l_norm.shape[0] + 1
        strain = np.zeros(N - 1)
        for j in range(N - 1):
            l0_eff = self.effective_rest_length(j, t_abs)
            strain[j] = (l_norm[j] - l0_eff) / max(l0_eff, 1e-12)
        return strain

    def compute_stress_from_tension(self, T, l_unit=None):
        """
        === 修复2：统一应力计算方法 ===
        
        从张力计算应力（推荐方法）
        
        应力 σ = |T| / A
        
        这是最准确的方法，因为张力已经考虑了所有刚度修正。
        无论增长段的 E 如何动态调整，张力都是真实的物理量，
        因此 T/A 给出的应力是物理上正确的。
        
        参数:
            T: 张力向量数组 (N-1, 3)
            l_unit: 段单位向量（可选，用于计算轴向应力分量）
        
        返回:
            stress: 应力数组 (N-1,) 单位 Pa
        """
        if T.size == 0:
            return np.array([])
        
        Tmag = np.linalg.norm(T, axis=1)
        stress = Tmag / self.A
        return stress

    def _compute_weight(self, t_abs):
        """
        计算重力 - 钢丝绳 + 重物单元（PCH + 工具串）
        
        重物单元从初始时刻就浸没在水中，始终受到浮力作用。
        钢丝绳使用等效长度，让新段重力随下放逐渐增长。
        
        参数:
            t_abs: 绝对时间，用于计算增长段的等效长度
        """
        N = self.config['N']
        W = np.zeros((N, 3))
        g = self.config['g']
        rho_w = self.config['rho_w']
        
        # ===== 钢丝绳重力（使用等效长度）=====
        for j in range(N - 1):
            l_eff = self.effective_rest_length(j, t_abs)
            seg_volume = self.A * l_eff
            seg_w = g * (self.config['rho_c'] - rho_w) * seg_volume
            W[j, 2] -= 0.5 * seg_w
            W[j + 1, 2] -= 0.5 * seg_w
        
        # ===== 重物单元重力（PCH + 工具串）=====
        # 重物单元从初始时刻就浸没在水中，始终受到浮力
        # 湿重 = 干重 - 浮力 = m*g - ρ_w*V*g
        
        # PCH湿重
        pch_wet_weight = self.pch_mass * g - rho_w * self.pch_volume * g
        
        # 工具串湿重
        tool_wet_weight = self.tool_mass * g - rho_w * self.tool_volume * g
        
        # 总重物单元湿重，作用在钢丝绳末端节点
        payload_wet_weight = pch_wet_weight + tool_wet_weight
        W[N - 1, 2] -= payload_wet_weight
        
        return W

    def _compute_drag(self, r, v, t_abs):
        """
        计算阻力 - 钢丝绳 + 重物单元（PCH + 工具串）
        
        === 核心修复：从"节点法"改为"段单元法" (Lumped Mass Segment approach) ===
        
        改进说明：
        1. 在段的中点采样流速，而不是在节点处
        2. 将段阻力平均分配给两端节点
        3. 对离散化（N的变化）不敏感，显著提高收敛性
        4. 解决了 N 变化时因海流采样位置不同导致的计算误差
        
        结构说明：
        - 钢丝绳：节点 0 到 N-1，使用钢丝绳直径计算阻力
        - 重物单元（PCH + 工具串）：作为整体连接在钢丝绳末端节点 N-1
          重物单元从初始时刻就浸没在水中，始终受到水动力作用
        
        节点数只用于钢丝绳离散化，不与重物单元混淆
        """
        N = r.shape[0]
        D = np.zeros((N, 3))
        rho_w = self.config['rho_w']
        
        # ===== 钢丝绳阻力（段单元法）=====
        # 遍历每一段，在段中心计算阻力，然后分配给两端节点
        for j in range(N - 1):
            # 获取段两端节点状态
            r1, r2 = r[j], r[j + 1]
            v1, v2 = v[j], v[j + 1]
            
            # 计算段中心位置和速度
            r_mid = 0.5 * (r1 + r2)
            v_mid = 0.5 * (v1 + v2)
            
            # 段向量和长度
            seg_vec = r2 - r1
            seg_len_geo = np.linalg.norm(seg_vec)
            if seg_len_geo < 1e-6:
                continue
            tau = seg_vec / seg_len_geo  # 切向单位向量

            # === 核心修复：流动力积分项应全部使用无拉力恒定状态的原长 ===
            seg_len = self.effective_rest_length(j, t_abs)
            
            # 在段中心采样海流速度
            c_mid, a_mid = self.fluid_kinematics(r_mid, t_abs)
            
            # 计算 Morison 惯性力项 (F = rho * V * Cm * a_fluid)
            a_tangent = np.dot(a_mid, tau) * tau
            a_normal = a_mid - a_tangent
            Cm_n = 1.0 + self.config.get('Ca_rope_normal', 1.0)
            Cm_t = 1.0 + self.config.get('Ca_rope_tangent', 0.05)
            f_inertia = rho_w * self.A * seg_len * (Cm_n * a_normal + Cm_t * a_tangent)
            
            # 相对速度
            v_rel = v_mid - c_mid
            
            # 分解速度为切向和法向分量
            v_t_mag = np.dot(v_rel, tau)
            v_t_vec = v_t_mag * tau
            v_n_vec = v_rel - v_t_vec
            v_n_mag = np.linalg.norm(v_n_vec)
            
            # 钢丝绳直径
            d_eff = self.config['d']
            
            # 计算阻力 (Morison Equation drag term)
            # 切向阻力: 0.5 * rho * Ct * (pi*d) * L * |vt| * vt
            F_drag_t = -0.5 * rho_w * self.config['C_t'] * (np.pi * d_eff) * seg_len * np.abs(v_t_mag) * v_t_vec
            
            # 法向阻力: 0.5 * rho * Cn * d * L * |vn| * vn
            F_drag_n = -0.5 * rho_w * self.config['C_n'] * d_eff * seg_len * v_n_mag * v_n_vec
            
            F_segment_total = F_drag_t + F_drag_n
            
            # 将段阻力平均分配给两端节点 (Lumped Mass approach)
            D[j] += 0.5 * F_segment_total
            D[j + 1] += 0.5 * F_segment_total

        # ===== 重物单元阻力（PCH + 工具串），作用在末端节点 =====
        # 重物单元从初始时刻就浸没在水中，始终受到水动力
        # 计算末端切向方向
        if N >= 2:
            last_seg = r[N - 1] - r[N - 2]
            last_seg_norm = np.linalg.norm(last_seg)
            tau_last = last_seg / last_seg_norm if last_seg_norm > 1e-12 else np.array([0.0, 0.0, -1.0])
        else:
            tau_last = np.array([0.0, 0.0, -1.0])
        
        payload_drag = self._compute_payload_drag(r[N - 1], v[N - 1], tau_last, t_abs)
        D[N - 1] += payload_drag

        return D
    
    def _compute_payload_drag(self, pos, vel, tau_dir, t_abs):
        """[修复版] 计算重物单元阻力
        
        修复了轴向运动方向判断的逻辑反转问题
        """
        rho_w = self.config['rho_w']
        c, a_fluid = self.fluid_kinematics(pos, t_abs)
        u_rel = vel - c
        
        # 分解相对速度为轴向（沿tau_dir）和径向分量
        u_axial_mag = np.dot(u_rel, tau_dir)
        u_axial = u_axial_mag * tau_dir
        u_radial = u_rel - u_axial
        u_axial_norm = np.linalg.norm(u_axial)
        u_radial_norm = np.linalg.norm(u_radial)
        
        drag = np.zeros(3)
        
        # ===== PCH阻力（细化三段结构）=====
        # 轴向阻力：根据运动方向选择迎流面积
        if u_axial_norm > 1e-12:
            # tau_dir 是从上指向下的向量 (r[N-1] - r[N-2])
            # 向下运动时，速度与 tau_dir 同向，u_axial_mag > 0
            # 向上运动时，速度与 tau_dir 反向，u_axial_mag < 0
            if u_axial_mag > 0:  # [修复] 大于0才是向下运动（下圆柱底面迎流）
                A_eff = self.pch_A_axial_down  # 下圆柱底面 π*(0.25)²
                Cd_eff = self.pch_Cd_axial
            else:  # 向上运动（上圆柱顶面迎流）
                A_eff = self.pch_A_axial_up  # 上圆柱顶面 π*(0.05)²
                Cd_eff = 0.8  # 圆柱端面阻力系数
            drag += -0.5 * rho_w * Cd_eff * A_eff * u_axial_norm * u_axial
        
        # 径向阻力：使用总侧面投影面积
        if u_radial_norm > 1e-12:
            drag += -0.5 * rho_w * self.pch_Cd_radial * self.pch_A_radial * u_radial_norm * u_radial
        
        # ===== 工具串阻力（4段圆柱）=====
        # 轴向阻力（端面）
        if u_axial_norm > 1e-12:
            Cd_cyl_end = 0.8  # 圆柱端面阻力系数
            drag += -0.5 * rho_w * Cd_cyl_end * self.tool_A_axial * u_axial_norm * u_axial
        
        # 径向阻力（侧面）
        if u_radial_norm > 1e-12:
            drag += -0.5 * rho_w * self.tool_Cd * self.tool_A_radial * u_radial_norm * u_radial

        # 计算重物单元的 Morison 惯性力项 (F = rho * V * Cm * a_fluid)
        # PCH 和 工具串 均为圆柱体形态，这里使用简化版标量质量系数
        Cm_pch = 1.0 + self.pch_Ca
        Cm_tool = 1.0 + self.tool_Ca
        f_inertia_pch = rho_w * self.pch_volume * Cm_pch * a_fluid
        f_inertia_tool = rho_w * self.tool_volume * Cm_tool * a_fluid
        
        drag += f_inertia_pch + f_inertia_tool

        return drag

    def _compute_mass(self, t_abs):
        """
        计算质量 - 钢丝绳 + 重物单元（PCH + 工具串）
        
        钢丝绳使用几何长度（而非有效长度）计算质量，确保质量与实际物理长度一致。
        最小质量与目标段长成比例，避免惯性效应失真。
        
        参数:
            t_abs: 绝对时间
        """
        N = self.config['N']
        M = np.zeros(N)
        
        # 单位长度质量
        mass_per_length = self.config['rho_c'] * self.A
        
        # ===== 钢丝绳质量（使用几何长度）=====
        for j in range(N - 1):
            # 使用有效长度（与力学计算一致）
            l_eff = self.effective_rest_length(j, t_abs)

            # 计算准确受力质量：切分前后总质量应严格守恒，移除失真的min_mass膨胀
            effective_seg_mass = mass_per_length * l_eff
            M[j] += 0.5 * effective_seg_mass
            M[j + 1] += 0.5 * effective_seg_mass
        
        # ===== 重物单元质量（PCH + 工具串）=====
        M[N - 1] += self.payload_total_mass
        
        return M

    def _compute_added_mass(self, r, v, t_abs):
        """
        计算附加质量 - 钢丝绳 + 重物单元（PCH + 工具串）

        重物单元从初始时刻就浸没在水中，始终有附加质量效应。
        """
        N = r.shape[0]
        M_added = np.zeros(N)
        l, l_unit, l_norm = self._compute_segment_vectors(r)
        rho_w = self.config['rho_w']

        # ===== 钢丝绳附加质量 =====
        for j in range(N):
            if j == 0:
                V_eff = self.A * self.effective_rest_length(0, t_abs) / 2
            elif j == N - 1:
                V_eff = self.A * self.effective_rest_length(j - 1, t_abs) / 2
            else:
                V_eff = self.A * (self.effective_rest_length(j - 1, t_abs) + self.effective_rest_length(j, t_abs)) / 2

            Ca_rope = self.config['Ca_rope_normal']
            M_added[j] = Ca_rope * rho_w * V_eff

        # ===== 重物单元附加质量（PCH + 工具串）=====
        # 重物单元从初始时刻就浸没在水中，始终有附加质量
        # PCH附加质量
        M_added_pch = self.pch_Ca * rho_w * self.pch_volume
        # 工具串附加质量
        M_added_tool = self.tool_Ca * rho_w * self.tool_volume
        # 总附加质量作用在末端节点
        M_added[N - 1] += M_added_pch + M_added_tool

        return M_added

    def top_position_abs(self, t_abs):
        """绝对时间下的顶端位置（由船体运动决定，带 heave 补偿）。"""
        vessel_pos = self.vessel.get_vessel_position(t_abs)
        R = self.vessel.get_rotation_matrix(t_abs)
        winch_offset = np.array(self.config.get('winch_offset', [0.0, 0.0, 15.0]))
        
        # heave 补偿比例 (0=完全补偿, 1=不补偿)
        if bool(self.config.get('wave_compensation_enabled', True)):
            heave_ratio = self.config.get('heave_ratio', 0.3)
        else:
            heave_ratio = 1.0
        
        # 分离heave和水平运动
        heave_only = np.array([0.0, 0.0, vessel_pos[2]])
        horiz_only = vessel_pos - heave_only
        
        # 补偿后的位置
        compensated_pos = horiz_only + heave_ratio * heave_only
        return compensated_pos + R @ winch_offset

    def top_position(self, t):
        """局部求解时间下的顶端位置。"""
        return self.top_position_abs(self.time_offset + t)

    def top_velocity_abs(self, t_abs):
        """绝对时间下的顶端速度（带 heave 补偿）。"""
        vessel_vel = self.vessel.get_vessel_velocity(t_abs)
        omega = self.vessel.get_vessel_angular_velocity(t_abs)
        R = self.vessel.get_rotation_matrix(t_abs)
        winch_offset = np.array(self.config.get('winch_offset', [0.0, 0.0, 15.0]))
        
        # heave 补偿比例
        if bool(self.config.get('wave_compensation_enabled', True)):
            heave_ratio = self.config.get('heave_ratio', 0.3)
        else:
            heave_ratio = 1.0
        
        # 分离heave速度和水平速度
        heave_vel_only = np.array([0.0, 0.0, vessel_vel[2]])
        horiz_vel_only = vessel_vel - heave_vel_only
        
        # 补偿后的速度
        compensated_vel = horiz_vel_only + heave_ratio * heave_vel_only
        return compensated_vel + np.cross(omega, R @ winch_offset)

    def top_velocity(self, t):
        """局部求解时间下的顶端速度。"""
        return self.top_velocity_abs(self.time_offset + t)

    def top_velocity_filtered(self, t):
        """
        方案A：对顶端速度的heave分量做一阶低通滤波。
        截止频率 = 1/(2π·tau)，tau >> 波浪周期，滤掉高频heave冲击。
        水平分量（surge/sway）不滤波，保留流致偏移的低频响应。
        """
        t_abs = self.time_offset + t
        tau = float(self.config.get('top_bc_filter_tau', 30.0))
        v_raw = self.top_velocity_abs(t_abs)

        if self._top_vel_filtered is None:
            # 首次调用：用原始速度初始化滤波器
            self._top_vel_filtered = v_raw.copy()
            self._top_vel_filter_t = t_abs
            return v_raw.copy()

        dt = t_abs - self._top_vel_filter_t
        if dt <= 0.0:
            return self._top_vel_filtered.copy()

        alpha = dt / (tau + dt)  # 一阶低通离散系数

        # 只对z分量（heave）滤波，x/y保持原始值
        v_filt = self._top_vel_filtered.copy()
        v_filt[0] = v_raw[0]                                      # surge：不滤波
        v_filt[1] = v_raw[1]                                      # sway：不滤波
        v_filt[2] = (1.0 - alpha) * self._top_vel_filtered[2] + alpha * v_raw[2]  # heave：低通

        self._top_vel_filtered = v_filt.copy()
        self._top_vel_filter_t = t_abs
        return v_filt

    def system_equations(self, t, state):
        """系统运动方程"""
        N = self.config['N']
        r = state[:3 * N].reshape(N, 3)
        v = state[3 * N:].reshape(N, 3)

        t_abs = self.time_offset + t
        top_bc_mode = self.config.get('top_bc_mode', 'rigid')

        if top_bc_mode == 'filtered':
            # 位置强制跟随绞车点，速度heave分量低通滤波，消除高频应力冲击
            r[0] = self.top_position(t)
            v[0] = self.top_velocity_filtered(t)
        else:
            # 'rigid'（默认）：原始刚性强制位移边界，向后兼容
            r[0] = self.top_position(t)
            v[0] = self.top_velocity(t)

        # 计算力
        l, l_unit, l_norm = self._compute_segment_vectors(r)

        # 更新缓存的几何长度（供子类effective_rest_length使用）
        if hasattr(self, '_cached_l_norm'):
            self._cached_l_norm = l_norm.copy()

        T = self._compute_tension(l, l_unit, l_norm, t_abs)
        W = self._compute_weight(t_abs)  # 传入t_abs，让新段重力逐渐增长
        D = self._compute_drag(r, v, t_abs)
        M = self._compute_mass(t_abs)    # 传入t_abs，让新段质量逐渐增长
        M_added = self._compute_added_mass(r, v, t_abs)
        M_total = M + M_added

        
        F_damp = np.zeros((N, 3))
        
        xi_axial = self.config.get('xi_axial', 0.005)
        for j in range(N - 1):
            l0_j = self.effective_rest_length(j, t_abs)
            is_growing = self._is_growing_segment(j)

            # 阻尼刚度与张力计算保持一致；冻结短段不会因 l0 较短而放大 EA/l0。
            # 增长段质量仍用实际已放出长度，保证临界阻尼比随质量连续变化。
            E_current = self._segment_effective_modulus(j, l0_j)

            k_seg = E_current * self.A / max(l0_j, 1e-6)
            L_nom = getattr(self, 'segment_length', 100.0)
            m_nom = self.config.get('rho_c', 7850) * self.A * L_nom
            
            if is_growing:
                m_seg = self.config.get('rho_c', 7850) * self.A * max(l0_j, 1e-6)
            else:
                m_seg = m_nom
                
            c_crit = 2.0 * np.sqrt(k_seg * m_seg)
            c_seg = xi_axial * c_crit
            c_seg = min(c_seg, 1e6)

            dv = v[j + 1] - v[j]
            dv_par = np.dot(dv, l_unit[j])
            
            f_damp_mag = c_seg * dv_par
            f_damp_vec = f_damp_mag * l_unit[j]
            
            # 添加横向阻尼(Transverse Damping)消除非物理高频横波
            # 钢丝绳几乎没有抗弯刚度，因此其横波会无限震荡。引入微小的横向数值阻尼。
            xi_transverse = self.config.get('xi_transverse', 0.01)
            dv_perp = dv - dv_par * l_unit[j]
            # 横向刚度通常由轴向张力提供，这里近似用同一临界阻尼按比例缩减
            c_seg_trans = xi_transverse * c_crit
            c_seg_trans = min(c_seg_trans, 1e5)
            f_damp_vec += c_seg_trans * dv_perp

            F_damp[j] += f_damp_vec
            F_damp[j + 1] -= f_damp_vec

        # Rayleigh 质量比例阻尼：F = -alpha_M * M * v
        # 对全局低频模态提供阻尼（相对速度阻尼对此无效）
        alpha_M = self.config.get('rayleigh_alpha_M', 0.0)
        F_rayleigh = np.zeros((N, 3))
        if alpha_M > 0:
            for j in range(1, N):
                F_rayleigh[j] = -alpha_M * M_total[j] * v[j]

        F = np.zeros((N, 3))
        for j in range(1, N):
            F[j] = D[j] + W[j] + F_damp[j] + F_rayleigh[j]
            if j < N - 1:
                F[j] += T[j]
            if j > 0:
                F[j] -= T[j - 1]

        a = np.zeros((N, 3))
        for j in range(1, N):
            a[j] = F[j] / max(M_total[j], 1.0)

        dr = v.flatten()
        dv = a.flatten()
        dr[0:3] = 0.0
        dv[0:3] = 0.0
        return np.concatenate([dr, dv])

    def _get_sparsity_matrix(self):
        """
        生成雅可比矩阵的稀疏结构 pattern。
        这告诉求解器哪些变量之间存在关联，能带来 10-100 倍的提速。
        
        钢丝绳是链式结构，节点i只和i-1和i+1发生力的作用。
        """
        import scipy.sparse as sp
        N = self.config['N']
        dim = 6 * N
        
        rows = []
        cols = []

        # 状态向量 y = [r0...rN-1, v0...vN-1]
        # 索引映射: r_i -> i*3 : i*3+3
        #           v_i -> 3N + i*3 : 3N + i*3+3

        for i in range(N):
            # === 1. 位置导数 dr/dt = v ===
            # dr_i/dt 仅依赖于 v_i
            # Row: [3*i : 3*i+3], Col: [3*N + 3*i : 3*N + 3*i+3]
            for r in range(3*i, 3*i+3):
                c = 3*N + r
                rows.append(r)
                cols.append(c)

            # === 2. 速度导数 dv/dt = F/m ===
            # F_i 依赖于 (r_{i-1}, r_i, r_{i+1}) 和 (v_{i-1}, v_i, v_{i+1})     

            # 2.1 自身依赖 (i)
            row_start = 3*N + 3*i
            for r in range(row_start, row_start + 3):
                # 依赖 r_i
                for c in range(3*i, 3*i+3):
                    rows.append(r)
                    cols.append(c)
                # 依赖 v_i
                for c in range(3*N + 3*i, 3*N + 3*i+3):
                    rows.append(r)
                    cols.append(c)

            # 2.2 前节点依赖 (i-1)
            if i > 0:
                for r in range(row_start, row_start + 3):
                    # 依赖 r_{i-1}
                    for c in range(3*(i-1), 3*(i-1)+3):
                        rows.append(r)
                        cols.append(c)
                    # 依赖 v_{i-1}
                    for c in range(3*N + 3*(i-1), 3*N + 3*(i-1)+3):
                        rows.append(r)
                        cols.append(c)

            # 2.3 后节点依赖 (i+1)
            if i < N - 1:
                for r in range(row_start, row_start + 3):
                    # 依赖 r_{i+1}
                    for c in range(3*(i+1), 3*(i+1)+3):
                        rows.append(r)
                        cols.append(c)
                    # 依赖 v_{i+1}
                    for c in range(3*N + 3*(i+1), 3*N + 3*(i+1)+3):
                        rows.append(r)
                        cols.append(c)

        data = np.ones(len(rows), dtype=int)
        # 用 coo_matrix 构建更高效，然后再转为一个求解器好用的格式
        return sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()

    def _solve_ode(self, t_span, t_eval=None, events=None):
        """
        统一的 ODE 求解入口。

        这样 `solve` 和 `solve_with_event` 可以共用同一套求解参数，
        避免工程版和带事件版因为参数漂移而出现"同模型、不同设置"的问题。        
        """
        settings = self._get_solver_settings()
        start_time = time.time()
        method_used = settings['primary_method']
        use_sparse_jacobian = bool(self.config.get('use_sparse_jacobian', True))
        jac_sparsity = self._get_sparsity_matrix() if use_sparse_jacobian else None

        def _run(method, atol, rtol, max_step):
            kwargs = {
                'method': method,
                't_eval': t_eval,
                'atol': atol,
                'rtol': rtol,
                'max_step': max_step,
            }
            if events is not None:
                kwargs['events'] = events
            if jac_sparsity is not None and method in {'Radau', 'BDF'}:
                kwargs['jac_sparsity'] = jac_sparsity
            return solve_ivp(self.system_equations, t_span, self.init_state, **kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            try:
                sol = _run(
                    settings['primary_method'],
                    settings['primary_atol'],
                    settings['primary_rtol'],
                    settings['primary_max_step'],
                )
                if not sol.success:
                    method_used = settings['fallback_method']
                    sol = _run(
                        settings['fallback_method'],
                        settings['fallback_atol'],
                        settings['fallback_rtol'],
                        settings['fallback_max_step'],
                    )
            except Exception:
                method_used = settings['rescue_method']
                sol = _run(
                    settings['rescue_method'],
                    settings['rescue_atol'],
                    settings['rescue_rtol'],
                    settings['rescue_max_step'],
                )

        sol.solve_time = time.time() - start_time
        sol.method_used = method_used
        return sol

    def solve(self, t_span, t_eval=None):
        """
        求解不带事件的 ODE。

        求解精度、最大步长和是否启用稀疏雅可比由配置项决定:
        - `solver_profile`
        - `use_sparse_jacobian`
        """
        return self._solve_ode(t_span, t_eval=t_eval)


class DynamicWireRopeSystem3D(WireRopeSystem3D):
    """
    三维动态下放系统。

    该类在基类的静态钢丝绳模型之上增加了:
    - 顶部逐段放缆；
    - 增长段等效原长演化；
    - 25 m 高度停止判据；
    - PCH / 工具串姿态记录；
    - 摘要与数据导出接口。

    同时，它也是三个历史版本能力汇总后的可配置入口。
    """

    def __init__(self, config=None):
        """
        初始化动态下放系统。

        这里的关键工作有两件:
        1. 先根据 `total_depth` 和 `total_segments` 统一离散段长，
           确保初始段长与后续新增段长一致；
        2. 继承基类已经计算好的 PCH / 工具串几何量，避免被旧注释和旧字段覆盖。
        """
        # 下放参数 - 必须在调用父类初始化之前计算，以确保初始段长度正确
        # 优先使用 total_depth（总下放深度），根据 total_segments 自动计算 segment_length
        # 这样可以通过增加段数来提高模拟精度，同时保持下放深度不变
        if config is None:
            config = {}

        verbose = bool(config.get('verbose', DEFAULT_CONFIG.get('verbose', True)))
        
        total_segments = int(config.get('total_segments', 5))
        growth_initial_length = float(config.get(
            'growth_initial_length',
            DEFAULT_CONFIG.get('growth_initial_length', 1.0),
        ))
        
        if 'total_depth' in config:
            # 新方式：指定总深度，自动计算每段长度
            total_depth = float(config['total_depth'])
            # 新段插入时为了避免零长度奇异，会先占用 growth_initial_length。
            # add_new_segment 会把原顶段切短这部分长度；若仍用 total_depth/N，
            # 完成 N 段后总无应力长度会少 (N-1)*growth_initial_length。
            # 这里反向修正目标段长，保证最终 sum(l0) == total_depth。
            segment_length = (
                total_depth + max(total_segments - 1, 0) * growth_initial_length
            ) / max(total_segments, 1)
            if verbose:
                print(f"[下放参数] 总深度={total_depth}m, 段数={total_segments}, 每段长度={segment_length:.2f}m")
        else:
            # 兼容旧方式：直接指定每段长度
            segment_length = config.get('segment_length', 100.0)
            total_depth = segment_length * total_segments
            if verbose:
                print(f"[下放参数] 每段长度={segment_length}m, 段数={total_segments}, 总深度={total_depth:.2f}m")
        
        # 更新配置中的L，确保初始段长度与后续段一致
        config['L'] = segment_length
        
        # 调用父类初始化
        super().__init__(config)
        
        # 保存下放参数
        self.total_segments = total_segments
        self.total_depth = total_depth
        self.segment_length = segment_length
        
        self.release_speed = self.config.get('release_speed', 0.5)  # m/s，默认值仅作后备
        self.release_interval = self.config.get('release_interval', None)
        self.release_duration = (self.segment_length / self.release_speed) if self.release_interval is None else float(self.release_interval)
        if self.release_interval is None:
            self.release_interval = self.release_duration

        # 主动张力控制（ATC）配置
        self.base_release_speed = float(self.release_speed)
        self.current_release_speed_cmd = float(self.base_release_speed)
        self.atc_enabled = bool(self.config.get('active_tension_control_enabled', False))
        self.tension_control_target_scale = float(self.config.get('tension_control_target_scale', 1.08))
        self.tension_control_kp = float(self.config.get('tension_control_kp', 0.35))
        self.tension_control_ki = float(self.config.get('tension_control_ki', 0.02))
        self.tension_control_deadband_ratio = float(self.config.get('tension_control_deadband_ratio', 0.02))
        self.tension_control_speed_min = float(self.config.get('tension_control_speed_min', max(1e-3, 0.5 * self.base_release_speed)))
        self.tension_control_speed_max = float(self.config.get('tension_control_speed_max', max(self.base_release_speed, 1.5 * self.base_release_speed)))
        self.tension_control_rate_limit = float(self.config.get('tension_control_rate_limit', 0.003))
        self.tension_control_integral_limit = float(self.config.get('tension_control_integral_limit', 80.0))
        self._tension_control_integral = 0.0
        self._atc_segment_times = []
        self._atc_segment_measured = []
        self._atc_segment_target = []
        self._atc_segment_speed = []

        # 继承基类真实几何量，避免被旧的遗留字段覆盖。
        # `pch_base_diameter` 只是为了摘要输出保留的兼容别名。
        self.pch_base_diameter = self.pch_lower_cyl_d
        self.tool_length = self.tool_segment_count * self.tool_segment_length
        self.tool_diameter = self.config.get('tool_diameter', self.tool_diameter)

        # 当前已下放段数
        self.current_segments = self.config['N'] - 1

        # 增长段信息
        self.growing_segment_idx = None
        self.growth_start_time_abs = None
        self.growth_initial_length = float(self.config.get('growth_initial_length', 1.0))
        self.growth_target_length = self.segment_length
        self._segment_stiffness_reference_lengths = {}

        # 停止条件标志
        self.target_reached = False
        self.target_reach_time = None

        # 结果记录
        self.full_solution = None
        self.release_data = {
            'times': [],
            'weight_x': [],
            'weight_y': [],
            'weight_z': [],
            'positions': [],
            'strain': [],
            'stress': [],
            'tension': [],
            'segments_released': [],
            'pch_tilt_deg': [],           # PCH倾角
            'pch_horizontal_offset': [],  # PCH水平偏移量
            'height_above_seabed': [],    # 距海底高度
            'tool_tilt_deg': []           # 工具串倾角
        }
        self._sci_plot_cache = None

    def effective_rest_length(self, j, t_abs):
        """
        动态返回第j段的等效原始长度
        """
        if self.growing_segment_idx is None or j != self.growing_segment_idx:
            return self.l0[min(j, len(self.l0) - 1)]
            
        t_start = self.growth_start_time_abs
        if hasattr(self, 'release_data') and 'time' in self.release_data:
            import numpy as np
            hx = np.array(self.release_data['time'])
            vx = hx[hx <= t_abs + 1e-4]
            if len(vx) > 0:
                t_start = vx[-1]

        if t_start is None:
            return self.l0[j]

        # 使用线性增长函数
        t_rel = max(0.0, t_abs - t_start)
        alpha = self._smooth_step(t_rel, getattr(self, 'release_duration', 1e-6))
        L = getattr(self, 'growth_initial_length', 0.0) + alpha * (getattr(self, 'growth_target_length', 100.0) - getattr(self, 'growth_initial_length', 0.0))
        return L

    def _freeze_interrupted_growth_segment(self, t_abs):
        """
        目标高度事件中断增长段时，停止继续下放并保持短段目标刚度。

        短段的实际原长 l0 保留为事件触发时的部分长度；轴向刚度使用
        growth_target_length 作为参考长度，避免额外仿真阶段出现 k=EA/l0 放大。
        """
        idx = getattr(self, 'growing_segment_idx', None)
        if idx is None:
            return
        if isinstance(idx, (list, tuple, set, np.ndarray)):
            idx = int(list(idx)[0])
        else:
            idx = int(idx)

        current_l0 = float(self.effective_rest_length(idx, t_abs))
        target_len = float(max(getattr(self, 'growth_target_length', current_l0), 1e-6))
        self.l0[idx] = current_l0
        self.config['L'] = float(np.sum(self.l0))

        if current_l0 < target_len - 1e-6:
            self._segment_stiffness_reference_lengths[idx] = target_len
            self._log(
                f"目标高度中断第 {idx + 1} 段增长："
                f"l0={current_l0:.3f} m，刚度参考长度={target_len:.3f} m"
            )
        else:
            self._segment_stiffness_reference_lengths.pop(idx, None)
            self.l0[idx] = target_len

        self.growing_segment_idx = None
        self.growth_start_time_abs = None

    def _smooth_step(self, t_rel, T_grow):
        """
        匀速下放曲线（线性增长）
        
        L(t) = L_init + v * t
        
        其中 v = (L_target - L_init) / T_grow = release_speed
        
        参数:
            t_rel: 相对增长时间 = t_abs - growth_start_time_abs
            T_grow: 增长总时长 = release_duration
        
        返回:
            alpha: 0~1 的位移插值系数
        """
        if t_rel <= 0:
            return 0.0
        elif t_rel >= T_grow:
            return 1.0
        else:
            # 线性插值：α = t / T
            return t_rel / T_grow

    def _set_release_speed_command(self, speed_cmd):
        """设置当前放缆速度指令，并同步更新本段增长时长。"""
        speed_cmd = max(float(speed_cmd), 1e-4)
        if self.atc_enabled:
            vmin = max(1e-4, self.tension_control_speed_min)
            vmax = max(vmin, self.tension_control_speed_max)
            speed_cmd = float(np.clip(speed_cmd, vmin, vmax))

        self.current_release_speed_cmd = speed_cmd
        self.release_duration = self.segment_length / max(self.current_release_speed_cmd, 1e-6)
        self.release_interval = self.release_duration

    def _estimate_static_top_tension_from_positions(self, positions):
        """基于当前几何弧长估算静水顶张力。"""
        rho_w = self.config.get('rho_w', 1025.0)
        rho_c = self.config.get('rho_c', 7850.0)
        g = self.config.get('g', 9.81)

        _, _, l_norm = self._compute_segment_vectors(positions)
        rope_length = np.sum(l_norm)
        rope_unit_wet_weight = (rho_c - rho_w) * self.A * g
        payload_wet_weight = (
            (self.pch_mass - rho_w * self.pch_volume) * g
            + (self.tool_mass - rho_w * self.tool_volume) * g
        )
        return float(payload_wet_weight + rope_unit_wet_weight * rope_length)

    def _compute_top_tension_from_positions(self, positions, t_abs):
        """计算当前状态的顶端张力幅值。"""
        if positions.shape[0] < 2:
            return 0.0
        l, l_unit, l_norm = self._compute_segment_vectors(positions)
        T = self._compute_tension(l, l_unit, l_norm, t_abs)
        if T.size == 0:
            return 0.0
        return float(np.linalg.norm(T[0]))

    def _update_active_tension_control(self, positions, t_abs, control_dt):
        """
        主动张力控制：按段更新下一段的放缆速度。

        控制思想：
        - 张力高于目标 -> 降低放缆速度；
        - 张力低于目标 -> 适度提高放缆速度；
        - 使用 deadband、积分限幅和速度变化率限幅，避免抖振。
        """
        if not self.atc_enabled:
            return

        measured = self._compute_top_tension_from_positions(positions, t_abs)
        static_ref = self._estimate_static_top_tension_from_positions(positions)
        target = max(0.0, self.tension_control_target_scale * static_ref)

        err_ratio = (measured - target) / max(target, 1.0)
        if abs(err_ratio) < self.tension_control_deadband_ratio:
            err_ratio = 0.0

        dt_ctrl = max(float(control_dt), 1e-3)
        self._tension_control_integral += err_ratio * dt_ctrl
        self._tension_control_integral = float(np.clip(
            self._tension_control_integral,
            -self.tension_control_integral_limit,
            self.tension_control_integral_limit,
        ))

        # PI 控制：正误差（偏大）时减速，负误差（偏小）时加速
        delta_v = -(
            self.tension_control_kp * err_ratio
            + self.tension_control_ki * self._tension_control_integral
        )
        desired_speed = self.base_release_speed + delta_v

        # 速度变化率限幅，防止段间命令跳变过大
        max_delta = max(self.tension_control_rate_limit * dt_ctrl, 1e-5)
        desired_speed = float(np.clip(
            desired_speed,
            self.current_release_speed_cmd - max_delta,
            self.current_release_speed_cmd + max_delta,
        ))

        self._set_release_speed_command(desired_speed)

        self._atc_segment_times.append(float(t_abs))
        self._atc_segment_measured.append(float(measured))
        self._atc_segment_target.append(float(target))
        self._atc_segment_speed.append(float(self.current_release_speed_cmd))

        self._log(
            f"[ATC] t={t_abs:.1f}s | T_meas={measured/1000:.1f}kN, "
            f"T_tar={target/1000:.1f}kN, v_cmd={self.current_release_speed_cmd:.3f}m/s"
        )

    def compute_pch_tilt_and_offset(self, positions):
        """
        计算 PCH 倾角和水平偏移量。

        `payload_motion_mode` 控制姿态/位置估算假设:
        - `rope_aligned`:
          工程主版本默认。重物单元沿末端钢丝绳方向外推，几何上更一致。
        - `vertical_hanging`:
          `novibrant` 风格。重物底端默认垂直下垂，偏移量只取末端节点水平坐标，
          结果更平滑，但会弱化末端段倾斜的几何影响。

        参数:
            positions: 钢丝绳节点位置数组 (N, 3)

        返回:
            pch_tilt_deg: PCH倾角（度，与竖直向下的夹角）
            horizontal_offset: 水平偏移量（米）
            tool_tilt_deg: 工具串倾角（度，与PCH相同）
        """
        N = positions.shape[0]

        if N < 2:
            return 0.0, 0.0, 0.0

        # 使用钢丝绳末端段方向来估算PCH姿态
        # 末端段：从 positions[-2] 到 positions[-1]
        rope_end_dir = positions[-1] - positions[-2]
        rope_end_norm = np.linalg.norm(rope_end_dir)

        if rope_end_norm < 1e-12:
            return 0.0, 0.0, 0.0

        rope_end_unit = rope_end_dir / rope_end_norm

        # PCH倾角：钢丝绳末端方向与竖直向下方向的夹角
        # PCH尖端朝上，所以PCH轴向与钢丝绳方向相反
        vertical_down = np.array([0.0, 0.0, -1.0])
        cos_theta = np.clip(np.dot(rope_end_unit, vertical_down), -1.0, 1.0)
        pch_tilt_rad = np.arccos(cos_theta)
        pch_tilt_deg = np.degrees(pch_tilt_rad)

        ship_x, ship_y = positions[0, 0], positions[0, 1]
        payload_motion_mode = self.config.get('payload_motion_mode', 'rope_aligned')

        if payload_motion_mode == 'vertical_hanging':
            payload_ref = self.compute_payload_reference_point(positions)
            horizontal_offset = np.sqrt((payload_ref[0] - ship_x)**2 + (payload_ref[1] - ship_y)**2)
        elif payload_motion_mode == 'rope_aligned':
            payload_ref = self.compute_payload_reference_point(positions)
            horizontal_offset = np.sqrt((payload_ref[0] - ship_x)**2 + (payload_ref[1] - ship_y)**2)
        else:
            raise ValueError(f"未知 payload_motion_mode: {payload_motion_mode}")

        # 工具串倾角：假设与PCH刚性连接，倾角相同
        tool_tilt_deg = pch_tilt_deg

        return pch_tilt_deg, horizontal_offset, tool_tilt_deg

    def compute_payload_reference_point(self, positions):
        """
        计算重物参考点坐标。

        当前可视化统一使用重物单元底端（工具串底端）作为参考点，
        这样其 Z 坐标会与 25 m 停止准则直接对应。
        """
        N = positions.shape[0]
        payload_motion_mode = self.config.get('payload_motion_mode', 'rope_aligned')

        if payload_motion_mode == 'vertical_hanging' or N < 2:
            return np.array([
                positions[-1, 0],
                positions[-1, 1],
                positions[-1, 2] - self.payload_total_length,
            ])

        rope_end_dir = positions[-1] - positions[-2]
        rope_end_norm = np.linalg.norm(rope_end_dir)
        if rope_end_norm < 1e-12:
            rope_end_unit = np.array([0.0, 0.0, -1.0])
        else:
            rope_end_unit = rope_end_dir / rope_end_norm

        if payload_motion_mode == 'rope_aligned':
            return positions[-1] + rope_end_unit * self.payload_total_length

        raise ValueError(f"未知 payload_motion_mode: {payload_motion_mode}")

    def compute_height_above_seabed(self, positions):
        """
        计算重物单元底端（工具串底部）距离海底的高度。

        这里和 `compute_pch_tilt_and_offset` 共用同一套 `payload_motion_mode`
        假设，避免几何解释前后不一致。

        参数:
            positions: 钢丝绳节点位置数组 (N, 3)

        返回:
            height: 距海底高度（米）
        """
        seabed_depth = self.config['seabed_depth']  # 负值，如 -1500
        payload_ref = self.compute_payload_reference_point(positions)
        
        # 距海底高度
        height = payload_ref[2] - seabed_depth
        return height

    def get_convergent_offset(self, solution, lookback_time=30.0):
        """
        计算收敛的偏移量指标：使用最后一段时间的平均值
        
        使用"平均偏移量"代替"瞬时偏移量"，取下放结束前最后 2-3 个波浪周期内的
        平均位置，消除振荡带来的相位噪声，反映系统真正的收敛性。
        
        参数:
            solution: 求解结果对象，包含 t, y, N_of_t 等属性
            lookback_time: 回溯时间（秒），应覆盖至少2-3个波浪周期，默认30秒
        
        返回:
            avg_horizontal_offset: 平均水平偏移量（米）
            avg_pch_tilt: 平均PCH倾角（度）
            avg_tool_tilt: 平均工具串倾角（度）
        """
        times = solution.t
        Y = solution.y
        t_end = times[-1]
        N_of_t = getattr(solution, 'N_of_t', np.full(times.shape, self.config['N'], dtype=int))
        
        # 找到最后 lookback_time 秒的数据索引
        mask = times >= (t_end - lookback_time)
        if not np.any(mask):
            # 如果时间太短，取最后10个点
            valid_indices = np.arange(max(0, len(times) - 10), len(times))
        else:
            valid_indices = np.where(mask)[0]
        
        # 提取末端位置并计算偏移量
        offsets = []
        pch_tilts = []
        tool_tilts = []
        
        for idx in valid_indices:
            N_curr = int(N_of_t[idx])
            pos = Y[:3*N_curr, idx].reshape(N_curr, 3)
            
            # 计算PCH倾角和水平偏移量
            pch_tilt, h_offset, tool_tilt = self.compute_pch_tilt_and_offset(pos)
            
            offsets.append(h_offset)
            pch_tilts.append(pch_tilt)
            tool_tilts.append(tool_tilt)
        
        # 返回平均值
        avg_horizontal_offset = np.mean(offsets)
        avg_pch_tilt = np.mean(pch_tilts)
        avg_tool_tilt = np.mean(tool_tilts)
        
        return avg_horizontal_offset, avg_pch_tilt, avg_tool_tilt

    def _record_release_snapshot(self, t_abs, positions):
        """
        记录下放快照 - 使用统一的应力计算方法
        
        使用 compute_stress_from_tension 方法计算应力，
        这是物理上最准确的方法，因为张力已经考虑了所有修正。
        """
        l, l_unit, l_norm = self._compute_segment_vectors(positions)
        
        # 1. 计算真实的张力 T
        T = self._compute_tension(l, l_unit, l_norm, t_abs)
        Tmag = np.linalg.norm(T, axis=1) if T.size else np.array([])
        
        # 2. 使用统一的应力计算方法（修复2）
        stress = self.compute_stress_from_tension(T)
        
        # 3. 计算名义应变 (仅作参考)
        strain = self._compute_strain(l_norm, t_abs)

        # 计算PCH相关参数
        pch_tilt_deg, horizontal_offset, tool_tilt_deg = self.compute_pch_tilt_and_offset(positions)
        height_above_seabed = self.compute_height_above_seabed(positions)

        # 保存数据
        self.release_data['times'].append(t_abs)
        self.release_data['weight_x'].append(positions[-1, 0])
        self.release_data['weight_y'].append(positions[-1, 1])
        self.release_data['weight_z'].append(positions[-1, 2])
        self.release_data['positions'].append(positions.copy())
        self.release_data['strain'].append(strain.copy())
        self.release_data['stress'].append(stress.copy())  # 这里存的是真实应力
        self.release_data['tension'].append(Tmag.copy())
        self.release_data['segments_released'].append(self.current_segments)
        self.release_data['pch_tilt_deg'].append(pch_tilt_deg)
        self.release_data['pch_horizontal_offset'].append(horizontal_offset)
        self.release_data['height_above_seabed'].append(height_above_seabed)
        self.release_data['tool_tilt_deg'].append(tool_tilt_deg)

    def add_new_segment(self, prev_solution, current_time_abs):
        """
        添加新段钢丝绳 - 改进版
        
        改进点：
        1. 摒弃固定长度暴力插入，采用基于原长（无应力长度）比例的切分方式。
        2. 原最上段（刚好达到 target_length）被等比例切分为 target_length-initial_length 和 initial_length。
        3. 切分后，切口两侧子段的几何拉伸应变（从而张力）完美守恒，根除添加新段时的张力骤降和回弹冲击。
        """
        N = self.config['N']
        final_state = prev_solution.y[:, -1]
        
        # --- 记录能量 (跳变前) ---
        E_before = self._compute_system_energy(final_state)
        pos = final_state[:3 * N].reshape(N, 3)
        vel = final_state[3 * N:].reshape(N, 3)

        # 记录下放前状态
        self._record_release_snapshot(current_time_abs, pos)

        # 改版：按无应力原长比例来切分物理距离，以完美保持切分前后的应变和局部张力
        if N >= 2:
            top_vec = pos[1] - pos[0]
            # ratio是新段刚开始伸出的无应力原长占刚好完成下放的老顶段总无应力原长的比例
            ratio = self.growth_initial_length / self.growth_target_length
            ratio = min(max(ratio, 0.0), 1.0)
        else:
            top_vec = np.array([0.0, 0.0, -1.0])
            ratio = 1.0

        new_node_pos = pos[0] + ratio * top_vec
        
        # 改进：新节点的下放速度应为真实的材料输送速度！
        # vel[0] 是船体边界静止，不能代表绳带滑动速度
        if hasattr(self, 'release_duration') and self.release_duration > 1e-6:
            v_payout = (self.growth_target_length - self.growth_initial_length) / self.release_duration
        else:
            v_payout = getattr(self, 'release_speed', 0.0)
            
        top_dir_normalized = top_vec / (np.linalg.norm(top_vec) + 1e-12)
        material_vel_top = vel[0] + v_payout * top_dir_normalized
        
        # 新节点速度按真实材料速度比例插值，消灭迟滞落回冲击
        new_node_vel = (1 - ratio) * material_vel_top + ratio * vel[1]

        # 组合：[node0, new_node, node1, node2, ...]
        new_positions = np.vstack([pos[0:1], new_node_pos[None, :], pos[1:]])
        new_velocities = np.vstack([vel[0:1], new_node_vel[None, :], vel[1:]])

        # 更新系统
        self.config['N'] = N + 1
        
        # 核心修复：更新数组时，原来在 index 0 的整段被切分。
        # 留在 index 1 的实际原段变成 target - initial。这步是消灭锯齿波动的最关键环节！
        if len(self.l0) > 0:
            self.l0[0] = self.growth_target_length - self.growth_initial_length
            
        if self._segment_stiffness_reference_lengths:
            self._segment_stiffness_reference_lengths = {
                int(seg_idx) + 1: ref_len
                for seg_idx, ref_len in self._segment_stiffness_reference_lengths.items()
            }

        self.l0 = np.insert(self.l0, 0, self.segment_length)
        self.config['L'] = np.sum(self.l0)

        # 设置增长段信息
        self.growing_segment_idx = 0  # 新插入的段在索引0
        self.growth_start_time_abs = current_time_abs

        # 更新初始状态
        self.init_state = np.concatenate([new_positions.flatten(), new_velocities.flatten()])
        self.current_segments += 1

        # --- 记录能量 (跳变后) ---
        E_after = self._compute_system_energy(self.init_state)
        if not hasattr(self, 'energy_jumps'):
            self.energy_jumps = []
        
        rel_dE = abs(E_after - E_before) / (abs(E_before) + 1e-6)
        self.energy_jumps.append({
            'seg': self.current_segments,
            'E_before': E_before,
            'E_after': E_after,
            'dE': E_after - E_before,
            'rel_dE': rel_dE
        })

        self._log(f"已添加第 {self.current_segments} 段，节点数 = {self.config['N']}")

    def check_target_reached(self, positions):
        """
        检查是否到达目标高度（距海底25m）
        
        修复：移除双向误差范围，只检查是否已低于目标高度
        精确停止由 ODE 事件检测处理

        参数:
            positions: 节点位置数组 (N, 3)

        返回:
            reached: 是否到达目标
        """
        target_height = self.config['target_height_above_seabed']
        current_height = self.compute_height_above_seabed(positions)

        # 简单条件：只有当已经低于或等于目标时才返回True
        # 这避免了双向误差范围导致的非单调性
        return current_height <= target_height

    def _target_height_event(self, t, state):
        """
        ODE 事件函数：检测到达目标高度
        
        当返回值从正变负时，表示越过目标高度
        
        参数:
            t: 时间
            state: 状态向量 [positions, velocities]
        
        返回:
            float: height_above_seabed - target_height
                   正值表示还在目标上方，负值表示已过目标
        """
        N = self.config['N']
        positions = state[:3 * N].reshape(N, 3)
        
        # 计算当前高度
        current_height = self.compute_height_above_seabed(positions)
        target_height = self.config['target_height_above_seabed']
        
        return current_height - target_height

    def solve_with_event(self, t_span, t_eval=None):
        """
        带事件检测的 ODE 求解。

        事件函数只负责"何时停止"；实际求解参数与 `solve` 完全复用 `_solve_ode`。
        """
        # 创建事件函数（带属性）
        def event_func(t, y):
            return self._target_height_event(t, y)
        event_func.terminal = True   # 事件触发时终止积分
        event_func.direction = -1    # 只检测从正到负的穿越（下降到目标）
        sol = self._solve_ode(t_span, t_eval=t_eval, events=event_func)
        
        # 检查是否因事件而停止
        if sol.t_events is not None and len(sol.t_events) > 0 and len(sol.t_events[0]) > 0:
            sol.stopped_by_event = True
            sol.event_time = float(sol.t_events[0][0])
            if getattr(sol, 'y_events', None) is not None and len(sol.y_events) > 0 and len(sol.y_events[0]) > 0:
                event_state = np.asarray(sol.y_events[0][0], dtype=float)
                if len(sol.t) == 0 or abs(float(sol.t[-1]) - sol.event_time) > 1e-9:
                    sol.t = np.append(sol.t, sol.event_time)
                    sol.y = np.column_stack([sol.y, event_state])
                else:
                    sol.y[:, -1] = event_state
        else:
            sol.stopped_by_event = False
            sol.event_time = None
        
        return sol




    def _compute_system_energy(self, state):
        """
        计算系统当前机械能 (动能 + 重力势能 + 弹性势能)
        用于验证数值稳定性和能量跳变
        """
        import numpy as np
        N = self.config['N']
        pos = state[:3 * N].reshape(N, 3)
        vel = state[3 * N:].reshape(N, 3)

        # 1. 动能 (忽略两端边界节点速度)
        # 近似: 绳段质量 m_segment = rho_c * A * L_i, 这里简化为按点分配
        # 实际为简单比对不要求绝对值精确，但求前后口径一致
        v_sq = np.sum(vel**2, axis=1)
        # 用 m_b 和缆质量粗略计算动能
        # 简化跳变：仅使用绳节点的平均质量
        KE = 0.5 * np.sum(v_sq) * 100.0  # mock mass multiplier for relative tracking

        # 2. 重力势能
        z = pos[:, 2]
        PE = np.sum(z) * 100.0 * 9.81

        # 3. 弹性应变能 SE = 0.5 * K * dx^2
        SE = 0.0
        for i in range(N - 1):
            vec = pos[i+1] - pos[i]
            l = np.linalg.norm(vec)
            l0_i = self.l0[i]
            if l0_i > 1e-6 and l > l0_i:
                strain = (l - l0_i) / l0_i
                tension = self.config['E'] * self.A * strain
                # 能 = 0.5 * F * dx
                SE += 0.5 * tension * (l - l0_i)

        return KE + PE + SE

    def compute_engineering_metrics(self):
        """
        计算工程论文必须的动力学指标: DAF 与 Snap Load
        返回字典结构包含各项指标
        """
        import numpy as np

        tensions = getattr(self, '_hist_tension', None)
        if not tensions:
            if hasattr(self, 'release_data') and 'tension' in self.release_data:
                tensions = self.release_data['tension']
            else:
                print("未检测到张力历史数据，请先执行模拟。")
                return None

        # Compute max top tension and max bottom tension manually for inhomogeneous array
        top_tensions = []
        bot_tensions = []
        for t_arr in tensions:
            if len(t_arr) > 0:
                top_tensions.append(np.linalg.norm(t_arr[0]))
                bot_tensions.append(np.linalg.norm(t_arr[-1]))
        
        T_dyn_max = np.max(top_tensions) if top_tensions else 0

        L_total = sum(self.l0)
        g = self.config.get('g', 9.81)
        rho_w = self.config.get('rho_w', 1025.0)
        rho_c = self.config.get('rho_c', 7850.0)

        W_pch_wet = (self.pch_mass - rho_w * self.pch_volume) * g
        W_tool_wet = (self.tool_mass - rho_w * self.tool_volume) * g
        W_rope_wet = (rho_c * self.A * L_total - rho_w * self.A * L_total) * g

        T_static = W_pch_wet + W_tool_wet + W_rope_wet
        DAF = T_dyn_max / T_static if T_static > 0 else 1.0

        bot_tensions = np.array(bot_tensions)
        SLACK_THRESHOLD = 5.0 # Set slightly higher for N norm
        slack_taut_events = 0
        
        T_mean_bot = np.mean(bot_tensions) if len(bot_tensions)>0 else 0
        T_max_bot = np.max(bot_tensions) if len(bot_tensions)>0 else 0
        impact_ratio_bot = T_max_bot / T_mean_bot if T_mean_bot > 0 else 0.0
        
        is_slack = bot_tensions <= SLACK_THRESHOLD
        for i in range(1, len(is_slack)):
            if is_slack[i-1] and not is_slack[i]:
                if bot_tensions[i] > T_mean_bot * 1.5:
                    slack_taut_events += 1

        t_slack_duration = np.sum(is_slack) / len(is_slack) * 100.0 if len(is_slack)>0 else 0.0

        metrics = {
            "DAF": float(DAF),
            "T_dyn_max": float(T_dyn_max),
            "T_static": float(T_static),
            "SnapLoad_Events": int(slack_taut_events),
            "Impact_Ratio_Bot": float(impact_ratio_bot),
            "Slack_Time_Percentage": float(t_slack_duration)
        }

        pass
        pass
        print("海洋工程动力学核心指标分析 (Ocean Engineering Metrics)")
        print("="*50)
        print(f"1. 动力放大系数 (DAF)   : {DAF:.3f}")
        print(f"   - 最大动态张力 T_dyn : {T_dyn_max/1000:.2f} kN")
        print(f"   - 理论等效全长静载 : {T_static/1000:.2f} kN")
        print(f"2. 突然加载 (Snap Load) 分析 (取底部载荷端)")
        print(f"   - Slack-Taut 循环次数: {slack_taut_events} 次")
        print(f"   - 冲击载荷峰值/均值比: {impact_ratio_bot:.2f} 倍")
        print(f"   - 绳索处于松弛(张力接近0)时间比: {t_slack_duration:.2f} %")
        print("="*50 + "\n")
        return metrics



    def simulate_dynamic_release(self, t_end, dt_output=1.0):
        """
        模拟动态下放过程。

        默认行为:
        - 使用事件检测在 25 m 停止高度处精确停下；
        - 若关闭事件检测，则回退到离散检查；
        - 日志输出受 `verbose` 控制。
        """
        self._log("="*70)
        self._log("开始三维连续动态下放模拟（带25m停止条件）")
        self._log("="*70)

        self._hist_max_stress = None
        self._hist_mean_stress = None
        self._hist_top_stress = None
        self._hist_max_strain = None
        self._hist_mean_strain = None
        self._hist_tension = None
        self._hist_top_tension = None
        self._hist_mean_tension = None
        self._hist_max_tension = None
        self._hist_segment_mid_z = None
        self._hist_segment_stress = None
        self._hist_tension_control_target = None
        self._hist_tension_control_error = None
        self._hist_release_speed_cmd = None
        self._sci_plot_cache = None
        self.release_data = {key: [] for key in self.release_data}

        self.current_release_speed_cmd = float(self.base_release_speed)
        self._tension_control_integral = 0.0
        self._atc_segment_times = []
        self._atc_segment_measured = []
        self._atc_segment_target = []
        self._atc_segment_speed = []
        self._set_release_speed_command(self.current_release_speed_cmd)

        simulation_start_time = time.time()

        abs_time = 0.0
        all_times = []
        block_states = []
        block_Ns = []
        use_event_detection = bool(self.config.get('event_detection_enabled', True))
        
        # 求解时间记录
        solve_times = []
        methods_used = []

        # 初始快照
        init_positions = self.init_state[:3 * self.config['N']].reshape(self.config['N'], 3)
        self._record_release_snapshot(abs_time, init_positions)

        target_total_segments = self.total_segments

        while (self.current_segments < target_total_segments) and (abs_time < t_end) and (not self.target_reached):
            # 每一段积分前使用当前控制器速度命令更新增长时长
            self._set_release_speed_command(self.current_release_speed_cmd)

            # 创建伪解用于add_new_segment
            class _SimpleSol: pass
            pseudo = _SimpleSol()
            pseudo.y = self.init_state.reshape(-1, 1)

            # 添加新段
            self.add_new_segment(pseudo, abs_time)

            # 计算增长时长
            growth_T = min(self.release_duration, max(0.0, t_end - abs_time))
            if growth_T <= 0:
                break

            # 积分（使用带事件检测的求解）
            self.time_offset = abs_time
            t_span = (0.0, growth_T)
            t_eval = np.arange(0.0, growth_T, dt_output)
            if len(t_eval) == 0 or t_eval[-1] < growth_T:
                t_eval = np.append(t_eval, growth_T)

            self._log(f"积分第 {self.current_segments} 段（{growth_T:.1f}s）...")
            if use_event_detection:
                sol = self.solve_with_event(t_span, t_eval)
            else:
                sol = self.solve(t_span, t_eval)
                sol.stopped_by_event = False
                sol.event_time = None
            solve_times.append(sol.solve_time)
            methods_used.append(sol.method_used)
            self._log(f"  -> 求解耗时: {sol.solve_time:.2f}s, 方法: {sol.method_used}")

            # 检查是否因事件而停止（精确检测）
            if getattr(sol, 'stopped_by_event', False):
                self.target_reached = True
                self.target_reach_time = abs_time + sol.event_time
                actual_growth_T = sol.event_time
                self._log(f"\n*** 事件检测：在 t={self.target_reach_time:.2f}s 精确到达目标高度 ***\n")
            else:
                actual_growth_T = growth_T
                # 备用：离散检查（作为后备）
                final_pos = sol.y[:3 * self.config['N'], -1].reshape(self.config['N'], 3)
                if self.check_target_reached(final_pos):
                    self.target_reached = True
                    self.target_reach_time = abs_time + growth_T
                    self._log("\n*** 离散检测：已到达目标高度（距海底25m），停止下放 ***")
                    self._log(f"*** 到达时间: {self.target_reach_time:.2f}s ***\n")

            # 保存结果
            times_abs = sol.t + abs_time
            if not all_times:
                all_times.extend(times_abs)
                block_states.append(sol.y)
                block_Ns.append(self.config['N'])
                self._evaluate_block_history(sol, abs_time, True)
            else:
                all_times.extend(times_abs[1:])
                block_states.append(sol.y[:, 1:])
                block_Ns.append(self.config['N'])
                self._evaluate_block_history(sol, abs_time, False)

            abs_time += actual_growth_T  # 使用实际增长时间
            self.init_state = sol.y[:, -1]

            if getattr(self, 'growing_segment_idx', None) is not None:
                final_l0 = self.effective_rest_length(self.growing_segment_idx, abs_time)
                self.l0[self.growing_segment_idx] = final_l0

            # 如果目标高度事件中断增长段，则停止继续下放，只冻结当前短段刚度。
            if self.target_reached and actual_growth_T < growth_T - 1e-6:
                self._freeze_interrupted_growth_segment(abs_time)
            else:
                # 正常完成增长
                self.growing_segment_idx = None
                self.growth_start_time_abs = None

            # 记录快照
            final_pos = self.init_state[:3 * self.config['N']].reshape(self.config['N'], 3)
            self._record_release_snapshot(abs_time, final_pos)

            # ATC 更新：基于本段末状态，生成下一段放缆速度命令
            if self.atc_enabled and (not self.target_reached):
                self._update_active_tension_control(final_pos, abs_time, actual_growth_T)
            
            # 如果事件触发，跳出循环
            if self.target_reached:
                break

        # 剩余自由演化（下放完毕或到达目标后继续仿真）
        # 配置：到达目标后额外仿真时间（秒）
        extra_sim_time_after_target = self.config.get('extra_sim_time_after_target', 0.0)
        
        # 计算剩余仿真时间
        if self.target_reached:
            # 到达目标后，继续仿真指定时间
            remain_T = extra_sim_time_after_target
            self._log(f"到达目标高度，继续仿真 {remain_T:.1f}s...")
        elif abs_time < t_end:
            # 未到达目标但还有时间，继续到 t_end
            remain_T = t_end - abs_time
            self._log(f"完成所有下放，继续模拟至 {t_end:.1f}s...")
        else:
            remain_T = 0
        
        if remain_T > 0:
            # 将额外仿真分段求解（每段不超过 chunk_T 秒），
            # 避免长时间连续积分导致数值振荡累积。
            chunk_T = min(self.release_duration, 50.0)
            elapsed = 0.0
            first_chunk = (len(all_times) == 0)
            while elapsed < remain_T - 1e-6:
                this_T = min(chunk_T, remain_T - elapsed)
                self.time_offset = abs_time + elapsed
                t_span_chunk = (0.0, this_T)
                t_eval_chunk = np.arange(0.0, this_T, dt_output)
                if len(t_eval_chunk) == 0 or t_eval_chunk[-1] < this_T:
                    t_eval_chunk = np.append(t_eval_chunk, this_T)

                if use_event_detection and not self.target_reached:
                    sol2 = self.solve_with_event(t_span_chunk, t_eval_chunk)
                else:
                    sol2 = self.solve(t_span_chunk, t_eval_chunk)
                    sol2.stopped_by_event = False
                    sol2.event_time = None

                if getattr(sol2, 'stopped_by_event', False):
                    self.target_reached = True
                    self.target_reach_time = abs_time + elapsed + sol2.event_time
                    self._log(f"\n*** 自由演化事件检测：在 t={self.target_reach_time:.2f}s 精确到达目标高度 ***\n")

                solve_times.append(sol2.solve_time)
                methods_used.append(sol2.method_used)

                times_abs = sol2.t + abs_time + elapsed
                if first_chunk and not all_times:
                    all_times.extend(times_abs)
                    block_states.append(sol2.y)
                    block_Ns.append(self.config['N'])
                    self._evaluate_block_history(sol2, abs_time + elapsed, True)
                    first_chunk = False
                else:
                    all_times.extend(times_abs[1:])
                    block_states.append(sol2.y[:, 1:])
                    block_Ns.append(self.config['N'])
                    self._evaluate_block_history(sol2, abs_time + elapsed, False)

                self.init_state = sol2.y[:, -1]
                elapsed += this_T

                if getattr(sol2, 'stopped_by_event', False):
                    break

            self._log(f"  -> 自由演化求解耗时: {sum(solve_times[-int(np.ceil(remain_T/chunk_T))::]):.2f}s")

        # 处理退化情形
        if not all_times:
            all_times = [0.0]
            block_states = [self.init_state.reshape(-1, 1)]
            block_Ns = [self.config['N']]

        # 对齐数据
        max_N = max(block_Ns)
        T_total = sum(arr.shape[1] for arr in block_states)
        Y = np.zeros((6 * max_N, T_total))
        node_counts_t = np.empty(T_total, dtype=int)

        col = 0
        for arr, Nk in zip(block_states, block_Ns):
            Ti = arr.shape[1]
            Y[0:3*Nk, col:col+Ti] = arr[0:3*Nk, :]
            Y[3*max_N:3*max_N + 3*Nk, col:col+Ti] = arr[3*Nk:6*Nk, :]
            node_counts_t[col:col+Ti] = Nk
            col += Ti

        full = SimpleNamespace()
        full.t = np.array(all_times)
        full.y = Y
        full.success = True
        full.N_of_t = node_counts_t
        
        # 记录求解时间统计
        total_simulation_time = time.time() - simulation_start_time
        total_solve_time = sum(solve_times) if solve_times else 0.0
        full.total_solve_time = total_solve_time
        full.total_simulation_time = total_simulation_time
        full.solve_times = solve_times
        full.methods_used = methods_used

        self.full_solution = full
        self._log(f"\n模拟完成：总时长 {full.t[-1]:.2f}s，最大节点数 {max_N}，段数 {self.current_segments}")
        self._log(f"求解时间统计：ODE求解 {total_solve_time:.2f}s，总耗时 {total_simulation_time:.2f}s")
        self._log("="*70)
        return full

    def _evaluate_block_history(self, sol, abs_time, include_first):
        if getattr(self, '_hist_max_stress', None) is None:
            self._hist_max_stress = []
            self._hist_mean_stress = []
            self._hist_top_stress = []
            self._hist_max_strain = []
            self._hist_mean_strain = []
            self._hist_tension = []
            self._hist_top_tension = []
            self._hist_mean_tension = []
            self._hist_max_tension = []
            self._hist_segment_mid_z = []
            self._hist_segment_stress = []
            self._hist_tension_control_target = []
            self._hist_tension_control_error = []
            self._hist_release_speed_cmd = []

        def _append_history_sample(
            stress,
            strain,
            T,
            Tmag,
            seg_mid_z,
            control_target,
            control_error,
        ):
            self._hist_max_stress.append(float(np.max(stress)) if len(stress) > 0 else 0.0)
            self._hist_mean_stress.append(float(np.mean(stress)) if len(stress) > 0 else 0.0)
            self._hist_top_stress.append(float(stress[0]) if len(stress) > 0 else 0.0)
            self._hist_max_strain.append(float(np.max(strain)) if len(strain) > 0 else 0.0)
            self._hist_mean_strain.append(float(np.mean(strain)) if len(strain) > 0 else 0.0)
            self._hist_tension.append(T.copy())
            self._hist_top_tension.append(float(Tmag[0]) if len(Tmag) > 0 else 0.0)
            self._hist_mean_tension.append(float(np.mean(Tmag)) if len(Tmag) > 0 else 0.0)
            self._hist_max_tension.append(float(np.max(Tmag)) if len(Tmag) > 0 else 0.0)
            self._hist_segment_mid_z.append(seg_mid_z.copy())
            self._hist_segment_stress.append(stress.copy())
            self._hist_tension_control_target.append(float(control_target))
            self._hist_tension_control_error.append(float(control_error))
            self._hist_release_speed_cmd.append(float(self.current_release_speed_cmd))
            
        start_idx = 0 if include_first else 1
        N = self.config['N']
        for i in range(start_idx, len(sol.t)):
            t = sol.t[i] + abs_time
            pos = sol.y[:3*N, i].reshape(N, 3)
            l, l_unit, l_norm = self._compute_segment_vectors(pos)
            strain = self._compute_strain(l_norm, t)
            T = self._compute_tension(l, l_unit, l_norm, t)
            
            if T.size > 0:
                stress = self.compute_stress_from_tension(T) if hasattr(self, 'compute_stress_from_tension') else (np.linalg.norm(T, axis=1) / self.A)
            else:
                stress = np.array([])
            Tmag = np.linalg.norm(T, axis=1) if T.size > 0 else np.array([])
                
            static_top = self._estimate_static_top_tension_from_positions(pos)
            if self.atc_enabled:
                control_target = self.tension_control_target_scale * static_top
                if len(stress) > 0:
                    control_error = Tmag[0] - control_target
                else:
                    control_error = -control_target
            else:
                control_target = np.nan
                control_error = np.nan

            _append_history_sample(
                stress=stress,
                strain=strain,
                T=T if len(stress) > 0 else np.zeros((0, 3)),
                Tmag=Tmag if len(stress) > 0 else np.array([]),
                seg_mid_z=(0.5 * (pos[:-1, 2] + pos[1:, 2])) if len(stress) > 0 else np.array([]),
                control_target=control_target,
                control_error=control_error,
            )

    def _compute_stress_strain_history(self, times, Y, N_of_t):
        if getattr(self, '_hist_max_stress', None) is not None and len(self._hist_max_stress) == len(times):
            smooth_enabled = bool(self.config.get('visualization_smooth_enabled', False))
            if not smooth_enabled:
                return (
                    np.array(self._hist_max_stress),
                    np.array(self._hist_mean_stress),
                    np.array(self._hist_top_stress),
                    np.array(self._hist_max_strain),
                    np.array(self._hist_mean_strain),
                )

            import scipy.ndimage as ndimage
            import scipy.signal as signal

            def smooth_series(arr):
                arr = np.array(arr)
                if len(arr) > 201:
                    # 中值滤波去除由ODE数值刚度/质量突变导致的高频突刺
                    arr_med = ndimage.median_filter(arr, size=51)
                    # Savitzky-Golay进一步平滑曲线
                    return signal.savgol_filter(arr_med, window_length=201, polyorder=3)
                elif len(arr) > 51:
                    arr_med = ndimage.median_filter(arr, size=15)
                    return signal.savgol_filter(arr_med, window_length=51, polyorder=3)
                return arr
                
            return (
                smooth_series(np.array(self._hist_max_stress)),
                smooth_series(np.array(self._hist_mean_stress)),
                smooth_series(np.array(self._hist_top_stress)),
                smooth_series(np.array(self._hist_max_strain)),
                smooth_series(np.array(self._hist_mean_strain))
            )

        # Compute stress/strain histories using sigma=|T|/A
        max_stress_history = []
        mean_stress_history = []
        top_stress_history = []
        max_strain_history = []
        mean_strain_history = []

        # Find start times for each Ni block to correctly evaluate dynamic lengths
        start_time_of_N = {}
        for i, t in enumerate(times):
            Ni = int(N_of_t[i])
            if Ni not in start_time_of_N:
                start_time_of_N[Ni] = t

        is_dynamic = hasattr(self, 'growth_start_time_abs')
        orig_idx = getattr(self, 'growing_segment_idx', None)
        orig_t_abs = getattr(self, 'growth_start_time_abs', None)

        try:
            for i, t in enumerate(times):
                Ni = int(N_of_t[i])
                
                if is_dynamic:
                    self.growing_segment_idx = 0
                    self.growth_start_time_abs = start_time_of_N[Ni]
                    
                pos = Y[:3*Ni, i].reshape(Ni, 3)

                l, l_unit, l_norm = self._compute_segment_vectors(pos)
                strain = self._compute_strain(l_norm, t)
                T = self._compute_tension(l, l_unit, l_norm, t)

                if T.size > 0:
                    Tmag = np.linalg.norm(T, axis=1)
                    stress = Tmag / self.A
                else:
                    stress = np.array([])

                if len(stress) > 0:
                    max_stress_history.append(np.max(stress))
                    mean_stress_history.append(np.mean(stress))
                    top_stress_history.append(stress[0])
                    max_strain_history.append(np.max(strain))
                    mean_strain_history.append(np.mean(strain))
                else:
                    max_stress_history.append(0.0)
                    mean_stress_history.append(0.0)
                    top_stress_history.append(0.0)
                    max_strain_history.append(0.0)
                    mean_strain_history.append(0.0)
        finally:
            if is_dynamic:
                self.growing_segment_idx = orig_idx
                self.growth_start_time_abs = orig_t_abs

        return (np.array(max_stress_history), np.array(mean_stress_history),
                np.array(top_stress_history), np.array(max_strain_history),
                np.array(mean_strain_history))

    def _get_segment_transition_times(self):
        """
        提取每次新增段发生的起止时刻。

        该信息主要用于展示型曲线，在段切换的瞬态窗口内做平滑过渡，
        避免图上出现过分尖锐的可视化跳变。
        """
        times = self.release_data.get('times', [])
        segments = self.release_data.get('segments_released', [])
        if len(times) < 2 or len(times) != len(segments):
            return [], []

        segment_start_times = []
        segment_end_times = []

        for i in range(1, len(segments)):
            if segments[i] > segments[i - 1]:
                segment_start_times.append(times[i])

                next_end_time = times[-1]
                for j in range(i + 1, len(segments)):
                    if segments[j] > segments[i]:
                        next_end_time = times[j]
                        break
                segment_end_times.append(next_end_time)

        return segment_start_times, segment_end_times

    def _compute_reference_depth_stress(self, stress, positions, reference_depth):
        """
        返回固定深度处的参考应力。

        `novibrant` 的一个有用经验是：顶部段索引会因新段插入而跳变，
        所以展示图中更稳定的做法是跟踪固定深度处的应力。
        """
        if len(stress) == 0:
            return 0.0

        if positions.shape[0] <= 1:
            return float(stress[0])

        segment_depths = np.zeros(len(stress))
        for seg_j in range(len(stress)):
            mid_z = 0.5 * (positions[seg_j, 2] + positions[seg_j + 1, 2])
            segment_depths[seg_j] = -mid_z

        idx = np.searchsorted(segment_depths, reference_depth)
        idx = min(max(idx, 0), len(stress) - 1)
        return float(stress[idx])

    def print_deployment_summary(self, solution=None):
        """
        打印下放过程总结信息
        包括：最终落点坐标、应力峰值、推荐平均偏移指标、总下放时间和关键几何参数。

        工程约定:
        - `summary_safety_factor_basis='raw'` 为默认推荐设置；
        - 若切到 `filtered`，仅建议用于汇报和趋势对比，不建议替代设计峰值校核。
        """
        if solution is None:
            solution = self.full_solution
        metrics = self._collect_deployment_summary_metrics(solution)
        if metrics is None:
            print("错误：没有可用的解决方案")
            return

        final_payload_pos = metrics['final_payload_pos']
        final_height = metrics['final_height']
        final_pch_tilt = metrics['final_pch_tilt']
        final_h_offset = metrics['final_h_offset']
        final_tool_tilt = metrics['final_tool_tilt']
        avg_h_offset = metrics['avg_h_offset']
        avg_pch_tilt = metrics['avg_pch_tilt']
        avg_tool_tilt = metrics['avg_tool_tilt']
        global_max_stress = metrics['global_max_stress']
        global_max_stress_filtered = metrics['global_max_stress_filtered']
        max_stress_time = metrics['max_stress_time']
        max_stress_time_filtered = metrics['max_stress_time_filtered']
        max_stress_segment = metrics['max_stress_segment']
        deployment_complete_time = metrics['deployment_complete_time']
        safety_stress = metrics['safety_stress']
        safety_label = metrics['safety_label']
        raw_max_stress = metrics['raw_max_stress']
        safety_factor_basis = metrics['safety_factor_basis']
        times = metrics['times']

        # 打印总结
        print("\n" + "="*80)
        print(" " * 25 + "下放过程总结")
        print("="*80)
        print(f"\n【工具串最终位置（最下方）】")
        print(f"  X = {final_payload_pos[0]:>10.3f} m")
        print(f"  Y = {final_payload_pos[1]:>10.3f} m")
        print(f"  Z = {final_payload_pos[2]:>10.3f} m (深度: {-final_payload_pos[2]:.3f} m)")
        print(f"  距海底高度:   {final_height:>10.3f} m")

        print(f"\n【PCH姿态参数】")
        print(f"  PCH倾角(瞬时):   {final_pch_tilt:>10.3f} 度")
        print(f"  PCH倾角(平均):   {avg_pch_tilt:>10.3f} 度  ← 推荐指标")
        print(f"  水平偏移(瞬时):  {final_h_offset:>10.3f} m")
        print(f"  水平偏移(平均):  {avg_h_offset:>10.3f} m   ← 推荐指标")
        print(f"  工具串倾角(瞬时):{final_tool_tilt:>10.3f} 度")
        print(f"  工具串倾角(平均):{avg_tool_tilt:>10.3f} 度  ← 推荐指标")

        print(f"\n【钢丝绳应力信息】")
        print(f"  最大应力(原始):   {global_max_stress/1e6:>10.3f} MPa (t={max_stress_time:.2f}s)")
        print(f"  最大应力(滤波):   {global_max_stress_filtered/1e6:>10.3f} MPa (t={max_stress_time_filtered:.2f}s)")
        print(f"  发生段号:     第 {max_stress_segment} 段")

        # 改进：采用工程真实的 MBL (Minimum Breaking Load) 而非 0.2% 许用应变
        if 'breaking_load' in self.config:
            mbl = self.config['breaking_load']
            print(f"  破断拉力(MBL): {mbl / 1000:>10.3f} kN (DNV规范参考值)")
            
            max_p_tension = safety_stress * self.A
            max_p_raw = global_max_stress * self.A
            max_p_filt = global_max_stress_filtered * self.A
            
            safety_factor = mbl / max(max_p_tension, 1e-9)
            safety_factor_raw = mbl / max(max_p_raw, 1e-9)
            safety_factor_filtered = mbl / max(max_p_filt, 1e-9)
            
            print(f"  安全系数({safety_label}, MBL基准): {safety_factor:>10.3f}")
            print(f"  安全系数(原始, MBL基准): {safety_factor_raw:>10.3f}")
            print(f"  安全系数(滤波, MBL基准): {safety_factor_filtered:>10.3f}")
        else:
            # 兼容老配置，若没有 breaking_load 回退
            print(f"  材料极限:     {self.config['E'] * 0.002 / 1e6:>10.3f} MPa (假设许用应变=0.2%)")
            safety_factor = (self.config['E'] * 0.002) / max(safety_stress, 1e-9)
            safety_factor_raw = (self.config['E'] * 0.002) / max(global_max_stress, 1e-9)
            safety_factor_filtered = (self.config['E'] * 0.002) / max(global_max_stress_filtered, 1e-9)
            print(f"  安全系数({safety_label}, 应变判据): {safety_factor:>10.3f}")
            print(f"  安全系数(原始, 应变判据): {safety_factor_raw:>10.3f}")
            print(f"  安全系数(滤波, 应变判据): {safety_factor_filtered:>10.3f}")

        print(f"\n【下放时间信息】")
        print(f"  下放段数:     {self.current_segments} 段")
        print(f"  单段长度:     {self.segment_length:.1f} m")
        print(f"  下放速度:     {self.release_speed:.2f} m/s")
        print(f"  下放完毕时间: {deployment_complete_time:>10.2f} s")
        if self.target_reached:
            print(f"  到达目标时间: {self.target_reach_time:>10.2f} s *** (25m停止条件触发) ***")
        print(f"  总模拟时间:   {times[-1]:>10.2f} s")

        print(f"\n【系统参数】")
        print(f"  钢丝绳直径:   {self.config['d']*1000:.1f} mm")
        print(f"  钢丝绳总长:   {np.sum(self.l0):.1f} m")
        print(f"  重物质量:     {self.payload_total_mass:.1f} kg (PCH + 工具串)")
        print(f"  水深:         {self.config['total_water_depth']:.1f} m")
        print(f"  PCH高度:      {self.pch_height:.2f} m")
        print(f"  PCH底径:      {self.pch_base_diameter:.2f} m")
        print(f"  工具串长度:   {self.tool_length:.2f} m (4段x5m)")
        print(f"  工具串直径:   {self.tool_diameter:.2f} m")

        print("\n" + "="*80 + "\n")

        if hasattr(self, 'energy_jumps') and len(self.energy_jumps) > 0:
            max_rel_dE = max([jump['rel_dE'] for jump in self.energy_jumps])
            sum_abs_dE = sum([abs(jump['dE']) for jump in self.energy_jumps])
            print(f"【数值能量跳变 (Numerical Energy Jumps)】")
            print(f"  累计加段机械能变动: {sum_abs_dE:>10.2f} J")
            print(f"  最大单次相对能量跳变: {max_rel_dE*100:>8.4f} % (用于验证等几何插点稳定性)")
            print("\n" + "="*80 + "\n")

        # 返回关键数据
        summary = {
            'sum_energy_jump': sum_abs_dE if hasattr(self, 'energy_jumps') else 0.0,
            'max_relative_energy_jump_percent': max_rel_dE * 100 if hasattr(self, 'energy_jumps') else 0.0,
            'final_position': final_payload_pos,
            'final_height_above_seabed': final_height,
            'final_pch_tilt_deg': final_pch_tilt,
            'final_horizontal_offset': final_h_offset,
            'final_tool_tilt_deg': final_tool_tilt,
            # 平均值指标（推荐使用，消除振荡噪声）
            'avg_pch_tilt_deg': avg_pch_tilt,
            'avg_horizontal_offset': avg_h_offset,
            'avg_tool_tilt_deg': avg_tool_tilt,
            'max_stress_MPa': global_max_stress / 1e6,
            'max_stress_MPa_filtered': global_max_stress_filtered / 1e6,
            'max_stress_time': max_stress_time,
            'max_stress_time_filtered': max_stress_time_filtered,
            'max_stress_segment': max_stress_segment,
            'deployment_complete_time': deployment_complete_time,
            'target_reached': self.target_reached,
            'target_reach_time': self.target_reach_time,
            'total_simulation_time': times[-1],
            'total_segments': self.current_segments,
            'safety_factor': safety_factor,
            'safety_factor_raw': safety_factor_raw,
            'safety_factor_filtered': safety_factor_filtered,
            'safety_factor_basis': safety_factor_basis,
        }

        return summary

    def _collect_deployment_summary_metrics(self, solution=None):
        solution = self.full_solution if solution is None else solution
        if solution is None:
            return None
        times = np.asarray(solution.t, dtype=float); Y = solution.y
        N_of_t = getattr(solution, 'N_of_t', np.full(times.shape, self.config['N'], dtype=int))
        final_pos = Y[:3 * int(N_of_t[-1]), -1].reshape(int(N_of_t[-1]), 3)
        final_payload_pos = final_pos[-1]
        final_pch_tilt, final_h_offset, final_tool_tilt = self.compute_pch_tilt_and_offset(final_pos)
        final_height = self.compute_height_above_seabed(final_pos)
        avg_h_offset, avg_pch_tilt, avg_tool_tilt = self.get_convergent_offset(solution, max(30.0, 3 * self.config.get('wave_period', 10.0)))
        raw_max_stress, _, _, _, _ = self._compute_stress_strain_history(times, Y, N_of_t)
        filtered_max_stress = self._engineering_filter_series(times, raw_max_stress)
        if len(raw_max_stress):
            global_max_stress = float(np.max(raw_max_stress)); max_idx = int(np.argmax(raw_max_stress)); max_stress_time = float(times[max_idx])
            pos = Y[:3 * int(N_of_t[max_idx]), max_idx].reshape(int(N_of_t[max_idx]), 3)
            T = self._compute_tension(*self._compute_segment_vectors(pos), max_stress_time)
            max_stress_segment = int(np.argmax(np.linalg.norm(T, axis=1) / self.A)) + 1 if T.size > 0 else 1
            global_max_stress_filtered = float(np.max(filtered_max_stress))
            max_stress_time_filtered = float(times[int(np.argmax(filtered_max_stress))])
        else:
            global_max_stress = max_stress_time = 0.0; max_stress_segment = 1
            global_max_stress_filtered = max_stress_time_filtered = 0.0

        configured_basis = self.config.get('summary_safety_factor_basis', 'raw')
        if configured_basis != 'raw':
            self._log(
                f"提示：summary_safety_factor_basis={configured_basis!r} 已忽略；"
                "设计校核固定使用原始峰值。"
            )
        safety_factor_basis = 'raw'
        safety_stress, safety_label = global_max_stress, '原始峰值'
        return {
            'times': times, 'final_pos': final_pos, 'final_payload_pos': final_payload_pos,
            'final_pch_tilt': final_pch_tilt, 'final_h_offset': final_h_offset, 'final_tool_tilt': final_tool_tilt,
            'final_height': final_height, 'avg_h_offset': avg_h_offset, 'avg_pch_tilt': avg_pch_tilt,
            'avg_tool_tilt': avg_tool_tilt, 'raw_max_stress': raw_max_stress, 'global_max_stress': global_max_stress,
            'global_max_stress_filtered': global_max_stress_filtered, 'max_stress_time': max_stress_time,
            'max_stress_time_filtered': max_stress_time_filtered, 'max_stress_segment': max_stress_segment,
            'deployment_complete_time': self.release_data['times'][-1] if len(self.release_data['times']) > 1 else 0.0,
            'safety_stress': safety_stress, 'safety_label': safety_label, 'safety_factor_basis': safety_factor_basis,
        }

    def _build_tension_history_bundle(self, times, Y, N_of_t, include_segment_details=False):
        max_tension_history, mean_tension_history = [], []
        tension_hist = top_tension_hist = segment_mid_z_hist = segment_stress_hist = None
        if include_segment_details:
            tension_hist, top_tension_hist, segment_mid_z_hist, segment_stress_hist = [], [], [], []
        for i, t in enumerate(times):
            pos = Y[:3 * int(N_of_t[i]), i].reshape(int(N_of_t[i]), 3)
            T = self._compute_tension(*self._compute_segment_vectors(pos), t)
            Tmag = np.linalg.norm(T, axis=1) if T.size else np.array([])
            max_tension_history.append(float(Tmag.max()) if Tmag.size else 0.0)
            mean_tension_history.append(float(Tmag.mean()) if Tmag.size else 0.0)
            if include_segment_details:
                stress = self.compute_stress_from_tension(T)
                tension_hist.append(T.copy()); segment_stress_hist.append(stress.copy())
                top_tension_hist.append(float(Tmag[0]) if T.size > 0 else 0.0)
                segment_mid_z_hist.append(0.5 * (pos[:-1, 2] + pos[1:, 2]) if T.size > 0 else np.array([]))
        bundle = {'max_tension_history': np.asarray(max_tension_history, dtype=float), 'mean_tension_history': np.asarray(mean_tension_history, dtype=float)}
        if include_segment_details:
            bundle.update({'tension_hist': tension_hist, 'top_tension_hist': np.asarray(top_tension_hist, dtype=float), 'segment_mid_z_hist': segment_mid_z_hist, 'segment_stress_hist': segment_stress_hist})
        return bundle

    def export_data_to_excel(self, filename='simulation_data.xlsx', solution=None,
                             filter_enabled=None):
        """
        导出所有图像数据到Excel文件
        
        参数:
            filename: 输出文件名
            solution: 求解结果，默认使用self.full_solution
        
        导出内容:
            - Sheet1: 重物位置随时间变化 (Time, X, Y, Z, Depth)
            - Sheet2: PCH参数随时间变化 (Time, PCH_Tilt_deg, Horizontal_Offset_m, Height_Above_Seabed_m, Tool_Tilt_deg)
            - Sheet3: 应力应变随时间变化 (Time, Max_Stress_MPa, Mean_Stress_MPa, Top_Stress_MPa, Max_Strain, Mean_Strain)
            - Sheet4: 张力随时间变化 (Time, Max_Tension_kN, Mean_Tension_kN)
            - Sheet5: 摘要信息 (Summary)
            - If filter enabled, Sheet3 adds *_Smoothed columns
        """
        if solution is None:
            solution = self.full_solution
        if solution is None:
            print("错误：没有可用的解决方案")
            return
        
        print(f"\n正在导出数据到 {filename}...")
        
        times = solution.t
        Y = solution.y
        N_of_t = getattr(solution, 'N_of_t', np.full(times.shape, self.config['N'], dtype=int))
        if filter_enabled is None:
            filter_enabled = self.config.get('stress_filter_enabled', False)
        
        # ========== Sheet 1: 重物位置随时间变化 ==========
        xs = np.zeros_like(times)
        ys = np.zeros_like(times)
        zs = np.zeros_like(times)
        for i, Ni in enumerate(N_of_t):
            xs[i] = Y[3*(Ni-1), i]
            ys[i] = Y[3*(Ni-1)+1, i]
            zs[i] = Y[3*(Ni-1)+2, i]
        
        df_position = pd.DataFrame({
            'Time_s': times,
            'X_m': xs,
            'Y_m': ys,
            'Z_m': zs,
            'Depth_m': -zs
        })
        
        # ========== Sheet 2: PCH参数随时间变化 ==========
        pch_times = np.array(self.release_data['times'])
        pch_tilt = np.array(self.release_data['pch_tilt_deg'])
        h_offset = np.array(self.release_data['pch_horizontal_offset'])
        height_above = np.array(self.release_data['height_above_seabed'])
        tool_tilt = np.array(self.release_data['tool_tilt_deg'])
        
        df_pch = pd.DataFrame({
            'Time_s': pch_times,
            'PCH_Tilt_deg': pch_tilt,
            'Horizontal_Offset_m': h_offset,
            'Height_Above_Seabed_m': height_above,
            'Tool_Tilt_deg': tool_tilt
        })

        # ========== Sheet 3: 应力应变随时间变化 ==========
        raw_max_stress, raw_mean_stress, raw_top_stress, raw_max_strain, raw_mean_strain = (
            self._compute_stress_strain_history(times, Y, N_of_t)
        )

        df_stress_strain = pd.DataFrame({
            'Time_s': times,
            'Max_Stress_MPa': raw_max_stress / 1e6,
            'Mean_Stress_MPa': raw_mean_stress / 1e6,
            'Top_Stress_MPa': raw_top_stress / 1e6,
            'Max_Strain': raw_max_strain,
            'Mean_Strain': raw_mean_strain
        })

        if filter_enabled:
            smooth_max_stress = self._engineering_filter_series(times, raw_max_stress)
            smooth_mean_stress = self._engineering_filter_series(times, raw_mean_stress)
            smooth_top_stress = self._engineering_filter_series(times, raw_top_stress)
            smooth_max_strain = self._engineering_filter_series(times, raw_max_strain, nonnegative=False)
            smooth_mean_strain = self._engineering_filter_series(times, raw_mean_strain, nonnegative=False)

            df_stress_strain['Max_Stress_MPa_Smoothed'] = smooth_max_stress / 1e6
            df_stress_strain['Mean_Stress_MPa_Smoothed'] = smooth_mean_stress / 1e6
            df_stress_strain['Top_Stress_MPa_Smoothed'] = smooth_top_stress / 1e6
            df_stress_strain['Max_Strain_Smoothed'] = smooth_max_strain
            df_stress_strain['Mean_Strain_Smoothed'] = smooth_mean_strain

        # ========== Sheet 4: 张力随时间变化 ==========
        tension_bundle = self._build_tension_history_bundle(times, Y, N_of_t, include_segment_details=False)
        max_tension_history = tension_bundle['max_tension_history']
        mean_tension_history = tension_bundle['mean_tension_history']
        
        df_tension = pd.DataFrame({
            'Time_s': times,
            'Max_Tension_kN': max_tension_history / 1000,
            'Mean_Tension_kN': mean_tension_history / 1000
        })
        
        # ========== Sheet 5: 摘要信息 ==========
        metrics = self._collect_deployment_summary_metrics(solution)
        if metrics is None:
            print("错误：没有可用的解决方案")
            return

        final_payload_pos = metrics['final_payload_pos']
        final_height = metrics['final_height']
        final_pch_tilt = metrics['final_pch_tilt']
        final_h_offset = metrics['final_h_offset']
        final_tool_tilt = metrics['final_tool_tilt']
        global_max_stress = metrics['global_max_stress']
        max_stress_time = metrics['max_stress_time']
        
        summary_data = {
            '参数': [
                '最终X位置 (m)',
                '最终Y位置 (m)',
                '最终Z位置 (m)',
                '最终深度 (m)',
                '距海底高度 (m)',
                'PCH倾角 (度)',
                '水平偏移 (m)',
                '工具串倾角 (度)',
                '最大应力 (MPa)',
                '最大应力时刻 (s)',
                '下放段数',
                '单段长度 (m)',
                '下放速度 (m/s)',
                '总模拟时间 (s)',
                '钢丝绳直径 (mm)',
                '重物质量 (kg)',
                '水深 (m)',
                'PCH高度 (m)',
                'PCH底径 (m)',
                '工具串长度 (m)',
                '工具串直径 (m)'
            ],
            '数值': [
                final_payload_pos[0],
                final_payload_pos[1],
                final_payload_pos[2],
                -final_payload_pos[2],
                final_height,
                final_pch_tilt,
                final_h_offset,
                final_tool_tilt,
                global_max_stress / 1e6,
                max_stress_time,
                self.current_segments,
                self.segment_length,
                self.release_speed,
                times[-1],
                self.config['d'] * 1000,
                self.payload_total_mass,
                self.config['total_water_depth'],
                self.pch_height,
                self.pch_base_diameter,
                self.tool_length,
                self.tool_diameter
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        
        # ========== 写入Excel文件 ==========
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df_position.to_excel(writer, sheet_name='重物位置', index=False)
                df_pch.to_excel(writer, sheet_name='PCH参数', index=False)
                df_stress_strain.to_excel(writer, sheet_name='应力应变', index=False)
                df_tension.to_excel(writer, sheet_name='张力', index=False)
                df_summary.to_excel(writer, sheet_name='摘要', index=False)
            
            print(f"✓ 数据已成功导出到: {filename}")
            print(f"  - Sheet 1: 重物位置 ({len(df_position)} 行)")
            print(f"  - Sheet 2: PCH参数 ({len(df_pch)} 行)")
            print(f"  - Sheet 3: 应力应变 ({len(df_stress_strain)} 行)")
            print(f"  - Sheet 4: 张力 ({len(df_tension)} 行)")
            print(f"  - Sheet 5: 摘要 ({len(df_summary)} 行)")
            
        except Exception as e:
            print(f"✗ 导出失败: {e}")
            print("  提示: 请确保已安装 openpyxl 库 (pip install openpyxl)")

    def _smooth_series_for_plot(self, values, window):
        """简单滑动平均，仅用于汇报图的视觉平滑。"""
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return arr

        window = max(1, int(window))
        if window == 1:
            return arr.copy()

        return pd.Series(arr).rolling(window=window, center=True, min_periods=1).mean().to_numpy()

    def _engineering_filter_series(self, times, values, nonnegative=True):
        """
        工程绘图滤波：去除数值尖峰并保留波浪尺度响应。

        该函数只用于展示和汇报曲线，设计峰值/安全系数仍使用原始时程。
        """
        arr = np.asarray(values, dtype=float)
        times = np.asarray(times, dtype=float)
        if arr.size == 0 or arr.size != times.size:
            return arr.copy()

        finite = np.isfinite(arr)
        if not np.any(finite):
            return arr.copy()

        work = arr.copy()
        if not np.all(finite):
            work[~finite] = np.interp(times[~finite], times[finite], work[finite])

        diffs = np.diff(times)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size == 0:
            return work

        dt = float(np.median(diffs))
        fs = 1.0 / dt
        median_window_s = float(self.config.get('stress_filter_median_window_s', 1.5))
        median_window = max(3, int(round(median_window_s / dt)))
        if median_window % 2 == 0:
            median_window += 1

        filtered = (
            pd.Series(work)
            .rolling(window=median_window, center=True, min_periods=1)
            .median()
            .to_numpy()
        )

        wave_period = max(float(self.config.get('wave_period', 11.0)), 1e-6)
        cutoff_wave_multiple = float(self.config.get('stress_filter_cutoff_wave_multiple', 4.0))
        cutoff_hz = float(self.config.get('stress_filter_cutoff_hz', 0.45))
        cutoff = min(cutoff_wave_multiple / wave_period, cutoff_hz, 0.4 * fs)
        if cutoff <= 0.0 or cutoff >= 0.5 * fs or filtered.size <= 18:
            return np.maximum(filtered, 0.0) if nonnegative else filtered

        import scipy.signal as signal

        sos = signal.butter(4, cutoff, btype='lowpass', fs=fs, output='sos')
        filtered = signal.sosfiltfilt(sos, filtered)
        return np.maximum(filtered, 0.0) if nonnegative else filtered

    def _estimate_static_top_tension_history(self, solution):
        """
        估算静水顶张力。

        这里用当前几何弧长乘以单位湿重，再叠加重物单元湿重，
        只用于给出一条平滑上升的“静水重量”参考线。
        """
        times = solution.t
        Y = solution.y
        N_of_t = getattr(solution, 'N_of_t', np.full(times.shape, self.config['N'], dtype=int))

        static_top_tension = np.zeros_like(times, dtype=float)
        for i, Ni in enumerate(N_of_t):
            pos = Y[:3 * int(Ni), i].reshape(int(Ni), 3)
            static_top_tension[i] = self._estimate_static_top_tension_from_positions(pos)

        smooth_window = max(5, (len(times) // 120) * 2 + 1)
        return self._smooth_series_for_plot(static_top_tension, smooth_window)

    def _compute_sci_plot_data(self, solution=None, depth_bin_count=220):
        """重构 SCI 可视化所需的连续时程和包络量。"""
        if solution is None:
            solution = self.full_solution
        if solution is None:
            print("错误：没有可用的解决方案")
            return None

        cache_token = (id(solution), len(solution.t), int(depth_bin_count))
        if self._sci_plot_cache is not None and self._sci_plot_cache.get('token') == cache_token:
            return self._sci_plot_cache

        times = np.asarray(solution.t, dtype=float)
        Y = solution.y
        N_of_t = getattr(solution, 'N_of_t', np.full(times.shape, self.config['N'], dtype=int))

        n_times = len(times)
        top_xyz = Y[0:3, :].T.copy()
        vessel_xyz = np.array([self.vessel.get_vessel_position(t) for t in times], dtype=float)
        positions_history = []
        pch_node_xyz = np.zeros((n_times, 3), dtype=float)
        payload_ref_xyz = np.zeros((n_times, 3), dtype=float)
        pch_tilt = np.zeros(n_times, dtype=float)
        tool_tilt = np.zeros(n_times, dtype=float)
        height_above_seabed = np.zeros(n_times, dtype=float)

        for i, Ni in enumerate(N_of_t):
            pos = Y[:3 * int(Ni), i].reshape(int(Ni), 3)
            positions_history.append(pos.copy())
            pch_node_xyz[i] = pos[-1]
            payload_ref_xyz[i] = self.compute_payload_reference_point(pos)
            pch_tilt_i, _, tool_tilt_i = self.compute_pch_tilt_and_offset(pos)
            pch_tilt[i] = pch_tilt_i
            tool_tilt[i] = tool_tilt_i
            height_above_seabed[i] = self.compute_height_above_seabed(pos)

        tension_hist = getattr(self, '_hist_tension', None)
        top_tension_hist = getattr(self, '_hist_top_tension', None)
        segment_mid_z_hist = getattr(self, '_hist_segment_mid_z', None)
        segment_stress_hist = getattr(self, '_hist_segment_stress', None)

        if tension_hist is None or len(tension_hist) != n_times:
            tension_bundle = self._build_tension_history_bundle(times, Y, N_of_t, include_segment_details=True)
            tension_hist = tension_bundle['tension_hist']
            top_tension_hist = tension_bundle['top_tension_hist']
            segment_mid_z_hist = tension_bundle['segment_mid_z_hist']
            segment_stress_hist = tension_bundle['segment_stress_hist']

        top_tension = np.asarray(top_tension_hist, dtype=float)
        static_top_tension = self._estimate_static_top_tension_history(solution)

        release_speed_cmd_hist = getattr(self, '_hist_release_speed_cmd', None)
        if release_speed_cmd_hist is not None and len(release_speed_cmd_hist) == n_times:
            release_speed_cmd = np.asarray(release_speed_cmd_hist, dtype=float)
        else:
            release_speed_cmd = np.full(n_times, float(self.base_release_speed), dtype=float)

        control_target_hist = getattr(self, '_hist_tension_control_target', None)
        if control_target_hist is not None and len(control_target_hist) == n_times:
            tension_control_target = np.asarray(control_target_hist, dtype=float)
        elif self.atc_enabled:
            tension_control_target = self.tension_control_target_scale * static_top_tension
        else:
            tension_control_target = np.full(n_times, np.nan, dtype=float)

        control_error_hist = getattr(self, '_hist_tension_control_error', None)
        if control_error_hist is not None and len(control_error_hist) == n_times:
            tension_control_error = np.asarray(control_error_hist, dtype=float)
        elif np.all(np.isfinite(tension_control_target)):
            tension_control_error = top_tension - tension_control_target
        else:
            tension_control_error = np.full(n_times, np.nan, dtype=float)

        if self.atc_enabled:
            top_tension_controlled = top_tension.copy()
            top_tension_open_loop = np.full(n_times, np.nan, dtype=float)
        else:
            top_tension_controlled = np.full(n_times, np.nan, dtype=float)
            top_tension_open_loop = top_tension.copy()

        z_upper = max(float(np.max(top_xyz[:, 2])), 0.0)
        z_lower = float(self.config['seabed_depth'])
        depth_bins = np.linspace(z_lower, z_upper, int(depth_bin_count) + 1)
        depth_centers = 0.5 * (depth_bins[:-1] + depth_bins[1:])
        max_tension_envelope = np.full(depth_centers.shape, np.nan)
        max_stress_envelope = np.full(depth_centers.shape, np.nan)

        for T, seg_mid_z, stress in zip(tension_hist, segment_mid_z_hist, segment_stress_hist):
            if T.size == 0 or len(seg_mid_z) == 0:
                continue
            Tmag = np.linalg.norm(T, axis=1)
            bin_indices = np.clip(np.digitize(seg_mid_z, depth_bins) - 1, 0, len(depth_centers) - 1)
            for idx_seg, idx_bin in enumerate(bin_indices):
                t_val = Tmag[idx_seg]
                s_val = stress[idx_seg]
                if np.isnan(max_tension_envelope[idx_bin]) or t_val > max_tension_envelope[idx_bin]:
                    max_tension_envelope[idx_bin] = t_val
                if np.isnan(max_stress_envelope[idx_bin]) or s_val > max_stress_envelope[idx_bin]:
                    max_stress_envelope[idx_bin] = s_val

        flow_ref = self.current_velocity(np.array([0.0, 0.0, 0.5 * self.config['seabed_depth']]))[:2]
        if np.linalg.norm(flow_ref) < 1e-12:
            flow_dir_unit = np.array([1.0, 0.0])
        else:
            flow_dir_unit = flow_ref / np.linalg.norm(flow_ref)

        snapshot_targets = np.linspace(0.08, 0.95, 10)
        snapshot_indices = np.unique(np.clip((snapshot_targets * max(n_times - 1, 0)).astype(int), 0, max(n_times - 1, 0)))

        self._sci_plot_cache = {
            'token': cache_token,
            'times': times,
            'top_xyz': top_xyz,
            'vessel_xyz': vessel_xyz,
            'positions_history': positions_history,
            'pch_node_xyz': pch_node_xyz,
            'payload_ref_xyz': payload_ref_xyz,
            'pch_tilt': pch_tilt,
            'tool_tilt': tool_tilt,
            'height_above_seabed': height_above_seabed,
            'top_tension': top_tension,
            'static_top_tension': static_top_tension,
            'top_tension_controlled': top_tension_controlled,
            'top_tension_open_loop': top_tension_open_loop,
            'tension_control_target': tension_control_target,
            'tension_control_error': tension_control_error,
            'release_speed_command': release_speed_cmd,
            'depth_centers': depth_centers,
            'max_tension_envelope': max_tension_envelope,
            'max_stress_envelope': max_stress_envelope,
            'flow_dir_unit': flow_dir_unit,
            'snapshot_indices': snapshot_indices,
        }
        return self._sci_plot_cache

    def export_visualization_payload(self, output_path=None, solution=None, depth_bin_count=220):
        """导出独立绘图脚本所需数据到 NPZ 文件。"""
        if solution is None:
            solution = self.full_solution
        if solution is None:
            print("错误：没有可用的解决方案")
            return None

        plot_data = self._compute_sci_plot_data(solution=solution, depth_bin_count=depth_bin_count)
        if plot_data is None:
            return None

        times = np.asarray(plot_data['times'], dtype=float)
        Y = solution.y
        N_of_t = getattr(solution, 'N_of_t', np.full(times.shape, self.config['N'], dtype=int))
        raw_max_stress, raw_mean_stress, raw_top_stress, raw_max_strain, raw_mean_strain = (
            self._compute_stress_strain_history(times, Y, N_of_t)
        )
        filtered_top_tension = self._engineering_filter_series(times, plot_data['top_tension'])
        filtered_top_tension_controlled = self._engineering_filter_series(times, plot_data['top_tension_controlled'])
        filtered_top_tension_open_loop = self._engineering_filter_series(times, plot_data['top_tension_open_loop'])
        filtered_max_stress = self._engineering_filter_series(times, raw_max_stress)
        filtered_mean_stress = self._engineering_filter_series(times, raw_mean_stress)

        if output_path is None:
            output_path = os.path.join(os.getcwd(), 'results', 'visualization', 'visualization_payload.npz')

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        positions_obj = np.empty(len(plot_data['positions_history']), dtype=object)
        for i, pos in enumerate(plot_data['positions_history']):
            positions_obj[i] = np.asarray(pos, dtype=float)

        np.savez_compressed(
            output_path,
            times=times,
            top_xyz=np.asarray(plot_data['top_xyz'], dtype=float),
            vessel_xyz=np.asarray(plot_data['vessel_xyz'], dtype=float),
            positions_history=positions_obj,
            pch_node_xyz=np.asarray(plot_data['pch_node_xyz'], dtype=float),
            payload_ref_xyz=np.asarray(plot_data['payload_ref_xyz'], dtype=float),
            pch_tilt=np.asarray(plot_data['pch_tilt'], dtype=float),
            tool_tilt=np.asarray(plot_data['tool_tilt'], dtype=float),
            height_above_seabed=np.asarray(plot_data['height_above_seabed'], dtype=float),
            top_tension=np.asarray(plot_data['top_tension'], dtype=float),
            filtered_top_tension=np.asarray(filtered_top_tension, dtype=float),
            static_top_tension=np.asarray(plot_data['static_top_tension'], dtype=float),
            top_tension_controlled=np.asarray(plot_data['top_tension_controlled'], dtype=float),
            filtered_top_tension_controlled=np.asarray(filtered_top_tension_controlled, dtype=float),
            top_tension_open_loop=np.asarray(plot_data['top_tension_open_loop'], dtype=float),
            filtered_top_tension_open_loop=np.asarray(filtered_top_tension_open_loop, dtype=float),
            tension_control_target=np.asarray(plot_data['tension_control_target'], dtype=float),
            tension_control_error=np.asarray(plot_data['tension_control_error'], dtype=float),
            release_speed_command=np.asarray(plot_data['release_speed_command'], dtype=float),
            depth_centers=np.asarray(plot_data['depth_centers'], dtype=float),
            max_tension_envelope=np.asarray(plot_data['max_tension_envelope'], dtype=float),
            max_stress_envelope=np.asarray(plot_data['max_stress_envelope'], dtype=float),
            flow_dir_unit=np.asarray(plot_data['flow_dir_unit'], dtype=float),
            snapshot_indices=np.asarray(plot_data['snapshot_indices'], dtype=int),
            raw_max_stress=np.asarray(raw_max_stress, dtype=float),
            raw_mean_stress=np.asarray(raw_mean_stress, dtype=float),
            raw_top_stress=np.asarray(raw_top_stress, dtype=float),
            filtered_max_stress=np.asarray(filtered_max_stress, dtype=float),
            filtered_mean_stress=np.asarray(filtered_mean_stress, dtype=float),
            raw_max_strain=np.asarray(raw_max_strain, dtype=float),
            raw_mean_strain=np.asarray(raw_mean_strain, dtype=float),
            node_counts=np.asarray(N_of_t, dtype=int),
            release_times=np.asarray(self.release_data.get('times', []), dtype=float),
            seabed_z=float(self.config['seabed_depth']),
            target_height=float(self.config['target_height_above_seabed']),
            target_z=float(self.config['seabed_depth'] + self.config['target_height_above_seabed']),
            wave_period=float(self.config.get('wave_period', 10.0)),
            active_tension_control_enabled=bool(self.atc_enabled),
            release_speed_nominal=float(self.base_release_speed),
            tension_control_target_scale=float(self.tension_control_target_scale),
            target_reach_time=(
                np.nan if self.target_reach_time is None else float(self.target_reach_time)
            ),
        )

        print(f"✓ 可视化数据包已导出: {output_path}")
        return output_path


def run_dynamic_3d_simulation():
    """
    运行三维连续动态下放模拟。

    本入口不再做任何参数覆盖。
    需要修改工况时，请直接更新 DEFAULT_CONFIG。
    """
    # 使用工程主版本默认配置（不做覆盖）
    config = DEFAULT_CONFIG.copy()

    print("\n三维连续动态下放系统初始化（带PCH细化和25m停止）...")
    system = DynamicWireRopeSystem3D(config)
    output_dir = os.path.join(os.getcwd(), 'results', 'visualization')
    os.makedirs(output_dir, exist_ok=True)

    # 计算总时间（足够长，但会在25m处自动停止）
    per_segment_T = system.release_duration
    t_end = per_segment_T * system.total_segments + 100  # 额外100s用于稳定

    # 执行模拟
    sol = system.simulate_dynamic_release(t_end=t_end, dt_output=0.2)

    # 打印总结信息
    summary = system.print_deployment_summary(sol)
    
    # 导出数据到Excel
    system.export_data_to_excel(os.path.join(output_dir, 'simulation_data_original.xlsx'), sol)

    # 导出外部绘图脚本所需数据包
    system.export_visualization_payload(
        output_path=os.path.join(output_dir, 'visualization_payload.npz'),
        solution=sol,
    )

    # 计算工程动力学指标
    system.compute_engineering_metrics()

    print("\n[完成] 三维连续动态下放模拟完成！")
    print("\n提示：图像已拆分为 scripts/visualization/ 目录下独立脚本，请仿真后分别运行。")

    return system, sol, summary


if __name__ == "__main__":
    run_dynamic_3d_simulation()


