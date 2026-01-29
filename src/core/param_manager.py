# src/core/param_manager.py

import sys
import os
import torch
import yaml
import joblib
import logging
import numpy as np
from typing import Dict, Tuple, Any

# 配置日志
logger = logging.getLogger(__name__)

class ParamManager:
    """
    参数管理器 (Parameter Manager)
    
    职责：
    1. [IO] 直接加载兄弟项目的 .joblib 预处理文件，解析归一化参数。
    2. [Math] 将物理参数映射到模型归一化空间 (Physical -> Normalized)。
    3. [Tensor] 维护 Parameter/Buffer，支持全流程可微计算。
    4. [Assembly] 将 Action (优化变量) 与 State (工况变量) 拼装为模型输入。
    """

    def __init__(self, 
                 param_space_path: str, 
                 preprocessor_path: str,
                 surrogate_project_dir: str,
                 device: str = "cpu"):
        """
        Args:
            param_space_path: param_space.yaml 的路径
            preprocessor_path: preprocessors.joblib 的路径
            surrogate_project_dir: 损伤预测项目的根目录 (用于import类定义)
            device: 计算设备 ('cpu' or 'cuda')
        """
        self.device = device
        
        # 1. 环境准备：挂载兄弟项目路径以导入 DataProcessor 类
        self._setup_imports(surrogate_project_dir)
        
        # 2. 加载配置与预处理器
        self.params_config = self._load_yaml(param_space_path)
        self.processor = self._load_joblib(preprocessor_path)
        
        # 3. 解析参数角色 (Control vs State)
        self._parse_param_roles()
        
        # 4. 提取归一化参数并转为 Tensor
        self._build_normalization_tensors()
        
        # 5. 打印详细信息
        self.print_info()

    def _setup_imports(self, project_dir: str):
        """将兄弟项目加入 sys.path，防止 joblib 加载失败"""
        abs_path = os.path.abspath(project_dir)
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)
            logger.info(f"Added {abs_path} to sys.path for class definitions.")
        
        try:
            # 尝试导入，验证环境是否这就绪
            # 注意：根据 dataset_prepare.py，DataProcessor 在 utils.dataset_prepare 中
            from utils.dataset_prepare import DataProcessor
        except ImportError as e:
            logger.error(f"Failed to import DataProcessor from {abs_path}. "
                         f"Ensure 'utils' package exists and contains dataset_prepare.py")
            raise e

    def _load_yaml(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _load_joblib(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessor file not found at: {path}")
        logger.info(f"Loading preprocessor from: {path}")
        return joblib.load(path)

    def _parse_param_roles(self):
        """解析 param_space.yaml，区分可调参数和固定工况"""
        self.param_definitions = self.params_config['parameters']
        
        # 存储索引和边界
        self.control_indices = []
        self.state_indices = []
        self.control_names = []
        
        # 物理边界 (用于优化器的截断和归一化)
        control_mins = []
        control_maxs = []

        for p in self.param_definitions:
            idx = p['index']
            # 连续参数: 0-10, 离散参数: 11-12
            if p['role'] == 'control' and p.get('trainable', True):
                self.control_indices.append(idx)
                self.control_names.append(p['name'])
                control_mins.append(p['min'])
                control_maxs.append(p['max'])
            else:
                self.state_indices.append(idx)
        
        # 转为 Tensor
        self.control_min_t = torch.tensor(control_mins, device=self.device, dtype=torch.float32)
        self.control_max_t = torch.tensor(control_maxs, device=self.device, dtype=torch.float32)

    def _build_normalization_tensors(self):
        """
        核心逻辑：从 sklearn 对象中提取参数，构建统一的 (x - offset) * scale 计算图
        """
        proc = self.processor
        
        # --- A. 波形归一化 ---
        self.waveform_factor = float(proc.waveform_norm_factor)
        
        # --- B. 连续特征归一化 (Indices 0-10) ---
        # 目标：构建 shape=(11,) 的 offset 和 scale 向量
        num_continuous = 11
        self.cont_offset = torch.zeros(num_continuous, device=self.device, dtype=torch.float32)
        self.cont_scale = torch.ones(num_continuous, device=self.device, dtype=torch.float32)
        
        # 1. 提取 MinMaxScaler 参数 (X_std = (X - min) / (max - min))
        # 对应关系: proc.minmax_indices_in_continuous -> proc.scaler_minmax.data_min_
        min_vals = proc.scaler_minmax.data_min_
        max_vals = proc.scaler_minmax.data_max_
        
        for i, global_idx in enumerate(proc.minmax_indices_in_continuous):
            d_min = float(min_vals[i])
            d_max = float(max_vals[i])
            scale = d_max - d_min
            if abs(scale) < 1e-8: scale = 1.0
            
            self.cont_offset[global_idx] = d_min
            self.cont_scale[global_idx] = 1.0 / scale
            
        # 2. 提取 MaxAbsScaler 参数 (X_std = X / max_abs)
        # 对应关系: proc.maxabs_indices_in_continuous -> proc.scaler_maxabs.max_abs_
        max_abs_vals = proc.scaler_maxabs.max_abs_
        
        for i, global_idx in enumerate(proc.maxabs_indices_in_continuous):
            m_abs = float(max_abs_vals[i])
            if abs(m_abs) < 1e-8: m_abs = 1.0
            
            self.cont_offset[global_idx] = 0.0 # MaxAbs 无 offset
            self.cont_scale[global_idx] = 1.0 / m_abs
            
        # --- C. 离散特征编码 (Indices 11, 12) ---
        # 建立查表字典: {global_index: {physical_val: encoded_val}}
        self.discrete_maps = {}
        for i, encoder in enumerate(proc.encoders_discrete):
            global_idx = proc.discrete_indices[i]
            # sklearn classes_ 是排序后的唯一值
            mapping = {val: idx for idx, val in enumerate(encoder.classes_)}
            self.discrete_maps[global_idx] = mapping

    def print_info(self):
        """打印加载摘要"""
        print("\n" + "="*50)
        print(f" [ParamManager] Initialization Summary")
        print("="*50)
        print(f" Device: {self.device}")
        print(f" Waveform Norm Factor: {self.waveform_factor:.4f}")
        
        print("\n --- Control Parameters (Trainable) ---")
        for i, name in enumerate(self.control_names):
            idx = self.control_indices[i]
            c_min = self.control_min_t[i].item()
            c_max = self.control_max_t[i].item()
            # 打印该参数对应的归一化系数，验证是否正确加载
            offset = self.cont_offset[idx].item()
            scale = self.cont_scale[idx].item()
            print(f"  [{idx:02d}] {name:<10}: Phys Range=[{c_min:.1f}, {c_max:.1f}] | Norm: (x - {offset:.2f}) * {scale:.4f}")
            
        print("\n --- Discrete Encoders ---")
        for idx, mapping in self.discrete_maps.items():
            print(f"  [{idx:02d}] Mapping Size: {len(mapping)} keys -> {list(mapping.keys())}")
        print("="*50 + "\n")

    # =========================================================
    #  核心功能 1: 向量拼接与标准化 (可微)
    # =========================================================

    def get_model_input(self, 
                        state_dict: Dict[str, float], 
                        action_tensor: torch.Tensor,
                        waveform_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state_dict: 单个工况物理值, e.g. {'impact_velocity': 56.0, 'OT': 2}
            action_tensor: (B, N_ctrl) 优化的控制参数物理值
            waveform_tensor: (B, 2, 150) 原始物理波形

        Returns:
            acc_norm: (B, 2, 150)
            cont_norm: (B, 11)
            disc_enc: (B, 2)
        """
        B = action_tensor.shape[0]
        
        # 1. 拼装连续特征矩阵 (B, 11)
        # 初始化全零矩阵
        continuous_phys = torch.zeros((B, 11), device=self.device, dtype=torch.float32)
        
        # A. 填入 State (State -> BATCH Copy)
        for p in self.param_definitions:
            idx = p['index']
            if idx < 11: # 只处理连续特征
                if idx in self.state_indices:
                    val = state_dict.get(p['name'])
                    if val is None: raise ValueError(f"Missing state: {p['name']}")
                    continuous_phys[:, idx] = val
                # 如果是 control, 下一步会覆盖
        
        # B. 填入 Action (Control -> BATCH Assign)
        for i, global_idx in enumerate(self.control_indices):
            # 确保 action 也是连续特征 (理论上是的，离散 control 暂不支持梯度优化)
            if global_idx < 11:
                continuous_phys[:, global_idx] = action_tensor[:, i]

        # 2. 连续特征归一化 (Tensor Vectorization)
        # cont_norm = (x - offset) * scale
        # 利用广播机制: (B, 11) - (11,) -> (B, 11)
        cont_norm = (continuous_phys - self.cont_offset) * self.cont_scale
        
        # 3. 拼装离散特征 (B, 2)
        # 离散特征通常是 State，暂不支持作为 Control 优化
        discrete_enc_list = []
        for global_idx in [11, 12]: # Hardcoded for this project structure
            # 查找物理值
            p_name = next(p['name'] for p in self.param_definitions if p['index'] == global_idx)
            phys_val = state_dict.get(p_name)
            
            # 查表编码
            mapping = self.discrete_maps[global_idx]
            if phys_val not in mapping:
                # 容错：如果找不到 float 的 key，尝试 int
                if int(phys_val) in mapping:
                    enc_val = mapping[int(phys_val)]
                else:
                    raise ValueError(f"Discrete value {phys_val} for {p_name} not found in encoder classes: {list(mapping.keys())}")
            else:
                enc_val = mapping[phys_val]
            
            discrete_enc_list.append(enc_val)
            
        # 转为 Tensor 并扩展 Batch
        disc_enc = torch.tensor(discrete_enc_list, device=self.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        
        # 4. 波形归一化
        acc_norm = waveform_tensor / self.waveform_factor
        
        return acc_norm, cont_norm, disc_enc

    # =========================================================
    #  核心功能 2: 优化空间映射 (Optimization Space Helpers)
    # =========================================================

    def normalize_action(self, action_phys: torch.Tensor) -> torch.Tensor:
        """物理值 -> [-1, 1] (用于优化器内部)"""
        return 2.0 * (action_phys - self.control_min_t) / (self.control_max_t - self.control_min_t) - 1.0

    def denormalize_action(self, action_opt: torch.Tensor) -> torch.Tensor:
        """[-1, 1] -> 物理值 (用于输入模型)"""
        # Clamp 确保不越界
        action_opt = torch.clamp(action_opt, -1.0, 1.0)
        return 0.5 * (action_opt + 1.0) * (self.control_max_t - self.control_min_t) + self.control_min_t

    def get_random_action(self, batch_size: int = 1) -> torch.Tensor:
        """生成随机物理参数"""
        rand_opt = torch.rand((batch_size, len(self.control_indices)), device=self.device) * 2 - 1 # [-1, 1]
        return self.denormalize_action(rand_opt)