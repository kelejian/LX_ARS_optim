# src/core/param_manager.py

import sys
import os
import torch
import yaml
import joblib
import logging
import numpy as np
from typing import Dict, Tuple, List, Any

# 配置日志
logger = logging.getLogger(__name__)

class ParamManager:
    """
    参数管理器 (Parameter Manager)
    
    主要功能：
    1. 解析 param_space.yaml 配置，建立物理参数名称与模型输入索引的映射关系。
    2. 加载外部预处理器 (preprocessor)，构建物理空间到模型归一化空间的转换逻辑。
    3. 提供支持批量化 (Batch) 的张量组装接口，将工况状态 (State) 与控制参数 (Action) 
       合并为下游模型可接受的输入张量。
    4. 维护控制参数的物理边界，支持优化过程中的截断 (Clamp) 操作。
    """

    def __init__(self, 
                 param_space_path: str, 
                 preprocessor_path: str,
                 surrogate_project_dir: str,
                 device: str = "cpu"):
        """
        初始化参数管理器。

        Args:
            param_space_path: 参数配置文件路径 (.yaml)
            preprocessor_path: 预处理器文件路径 (.joblib)
            surrogate_project_dir: 代理模型项目根目录 (用于导入相关类定义)
            device: 计算设备
        """
        self.device = device
        
        # 1. 环境准备与文件加载
        self._setup_imports(surrogate_project_dir)
        self.params_config = self._load_yaml(param_space_path)
        self.processor = self._load_joblib(preprocessor_path)
        
        # 2. 解析参数元数据
        self._parse_parameter_metadata()
        
        # 3. 构建归一化张量 (Scale & Offset)
        self._build_normalization_tensors()
        
        # 4. 构建优化边界张量
        self._build_optimization_bounds()
        
        self.print_info()

    def _setup_imports(self, project_dir: str):
        """将代理模型项目路径加入 sys.path 以便正确反序列化 joblib 对象。"""
        abs_path = os.path.abspath(project_dir)
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)
        
        try:
            from utils.dataset_prepare import DataProcessor
        except ImportError as e:
            logger.error(f"Failed to import DataProcessor from {abs_path}. Check project structure.")
            raise e

    def _load_yaml(self, path: str) -> Dict:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _load_joblib(self, path: str) -> Any:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessor file not found at: {path}")
        return joblib.load(path)

    def _parse_parameter_metadata(self):
        """
        根据配置文件建立参数索引映射。
        区分连续/离散变量，以及状态/控制变量。
        """
        definitions = self.params_config['parameters']
        
        # 初始化索引容器
        self.all_indices: List[int] = []
        self.cont_indices: List[int] = []  # 连续变量索引
        self.disc_indices: List[int] = []  # 离散变量索引
        self.control_indices: List[int] = [] # 待优化的控制参数索引
        
        # 映射表: name -> config dict
        self.param_map: Dict[str, dict] = {} 
        
        for p in definitions:
            name = p['name']
            idx = p['index']
            role = p['role']
            p_type = p['type']
            
            self.param_map[name] = p
            self.all_indices.append(idx)
            
            # 按类型分类
            if p_type == 'continuous':
                self.cont_indices.append(idx)
            elif p_type == 'discrete':
                self.disc_indices.append(idx)
            else:
                raise ValueError(f"Unsupported parameter type '{p_type}' for {name}")
                
            # 按角色分类：只有标记为 trainable 的 control 参数才会被加入优化列表
            if role == 'control' and p.get('trainable', True):
                self.control_indices.append(idx)
        
        # 转换为 LongTensor 以支持高级索引
        self.cont_indices_t = torch.tensor(self.cont_indices, device=self.device, dtype=torch.long)
        self.disc_indices_t = torch.tensor(self.disc_indices, device=self.device, dtype=torch.long)
        
        # 计算特征维度
        self.num_total = max(self.all_indices) + 1 if self.all_indices else 0
        
        logger.info(f"Param metadata parsed. Total features: {self.num_total}, Control vars: {len(self.control_indices)}")

    def _build_normalization_tensors(self):
        """
        从预处理器中提取均值和方差，构建可微的归一化计算图。
        支持 MinMaxScaler 和 MaxAbsScaler。
        """
        proc = self.processor
        self.waveform_factor = float(proc.waveform_norm_factor)
        
        # 初始化 offset 和 scale 张量 (默认不缩放)
        self.cont_offset = torch.zeros(self.num_total, device=self.device, dtype=torch.float32)
        self.cont_scale = torch.ones(self.num_total, device=self.device, dtype=torch.float32)
        
        # 处理 MinMaxScaler 对应的特征
        if hasattr(proc, 'minmax_indices_in_continuous'):
            min_vals = proc.scaler_minmax.data_min_
            max_vals = proc.scaler_minmax.data_max_
            for i, global_idx in enumerate(proc.minmax_indices_in_continuous):
                # 仅当该索引确实被定义为连续变量时才应用
                if global_idx in self.cont_indices:
                    d_min = float(min_vals[i])
                    d_max = float(max_vals[i])
                    scale = d_max - d_min
                    # 防止除以零
                    if abs(scale) < 1e-8: scale = 1.0
                    
                    self.cont_offset[global_idx] = d_min
                    self.cont_scale[global_idx] = 1.0 / scale

        # 处理 MaxAbsScaler 对应的特征
        if hasattr(proc, 'maxabs_indices_in_continuous'):
            max_abs_vals = proc.scaler_maxabs.max_abs_
            for i, global_idx in enumerate(proc.maxabs_indices_in_continuous):
                if global_idx in self.cont_indices:
                    m_abs = float(max_abs_vals[i])
                    if abs(m_abs) < 1e-8: m_abs = 1.0
                    
                    self.cont_offset[global_idx] = 0.0
                    self.cont_scale[global_idx] = 1.0 / m_abs

        # 构建离散特征编码表
        self.discrete_maps = {}
        if hasattr(proc, 'discrete_indices'):
            for i, global_idx in enumerate(proc.discrete_indices):
                if global_idx in self.disc_indices:
                    encoder = proc.encoders_discrete[i]
                    # 建立 {物理值: 编码值} 的映射
                    mapping = {val: idx for idx, val in enumerate(encoder.classes_)}
                    self.discrete_maps[global_idx] = mapping

    def _build_optimization_bounds(self):
        """
        提取待优化控制参数的物理边界，用于后续的反归一化和截断操作。
        """
        mins = []
        maxs = []
        for idx in self.control_indices:
            name = self._get_name_by_index(idx)
            p = self.param_map[name]
            mins.append(p['min'])
            maxs.append(p['max'])
            
        self.ctrl_min_t = torch.tensor(mins, device=self.device, dtype=torch.float32)
        self.ctrl_max_t = torch.tensor(maxs, device=self.device, dtype=torch.float32)

    def _get_name_by_index(self, idx: int) -> str:
        """通过索引反查参数名称"""
        for name, p in self.param_map.items():
            if p['index'] == idx: return name
        raise KeyError(f"Index {idx} not found in configuration.")

    def print_info(self):
        """打印参数管理器状态摘要"""
        print(f"\n[ParamManager] Configured for {self.num_total} input features.")
        print(f" - Continuous Indices: {self.cont_indices}")
        print(f" - Discrete Indices: {self.disc_indices}")
        print(f" - Control Indices (Optimization Targets): {self.control_indices}")

    # =========================================================
    #  核心功能: 批量化张量组装
    # =========================================================

    def create_base_params_tensor(self, state_dict_list: List[Dict[str, float]]) -> torch.Tensor:
        """
        将工况列表转换为基础物理参数张量。
        此函数处理所有 State 参数的赋值，并为 Control 参数填充默认值 (如果有)。
        
        Args:
            state_dict_list: 包含 N 个工况字典的列表
            
        Returns:
            base_tensor: (N, Total_Features) 物理值张量
        """
        B = len(state_dict_list)
        base_tensor = torch.zeros((B, self.num_total), device=self.device, dtype=torch.float32)
        
        # 遍历所有参数定义，从输入 dict 或默认配置中取值
        for name, config in self.param_map.items():
            idx = config['index']
            default_val = config.get('default', 0.0)
            
            # 逐个样本填充
            for i, state in enumerate(state_dict_list):
                if name in state:
                    base_tensor[i, idx] = float(state[name])
                else:
                    # 如果输入 dict 中缺失该参数，使用 config 中的 default
                    base_tensor[i, idx] = float(default_val)
                    
        return base_tensor

    def get_model_input_from_tensor(self, 
                                    base_params: torch.Tensor, 
                                    action_tensor: torch.Tensor,
                                    waveform_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        组装模型输入张量。
        将当前的控制参数 (Action) 覆盖到基础参数张量 (Base Params) 中，并执行归一化。
        
        Args:
            base_params: (B, Total_Features) 基础物理参数张量
            action_tensor: (B, N_Controls) 当前优化的控制参数物理值
            waveform_tensor: (B, 2, 150) 物理波形张量
            
        Returns:
            acc_norm: (B, 2, 150) 归一化后的波形
            cont_norm: (B, N_Cont) 归一化后的连续特征
            disc_enc: (B, N_Disc) 编码后的离散特征索引
        """
        # 1. 融合 Action：使用 clone 确保不修改原始 base_params
        current_phys = base_params.clone()
        
        # 将 Action 填入对应的 Control 列
        # 注意：这里假设 action_tensor 的列顺序与 self.control_indices 的顺序一致
        for i, global_idx in enumerate(self.control_indices):
            current_phys[:, global_idx] = action_tensor[:, i]
            
        # 2. 连续特征归一化
        # 提取连续特征列
        cont_phys = current_phys[:, self.cont_indices_t] # (B, N_Cont)
        
        # 获取对应的 offset/scale 并执行线性变换
        c_offset = self.cont_offset[self.cont_indices_t]
        c_scale = self.cont_scale[self.cont_indices_t]
        cont_norm = (cont_phys - c_offset) * c_scale
        
        # 3. 离散特征编码
        # 提取离散特征列 (物理值)
        disc_phys = current_phys[:, self.disc_indices_t] # (B, N_Disc)
        
        disc_enc_list = []
        # 对每一列离散特征进行查表编码
        for i, global_idx in enumerate(self.disc_indices):
            col_vals = disc_phys[:, i].cpu().numpy() # 查表操作在 CPU 执行
            mapping = self.discrete_maps[global_idx]
            
            # 向量化查表
            # 若遇到未知值，默认映射为 0 (通常为 class 0)
            encoded_col = [mapping.get(int(v), 0) for v in col_vals]
            disc_enc_list.append(encoded_col)
            
        # 堆叠为 (B, N_Disc) 张量
        if disc_enc_list:
            disc_enc = torch.tensor(disc_enc_list, device=self.device, dtype=torch.long).t()
        else:
            # 处理无离散特征的边缘情况
            disc_enc = torch.zeros((len(base_params), 0), device=self.device, dtype=torch.long)
        
        # 4. 波形归一化
        acc_norm = waveform_tensor / self.waveform_factor
        
        return acc_norm, cont_norm, disc_enc

    def normalize_action(self, action_phys: torch.Tensor) -> torch.Tensor:
        """
        将物理空间的控制参数映射到优化空间 [-1, 1]。
        公式: y = 2 * (x - min) / (max - min) - 1
        """
        return 2.0 * (action_phys - self.ctrl_min_t) / (self.ctrl_max_t - self.ctrl_min_t) - 1.0

    def denormalize_action(self, action_opt: torch.Tensor) -> torch.Tensor:
        """
        将优化空间的控制参数 [-1, 1] 映射回物理空间。
        公式: x = 0.5 * (y + 1) * (max - min) + min
        同时执行数值截断，确保物理值不越界。
        """
        action_opt = torch.clamp(action_opt, -1.0, 1.0)
        return 0.5 * (action_opt + 1.0) * (self.ctrl_max_t - self.ctrl_min_t) + self.ctrl_min_t