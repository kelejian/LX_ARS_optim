# src/interface/data_loader.py

import sys
import os
import torch
import numpy as np
import yaml
from typing import Dict, Any, Tuple, Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ARSDataLoader:
    """
    ARS 数据加载器 (Data Loader)
    
    职责：
    1. 加载原始的测试/验证数据集 (基于InjuryPredict项目的 CrashDataset)。
    2. 提取"物理空间"的原始波形和工况参数。
    3. 将数据打包成 Optimizer 需要的格式 (state_dict, waveform)。
    """

    def __init__(self, config: Dict[str, Any], split: str = 'val'):
        """
        Args:
            config: 全局配置字典
            split: 数据集划分 ('val', 'test', 'train' or 'all')
                   注意：'all' 会直接从 .npz 加载所有数据，忽略划分。
        """
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        self.split = split
        
        # 1. 路径准备
        self.surrogate_dir = config['paths']['surrogate_project_dir']
        self._setup_imports()
        
        # 2. 加载参数定义 (用于解析 raw 数组到 dict)
        self.param_def = self._load_param_definitions()
        
        # 3. 加载数据集
        self.dataset = self._load_dataset()
        self.indices = self._get_split_indices()
        
        logger.info(f"ARSDataLoader initialized. Mode: {split}, Samples: {len(self.indices)}")

    def _setup_imports(self):
        """挂载兄弟项目路径以导入 CrashDataset"""
        abs_path = os.path.abspath(self.surrogate_dir)
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)
        
        try:
            from utils.dataset_prepare import CrashDataset
            self.dataset_cls = CrashDataset
        except ImportError as e:
            raise ImportError(f"Failed to import CrashDataset from {abs_path}: {e}")

    def _load_param_definitions(self) -> list:
        """加载 param_space.yaml 以获取参数名称和索引"""
        path = os.path.join("configs", "param_space.yaml")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Param space config not found at {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return cfg['parameters'] # 类型: list of dict

    def _load_dataset(self):
        """实例化 CrashDataset"""
        # 构造 npz 的绝对路径
        data_input_path = os.path.abspath(self.config['paths']['data_input'])
        data_labels_path = os.path.abspath(self.config['paths']['data_labels'])
        
        logger.info(f"Loading raw data from {data_input_path}...")
        
        # 实例化InjuryPredict项目的 Dataset 类; 它会自动读取 npz 并存储在 self.x_att_raw, self.x_acc_raw 等属性中
        ds = self.dataset_cls(input_file=data_input_path, label_file=data_labels_path)
        return ds

    def _get_split_indices(self) -> np.ndarray:
        """获取指定划分的索引列表"""
        if self.split == 'all':
            return np.arange(len(self.dataset))
            
        # 尝试加载划分文件 (train_dataset.pt 等)
        # 注意：这些 .pt 文件是 torch.utils.data.Subset 对象，包含 indices
        filename = f"{self.split}_dataset.pt"
        # 设 .pt 文件在 surrogate_project/data/ 目录下
        pt_path = os.path.join(self.surrogate_dir, "data", filename)
        
        if not os.path.exists(pt_path):
            logger.warning(f"Split file {pt_path} not found. Fallback to loading ALL data.")
            return np.arange(len(self.dataset))
            
        try:
            subset = torch.load(pt_path, weights_only=False) # weights_only=False 因包含自定义类
            logger.info(f"Loaded split indices from {filename}")
            return np.array(subset.indices)
        except Exception as e:
            logger.error(f"Failed to load {pt_path}: {e}")
            raise e

    def __len__(self):
        return len(self.indices)

    def get_data_by_index(self, idx: int) -> Dict[str, Any]:
        """
        根据数据集的全局索引获取数据 (Raw Physical Values)
        
        Args:
            idx: 在 dataset 中的全局索引 (注意区分 Subset 的相对索引)
        
        Returns:
            Dict: {
                "case_id": int,
                "state_dict": Dict[str, float],      # 包含所有13个参数的物理值
                "waveform": Tensor (1, 2, 150),      # 原始物理尺度碰撞波形(x,y方向加速度)
                "ground_truth": Dict[str, float]     # 真实损伤值 (HIC, Dmax, Nij, AIS_head, AIS_chest, AIS_neck, MAIS)
            }
        """
        # 1. 获取 Case ID
        case_id = int(self.dataset.case_ids[idx])
        
        # 2. 获取原始波形 (2, 150) -> (1, 2, 150)
        # 注意：CrashDataset 存储的是 numpy float
        wave_np = self.dataset.x_acc_raw[idx]
        waveform = torch.tensor(wave_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 3. 获取原始参数并解析为 state_dict
        # x_att_raw shape: (13,)
        params_np = self.dataset.x_att_raw[idx]
        state_dict = {}
        
        for p in self.param_def:
            p_idx = p['index']
            p_name = p['name']
            val = float(params_np[p_idx])
            
            # 如果是离散参数，确保转换为 int
            if p['type'] == 'discrete':
                state_dict[p_name] = int(val)
            else:
                state_dict[p_name] = val
                
        # 4. 获取 Ground Truth (用于评估对比)
        gt = {
            "HIC": float(self.dataset.y_HIC[idx]),
            "Dmax": float(self.dataset.y_Dmax[idx]),
            "Nij": float(self.dataset.y_Nij[idx]),
            "AIS_head": int(self.dataset.ais_head[idx]),
            "AIS_chest": int(self.dataset.ais_chest[idx]),
            "AIS_neck": int(self.dataset.ais_neck[idx]),
            "MAIS": int(self.dataset.mais[idx])
        }
        
        return {
            "case_id": case_id,
            "state_dict": state_dict,
            "waveform": waveform,
            "ground_truth": gt
        }

    def __getitem__(self, item):
        """支持 loader[i] 访问，自动映射 Subset 索引到 Global 索引"""
        global_idx = self.indices[item]
        return self.get_data_by_index(global_idx)