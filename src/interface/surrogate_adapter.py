# -*- coding: utf-8 -*-
import sys
import os
import json
import torch
import torch.nn as nn
import logging
from typing import Dict, Any
import importlib.util

# 配置日志
logger = logging.getLogger(__name__)

class SurrogateModelAdapter(nn.Module):
    """
    代理模型适配器 (Adapter Pattern)
    功能：
    1. 封装原有的 InjuryPredictModel。
    2. 自动从权重目录下的 JSON 文件读取模型结构参数 (解耦 config.py)。
    3. 提供标准的 forward 接口，支持梯度回传到输入。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 本项目的全局配置字典 (对应 default_config.yaml)
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        
        # 1. 路径设置与模块导入 (只导入类定义，不导入配置)
        self._setup_imports()
        
        # 2. 从权重目录加载模型结构配置
        self.model_structure_params = self._load_model_config()
        
        # 3. 初始化模型架构
        self.model = self._build_model()
        
        # 4. 加载预训练权重
        self._load_checkpoint()
        
        # 5. 冻结模型参数 (只优化输入, 不更新模型权重)
        self._freeze_model()
        
        logger.info("Surrogate Model initialized and frozen successfully (Config loaded from JSON).")

    def _setup_imports(self):
        """动态添加原项目路径到 sys.path，以便加载 utils.models"""
        surrogate_dir = self.config['paths']['surrogate_project_dir']
        abs_path = os.path.abspath(surrogate_dir)
        
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Surrogate project directory not found: {abs_path}")
            
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)
            logger.debug(f"Added {abs_path} to sys.path")

        try:
            # 仅导入模型定义模块，不再导入 config.py
            from utils import models
            self.origin_models_module = models
        except ImportError as e:
            raise ImportError(f"Failed to import 'utils.models' from surrogate project: {e}")

    def _load_model_config(self) -> Dict[str, Any]:
        """
        寻找并加载权重文件同目录下的 TrainingRecord.json
        """
        ckpt_path = self.config['paths']['surrogate_checkpoint']
        ckpt_dir = os.path.dirname(ckpt_path)
        
        # 假设配置文件名为 TrainingRecord.json (根据 train.py 的逻辑)
        json_path = os.path.join(ckpt_dir, "TrainingRecord.json")
        
        # 如果找不到，尝试搜索目录下的任意 json 文件
        if not os.path.exists(json_path):
            logger.warning(f"TrainingRecord.json not found at {json_path}. Searching for other .json files...")
            json_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.json')]
            if len(json_files) == 1:
                json_path = os.path.join(ckpt_dir, json_files[0])
                logger.info(f"Found alternative config file: {json_files[0]}")
            elif len(json_files) > 1:
                raise FileNotFoundError(f"Multiple JSON files found in {ckpt_dir}. Please ensure 'TrainingRecord.json' exists.")
            else:
                raise FileNotFoundError(f"No JSON config file found in {ckpt_dir}. Cannot reconstruct model architecture.")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                record = json.load(f)
            
            # 提取模型参数部分
            # 根据 train.py 结构: record["hyperparameters"]["model"]
            if "hyperparameters" in record and "model" in record["hyperparameters"]:
                return record["hyperparameters"]["model"]
            else:
                # 兼容可能的旧格式或直接存储参数的格式
                return record
                
        except Exception as e:
            raise ValueError(f"Failed to parse model config from {json_path}: {e}")

    def _build_model(self) -> nn.Module:
        """根据 JSON 配置重建模型结构"""
        mp = self.model_structure_params
        
        # 获取离散特征类别数 (从 JSON 读取)
        num_classes_list = mp.get('num_classes_discrete') or mp.get('num_classes_of_discrete')
        
        if num_classes_list is None:
            # 兜底策略：如果 JSON 里实在没有，回退到默认值并警告
            logger.warning("'num_classes_discrete' not found in JSON. Using default [2, 3].")
            num_classes_list = [2, 3] 

        # 实例化模型
        try:
            model = self.origin_models_module.InjuryPredictModel(
                num_classes_of_discrete=num_classes_list,
                Ksize_init=mp['Ksize_init'],
                Ksize_mid=mp['Ksize_mid'],
                num_blocks_of_tcn=mp['num_blocks_of_tcn'],
                tcn_channels_list=mp['tcn_channels_list'],
                tcn_output_dim=mp['tcn_output_dim'],
                mlp_encoder_output_dim=mp['mlp_encoder_output_dim'],
                mlp_decoder_output_dim=mp['mlp_decoder_output_dim'],
                mlpE_hidden=mp['mlpE_hidden'],
                mlpD_hidden=mp['mlpD_hidden'],
                num_layers_of_mlpE=mp['num_layers_of_mlpE'],
                num_layers_of_mlpD=mp['num_layers_of_mlpD'],
                dropout_MLP=mp['dropout_MLP'], 
                dropout_TCN=mp['dropout_TCN'],
                use_channel_attention=mp.get('use_channel_attention', True),
                fixed_channel_weight=mp.get('fixed_channel_weight', None)
            )
            return model.to(self.device)
        except KeyError as e:
            raise KeyError(f"JSON config is missing required model parameter: {e}")

    def _load_checkpoint(self):
        """加载权重文件"""
        ckpt_path = self.config['paths']['surrogate_checkpoint']
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
            
        logger.info(f"Loading surrogate weights from {ckpt_path}...")
        
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        # 兼容性处理：如果保存的是整个 state_dict，直接加载
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
             self.model.load_state_dict(checkpoint['state_dict'])
        else:
             self.model.load_state_dict(checkpoint)

    def _freeze_model(self):
        """冻结模型所有参数，设置为 eval 模式"""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, 
                acc_waveforms: torch.Tensor, 
                continuous_feats: torch.Tensor, 
                discrete_feats: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        # 确保输入在正确设备上
        acc_waveforms = acc_waveforms.to(self.device)
        continuous_feats = continuous_feats.to(self.device)
        discrete_feats = discrete_feats.to(self.device)

        preds, _, _ = self.model(acc_waveforms, continuous_feats, discrete_feats)
        return preds

    def get_input_gradients(self, 
                          acc_waveforms: torch.Tensor, 
                          continuous_feats: torch.Tensor, 
                          discrete_feats: torch.Tensor) -> torch.Tensor:
        """
        辅助函数：计算 Loss 关于 continuous_feats 的梯度
        """
        if not continuous_feats.requires_grad:
            continuous_feats.requires_grad_(True)
            
        preds = self.forward(acc_waveforms, continuous_feats, discrete_feats)
        
        # 简单求和作为伪 Loss，触发自动微分引擎（Autograd），验证并计算输入张量（Input Tensor）的梯度
        pseudo_loss = preds.sum()
        pseudo_loss.backward()
        
        return continuous_feats.grad