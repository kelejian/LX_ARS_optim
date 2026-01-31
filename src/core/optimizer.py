# src/core/optimizer.py

import os
import torch
import torch.optim as optim
import time
import numpy as np
from typing import Dict, Tuple, Any, Optional

from src.utils.logger import setup_logger
from src.core.param_manager import ParamManager
from src.interface.surrogate_adapter import SurrogateModelAdapter
from src.models.strategy_net import StrategyNet

logger = setup_logger(__name__)

class ARS_Optimizer:
    """
    自适应约束系统优化器 (ARS Optimizer)
    
    核心流程 (Pipeline):
    1. [State Encoding]: 将工况参数和波形转换为模型可读的归一化张量。
    2. [Global Proposal]: 使用策略网络 (StrategyNet) 生成初始参数猜测 a_0。
    3. [Local Refinement]: 基于代理模型 (Surrogate) 的梯度信息，对 a 进行精调。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 全局配置字典
        """
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        
        # --- 1. 初始化核心组件 ---
        logger.info("Initializing ARS Optimizer components...")
        
        # (1) 参数管理器
        self.param_manager = ParamManager(
            param_space_path=os.path.join("configs", "param_space.yaml"),
            preprocessor_path=os.path.join(config['paths']['preprocessors_path']),
            surrogate_project_dir=config['paths']['surrogate_project_dir'],
            device=self.device.type
        )
        
        # (2) 代理模型适配器
        self.surrogate = SurrogateModelAdapter(config)
        
        # (3) 策略网络 (Strategy Net)
        self._init_strategy_net()
        
        # --- 2. 加载优化超参 ---
        self.opt_config = config['optimization']
        self.obj_weights = self.opt_config['objectives']
        
        logger.info("ARS Optimizer initialized successfully.")

    def _init_strategy_net(self):
        """初始化并加载策略网络"""
        net_cfg = self.config['strategy_net']
        
        # 根据 ParamManager 获取维度信息
        cont_dim = len(self.param_manager.cont_scale) 
        disc_dims = [len(m) for m in self.param_manager.discrete_maps.values()]
        action_dim = len(self.param_manager.control_indices)
        
        self.strategy_net = StrategyNet(
            continuous_dim=cont_dim,
            discrete_dims=disc_dims,
            action_dim=action_dim,
            hidden_dims=net_cfg['hidden_dims'],
            dropout=net_cfg['dropout']
        ).to(self.device)
        
        # 尝试加载预训练权重
        weight_path = self.config['paths']['strategy_model_save_path']
        if os.path.exists(weight_path):
            try:
                state_dict = torch.load(weight_path, map_location=self.device)
                self.strategy_net.load_state_dict(state_dict)
                logger.info(f"Loaded StrategyNet weights from {weight_path}")
                self.strategy_net.eval()
            except Exception as e:
                logger.warning(f"Failed to load StrategyNet weights: {e}. Using random init.")
        else:
            logger.warning(f"No StrategyNet weights found at {weight_path}. Using random init.")

    # =========================================================================
    #  AIS3+ 损伤风险曲线计算 (可微实现)
    #  Sources: NHTSA Injury Risk Curves (Eppinger et al., 1999; Mertz et al., 2016)
    # =========================================================================

    def _prob_head_ais3(self, hic15: torch.Tensor) -> torch.Tensor:
        """
        Head Injury Risk (AIS 3+)
        Curve: P = Phi( (ln(HIC15) - 7.45231) / 0.73998 )
        """
        # 避免 log(0) 或负数
        hic_safe = torch.clamp(hic15, min=1.0)
        z = (torch.log(hic_safe) - 7.45231) / 0.73998
        # 使用 standard normal cdf
        return 0.5 * (1 + torch.erf(z / 1.41421356))

    def _prob_chest_ais3(self, dmax: torch.Tensor, ot_tensor: torch.Tensor) -> torch.Tensor:
        """
        Chest Injury Risk (AIS 3+)
        Curve: P = 1 / (1 + exp(10.5456 - 1.568 * (Dmax * Scale)^0.4612))
        
        Args:
            ot_tensor: OT (1=5th Female, 2=50th Male, 3=95th Male)
        """
        dmax_safe = torch.clamp(dmax, min=0.0)
        
        # 确定缩放因子 (Scaling Factor)
        # OT=1 (5th): 221/182.9 ≈ 1.208
        # OT=2 (50th): 1.0
        # OT=3 (95th): 221/246.38 ≈ 0.897
        # 注意：这里使用 mask 操作保持梯度流
        scale = torch.ones_like(dmax_safe)
        scale = torch.where(ot_tensor == 1, torch.tensor(1.2083, device=self.device), scale)
        scale = torch.where(ot_tensor == 3, torch.tensor(0.8970, device=self.device), scale)
        
        dmax_eq = dmax_safe * scale
        exponent = 10.5456 - 1.568 * torch.pow(dmax_eq + 1e-6, 0.4612)
        return torch.sigmoid(-exponent) # sigmoid(x) = 1/(1+exp(-x))

    def _prob_neck_ais3(self, nij: torch.Tensor) -> torch.Tensor:
        """
        Neck Injury Risk (AIS 3+)
        Curve: P = 1 / (1 + exp(3.2269 - 1.9688 * Nij))
        """
        nij_safe = torch.clamp(nij, min=0.0)
        exponent = 3.2269 - 1.9688 * nij_safe
        return torch.sigmoid(-exponent)

    def _compute_objective(self, predictions: torch.Tensor, state_dict: Dict[str, Any]) -> torch.Tensor:
        """
        计算基于 AIS3+ 联合概率的加权目标函数
        
        Loss = 1 - Product( (1 - P_k^w_k) )
        其中 w_k 为指数加权系数，越接近 0 表示该项损伤越重要（惩罚越重）。
        
        Args:
            predictions: (B, 3) -> [HIC, Dmax, Nij]
            state_dict: 包含 'OT' 等物理参数
        """
        # 1. 提取物理预测值
        hic = predictions[:, 0]
        dmax = predictions[:, 1]
        nij = predictions[:, 2]
        
        # 2. 获取乘员类型 (用于胸部缩放)
        # 假设 batch size 为 1 或所有样本 state 相同 (简化处理，若 batch 异构需从 tensor 获取)
        ot_val = state_dict.get('OT', 2) # 默认为 50th Male
        ot_tensor = torch.full_like(dmax, ot_val, device=self.device)

        # 3. 计算各部位 AIS3+ 概率 [0, 1]
        p_head = self._prob_head_ais3(hic)
        p_chest = self._prob_chest_ais3(dmax, ot_tensor)
        p_neck = self._prob_neck_ais3(nij)
        
        # 4. 应用指数加权 (Exponent Weighting)
        # 逻辑：P_weighted = P ^ weight
        # 若 weight -> 0 (e.g. 0.01), P^0.01 -> 1.0 (Risk Max), 从而使得 Survival (1-P) -> 0
        w_h = self.obj_weights.get('weight_hic', 1.0)
        w_c = self.obj_weights.get('weight_dmax', 1.0)
        w_n = self.obj_weights.get('weight_nij', 1.0)
        
        p_head_w = torch.pow(p_head + 1e-8, w_h)
        p_chest_w = torch.pow(p_chest + 1e-8, w_c)
        p_neck_w = torch.pow(p_neck + 1e-8, w_n)
        
        # 5. 计算联合损伤风险 (Joint Injury Risk)
        # Risk = 1 - P(Survival_Head) * P(Survival_Chest) * P(Survival_Neck)
        joint_survival = (1.0 - p_head_w) * (1.0 - p_chest_w) * (1.0 - p_neck_w)
        loss_risk = 1.0 - joint_survival
        
        # TODO: 这里可以根据专利加入 L_constraint (如 ReLU 边界惩罚)，目前仅返回风险项
        return loss_risk.mean()

    def optimize(self, 
                 state_dict: Dict[str, float], 
                 crash_waveform: torch.Tensor) -> Dict[str, Any]:
        """
        执行完整的 ARS 寻优流程
        """
        start_time = time.time()
        crash_waveform = crash_waveform.to(self.device)
        
        # Phase 1: 准备输入
        dummy_action = torch.zeros((1, len(self.param_manager.control_indices)), device=self.device)
        _, cont_norm_state, disc_enc_state = self.param_manager.get_model_input(
            state_dict, dummy_action, crash_waveform
        )
        
        # Phase 2: 摊销推理 (Step 3)
        with torch.no_grad():
            action_opt_init = self.strategy_net(cont_norm_state, disc_enc_state)
        
        action_phys_init = self.param_manager.denormalize_action(action_opt_init)
        
        # 计算初始 Loss (需传入 state_dict 以获取 OT)
        with torch.no_grad():
            preds_init = self.surrogate(crash_waveform, *self._split_model_input(state_dict, action_phys_init, crash_waveform)[1:])
            loss_init = self._compute_objective(preds_init, state_dict)
        
        # Phase 3: 局部精调 (Step 4)
        action_opt = action_opt_init.clone().detach().requires_grad_(True)
        refine_steps = self.opt_config['refine_steps']
        lr = self.opt_config['lr']
        optimizer = optim.Adam([action_opt], lr=lr)
        
        best_loss = loss_init.item()
        best_action_opt = action_opt_init.clone().detach()
        best_preds = preds_init.clone().detach()
        trajectory = []
        
        if refine_steps > 0:
            for step in range(refine_steps):
                optimizer.zero_grad()
                
                action_phys = self.param_manager.denormalize_action(action_opt)
                acc_in, cont_in, disc_in = self.param_manager.get_model_input(
                    state_dict, action_phys, crash_waveform
                )
                preds = self.surrogate(acc_in, cont_in, disc_in)
                
                # 更新：传入 state_dict
                loss = self._compute_objective(preds, state_dict)
                
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    action_opt.data.clamp_(-1.0, 1.0)
                
                curr_loss = loss.item()
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    best_action_opt = action_opt.clone().detach()
                    best_preds = preds.clone().detach()
                
                trajectory.append({
                    "step": step, 
                    "loss": curr_loss,
                    "preds": preds.detach().cpu().numpy().tolist()
                })
        
        # Phase 4: 结果封装
        best_action_phys = self.param_manager.denormalize_action(best_action_opt)
        
        result = {
            "initial": {
                "action_phys": action_phys_init.detach().cpu().numpy().flatten().tolist(),
                "preds": preds_init.detach().cpu().numpy().flatten().tolist(),
                "loss": loss_init.item()
            },
            "optimized": {
                "action_phys": best_action_phys.detach().cpu().numpy().flatten().tolist(),
                "preds": best_preds.detach().cpu().numpy().flatten().tolist(),
                "loss": best_loss
            },
            "trajectory": trajectory,
            "time_cost": time.time() - start_time
        }
        
        return result

    def _split_model_input(self, state, action, wave):
        acc, cont, disc = self.param_manager.get_model_input(state, action, wave)
        return acc, cont, disc