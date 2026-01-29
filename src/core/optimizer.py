# src/core/optimizer.py

import os
import torch
import torch.optim as optim
import time
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
    2. [Step 3 - Global Proposal]: 使用策略网络 (StrategyNet) 生成初始参数猜测 a_0。
       - 此时 a_0 在 Optimization Space [-1, 1]。
    3. [Step 4 - Local Refinement]: 基于代理模型 (Surrogate) 的梯度信息，对 a 进行精调。
       - 使用 Projected Gradient Descent (PGD) 确保参数在合法边界内。
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
            preprocessor_path=os.path.join(config['paths']['surrogate_project_dir'], "data", "preprocessors.joblib"),
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
        # 连续特征维度 (11)
        cont_dim = len(self.param_manager.cont_scale) 
        # 离散特征类别数 [2, 4] (hardcoded based on project knowledge, or parsed)
        # 这里为了稳健，我们通过 discrete_maps 的长度获取
        disc_dims = [len(m) for m in self.param_manager.discrete_maps.values()]
        # 动作维度 (5)
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
                self.strategy_net.eval() # 推理模式
            except Exception as e:
                logger.warning(f"Failed to load StrategyNet weights: {e}. Using random init.")
        else:
            logger.warning(f"No StrategyNet weights found at {weight_path}. Using random init (Amortized optimization may fail).")

    def _compute_objective(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        计算加权目标函数 Loss
        Args:
            predictions: (B, 3) -> [HIC, Dmax, Nij]
        Returns:
            loss: scalar tensor
        """
        # predictions 通常是归一化后的值还是原始值？
        # Surrogate Model 输出的是原始物理值 (基于 adapter 实现)。
        # 为了优化稳定，最好将其归一化到同一量级，或者依赖 weights 调节。
        # 这里假设 weights 已经考虑了量纲差异 (e.g. HIC ~100-1000, Dmax ~10-50)
        
        hic = predictions[:, 0]
        dmax = predictions[:, 1]
        nij = predictions[:, 2]
        
        # 简单的线性加权
        loss = (self.obj_weights['weight_hic'] * hic +
                self.obj_weights['weight_dmax'] * dmax +
                self.obj_weights['weight_nij'] * nij)
        
        return loss.mean() # Batch mean

    def optimize(self, 
                 state_dict: Dict[str, float], 
                 crash_waveform: torch.Tensor) -> Dict[str, Any]:
        """
        执行完整的 ARS 寻优流程 (Step 3 + Step 4)
        
        Args:
            state_dict: 单个工况的物理参数 (e.g. {'impact_velocity': 56.0, ...})
            crash_waveform: (1, 2, 150) 物理波形张量
            
        Returns:
            result: 包含优化结果的字典
        """
        start_time = time.time()
        
        # 确保波形在正确设备
        crash_waveform = crash_waveform.to(self.device)
        
        # =====================================================
        # Phase 1: 准备输入 (Prepare Input)
        # =====================================================
        # 我们需要为 StrategyNet 准备归一化的 State 输入
        # 这里使用一个小技巧：传入全零的 Action，利用 ParamManager 获取归一化后的 State
        dummy_action = torch.zeros((1, len(self.param_manager.control_indices)), device=self.device)
        
        # inputs: (acc_norm, cont_norm, disc_enc)
        # 注意：cont_norm 中包含了归一化的 state 和 归一化的 dummy action (0)
        _, cont_norm_state, disc_enc_state = self.param_manager.get_model_input(
            state_dict, dummy_action, crash_waveform
        )
        
        # =====================================================
        # Phase 2: 步骤三 - 摊销推理 (Step 3: Amortized Inference)
        # =====================================================
        # 使用 StrategyNet 直接预测 Action (在 [-1, 1] 空间)
        with torch.no_grad():
            action_opt_init = self.strategy_net(cont_norm_state, disc_enc_state) # (1, N_action)
        
        # 记录初始解
        action_phys_init = self.param_manager.denormalize_action(action_opt_init)
        with torch.no_grad():
            preds_init = self.surrogate(crash_waveform, *self._split_model_input(state_dict, action_phys_init, crash_waveform)[1:])
            loss_init = self._compute_objective(preds_init)
        
        logger.debug(f"Step 3 Init Loss: {loss_init.item():.4f}")

        # =====================================================
        # Phase 3: 步骤四 - 局部精调 (Step 4: Local Refinement)
        # =====================================================
        # 将 Action 设为可优化变量
        action_opt = action_opt_init.clone().detach().requires_grad_(True)
        
        refine_steps = self.opt_config['refine_steps']
        lr = self.opt_config['lr']
        
        # 定义优化器 (Adam 通常比 SGD 收敛更快且稳健)
        optimizer = optim.Adam([action_opt], lr=lr)
        
        # 追踪最佳结果
        best_loss = loss_init.item()
        best_action_opt = action_opt_init.clone().detach()
        best_preds = preds_init.clone().detach()
        
        trajectory = [] # 记录优化轨迹用于分析
        
        if refine_steps > 0:
            for step in range(refine_steps):
                optimizer.zero_grad()
                
                # 1. Opt Space -> Phys Space
                # 注意：这是计算图的一部分，必须保留梯度
                action_phys = self.param_manager.denormalize_action(action_opt)
                
                # 2. 拼装 Surrogate 输入
                # inputs: (acc, cont, disc)
                acc_in, cont_in, disc_in = self.param_manager.get_model_input(
                    state_dict, action_phys, crash_waveform
                )
                
                # 3. Surrogate Forward
                preds = self.surrogate(acc_in, cont_in, disc_in)
                
                # 4. Compute Loss
                loss = self._compute_objective(preds)
                
                # 5. Backward & Update
                loss.backward()
                optimizer.step()
                
                # 6. Projection (Clamp to [-1, 1])
                # 保证参数始终在合法物理边界内
                with torch.no_grad():
                    action_opt.data.clamp_(-1.0, 1.0)
                
                # 7. Track Best
                curr_loss = loss.item()
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    best_action_opt = action_opt.clone().detach()
                    best_preds = preds.clone().detach()
                
                # 记录轨迹
                trajectory.append({
                    "step": step, 
                    "loss": curr_loss,
                    "preds": preds.detach().cpu().numpy().tolist()
                })
        
        # =====================================================
        # Phase 4: 结果封装
        # =====================================================
        best_action_phys = self.param_manager.denormalize_action(best_action_opt)
        
        # 将 Tensor 转为 Python float/list
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
        """辅助函数：调用 param_manager 并解包，用于简化代码"""
        acc, cont, disc = self.param_manager.get_model_input(state, action, wave)
        return acc, cont, disc