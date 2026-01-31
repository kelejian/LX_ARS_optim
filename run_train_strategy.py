# run_train_strategy.py

import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 导入项目组件
from src.utils.logger import setup_logger
from src.interface.data_loader import ARSDataLoader
from src.core.param_manager import ParamManager
from src.interface.surrogate_adapter import SurrogateModelAdapter
from src.models.strategy_net import StrategyNet

# 配置日志
logger = setup_logger("TrainStrategy")

def parse_args():
    parser = argparse.ArgumentParser(description="离线训练策略网络 (Step 3 Model)")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="配置文件路径")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的 epoch 数")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖配置中的 batch_size")
    parser.add_argument("--lr", type=float, default=None, help="覆盖配置中的学习率")
    return parser.parse_args()

def batch_process_inputs(pm: ParamManager, 
                         state_dict_batch: dict, 
                         waveform_batch: torch.Tensor, 
                         device: torch.device):
    """
    [高效辅助函数] 批量处理工况状态，准备 StrategyNet 的输入
    替代 ParamManager.get_model_input 的单样本逻辑，支持 Batch State。
    
    Args:
        pm: ParamManager 实例
        state_dict_batch: DataLoader collate 后的字典 {key: Tensor(B,)}
        waveform_batch: Tensor(B, 2, 150)
    
    Returns:
        cont_norm: (B, 11) 归一化的连续特征（其中 Action 部分填 0）
        disc_enc: (B, 2) 编码后的离散特征
        acc_norm: (B, 2, 150) 归一化的波形
    """
    B = waveform_batch.shape[0]
    
    # 1. 构建连续特征矩阵 (B, 11)
    continuous_phys = torch.zeros((B, 11), device=device, dtype=torch.float32)
    
    for p in pm.param_definitions:
        idx = p['index']
        # 仅填充 State 参数 (Index < 11 的连续参数)
        if idx < 11 and idx in pm.state_indices:
            name = p['name']
            if name not in state_dict_batch:
                raise ValueError(f"Missing batch data for state: {name}")
            val = state_dict_batch[name].to(device, dtype=torch.float32)
            continuous_phys[:, idx] = val
            
    # 2. 连续特征归一化: (x - offset) * scale
    # Action 列此时为 0，归一化后会有数值，但 StrategyNet 会忽略或学习适应它
    # 重要的是 State 列是正确的归一化值
    cont_norm = (continuous_phys - pm.cont_offset) * pm.cont_scale
    
    # 3. 离散特征编码 (B, 2)
    disc_list = []
    # Hardcoded global indices for discrete features [11, 12]
    # 必须严格对应 ParamManager.discrete_maps 的顺序
    target_global_indices = [11, 12] 
    
    for global_idx in target_global_indices:
        # 找到对应的参数名
        p_name = next(p['name'] for p in pm.param_definitions if p['index'] == global_idx)
        raw_vals = state_dict_batch[p_name].to(device) # Tensor (B,)
        
        # 向量化查表
        # 创建一个 lookup tensor (假设类别 id 连续且从 0 开始，或者用 CPU map)
        # 鉴于 map 是非连续的 (e.g. {1:0, 2:1, 5:2}), 向量化比较麻烦。
        # 考虑到 Batch Size 不大 (e.g. 256)，在 CPU 做 map 也是可以接受的。
        
        mapping = pm.discrete_maps[global_idx]
        raw_vals_np = raw_vals.cpu().numpy()
        enc_vals_np = np.array([mapping.get(float(v), 0) for v in raw_vals_np], dtype=np.int64)
        
        disc_list.append(torch.from_numpy(enc_vals_np).to(device))
        
    disc_enc = torch.stack(disc_list, dim=1) # (B, 2)
    
    # 4. 波形归一化
    acc_norm = waveform_batch.to(device) / pm.waveform_factor
    
    return cont_norm, disc_enc, acc_norm

def train(args):
    # 1. 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 覆盖超参
    train_cfg = config['strategy_net']['train']
    if args.epochs: train_cfg['epochs'] = args.epochs
    if args.batch_size: train_cfg['batch_size'] = args.batch_size
    if args.lr: train_cfg['lr'] = args.lr
    
    device = torch.device(config['device'])
    logger.info(f"Training StrategyNet on {device}. Epochs: {train_cfg['epochs']}, BS: {train_cfg['batch_size']}")

    # 2. 初始化组件
    # (A) ParamManager
    pm = ParamManager(
        param_space_path="configs/param_space.yaml",
        preprocessor_path=os.path.join(config['paths']['preprocessors_path']),
        surrogate_project_dir=config['paths']['surrogate_project_dir'],
        device=device.type
    )
    
    # (B) Surrogate Model (Teacher)
    surrogate = SurrogateModelAdapter(config)
    surrogate.to(device)
    # 确保 Surrogate 不更新
    for param in surrogate.parameters():
        param.requires_grad = False
        
    # (C) StrategyNet (Student)
    # 自动获取维度
    cont_dim = 11
    disc_dims = [len(m) for m in pm.discrete_maps.values()]
    action_dim = len(pm.control_indices)
    
    net = StrategyNet(
        continuous_dim=cont_dim,
        discrete_dims=disc_dims,
        action_dim=action_dim,
        hidden_dims=config['strategy_net']['hidden_dims'],
        dropout=config['strategy_net']['dropout']
    ).to(device)
    net.train()

    # 3. 数据加载
    # 使用 'train' 划分的数据进行训练
    data_loader = ARSDataLoader(config, split='train')
    # 使用 PyTorch DataLoader 进行 batching
    # default_collate 会自动将 list of dicts 转为 dict of tensors
    train_loader = DataLoader(
        data_loader, 
        batch_size=train_cfg['batch_size'], 
        shuffle=True, 
        num_workers=0,
        drop_last=True
    )

    # 4. 优化器
    optimizer = optim.AdamW(
        net.parameters(), 
        lr=float(train_cfg['lr']), 
        weight_decay=float(train_cfg['weight_decay'])
    )

    # 5. 训练循环
    obj_weights = config['optimization']['objectives']
    best_loss = float('inf')
    save_path = config['paths']['strategy_model_save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    logger.info("Start Training Loop...")
    
    for epoch in range(train_cfg['epochs']):
        epoch_loss = 0.0
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}", unit="batch")
        
        for batch_data in pbar:
            # batch_data 结构: {'case_id': Tensor, 'state_dict': Dict[str, Tensor], 'waveform': Tensor...}
            state_dict_batch = batch_data['state_dict']
            waveform_batch = batch_data['waveform'] # (B, 1, 2, 150) from ARSDataLoader unsqueeze
            
            # ARSDataLoader 的 get_data_by_index 里做了 unsqueeze(0), 
            # 所以 batch 后的维度可能是 (B, 1, 2, 150)。需要 squeeze 掉 dim 1
            if waveform_batch.dim() == 4:
                waveform_batch = waveform_batch.squeeze(1)
            
            # --- A. 准备 StrategyNet 输入 ---
            cont_norm_in, disc_enc_in, acc_norm = batch_process_inputs(pm, state_dict_batch, waveform_batch, device)
            
            # --- B. 策略网络预测 (Forward) ---
            optimizer.zero_grad()
            
            # Action Opt Space: [-1, 1]
            action_opt = net(cont_norm_in, disc_enc_in)
            
            # --- C. 映射到 Surrogate 输入空间 ---
            # 1. Opt Space -> Phys Space
            action_phys = pm.denormalize_action(action_opt)
            
            # 2. 拼装完整的 Continuous Phys 矩阵 (State + Action)
            # 这里的 cont_phys_full 必须是完全正确的物理值
            cont_phys_full = torch.zeros_like(cont_norm_in) # shape (B, 11)
            
            # 填回 State (反归一化 Cont Norm 并不靠谱，直接用原始 State Phys 构建更准)
            # 因为 cont_norm_in 里的 Action 列是错误的 (0)
            for p in pm.param_definitions:
                idx = p['index']
                if idx < 11:
                    if idx in pm.state_indices:
                        cont_phys_full[:, idx] = state_dict_batch[p['name']].to(device)
                    elif idx in pm.control_indices:
                        # 找到这是第几个 control
                        ctrl_idx = pm.control_indices.index(idx)
                        cont_phys_full[:, idx] = action_phys[:, ctrl_idx]
            
            # 3. Phys Space -> Model Norm Space (Surrogate Input)
            # 这一步是关键：必须使用 ParamManager 的归一化参数，且保留梯度图
            cont_norm_surrogate = (cont_phys_full - pm.cont_offset) * pm.cont_scale
            
            # --- D. 代理模型计算 Loss ---
            # Surrogate 即使是 frozen 的，只要 input (action_phys derived) 有 grad，
            # 梯度就能回传给 StrategyNet
            preds = surrogate(acc_norm, cont_norm_surrogate, disc_enc_in) # (B, 3)
            
            # --- E. 计算加权损失 ---
            hic = preds[:, 0]
            dmax = preds[:, 1]
            nij = preds[:, 2]
            
            loss = (obj_weights['weight_hic'] * hic +
                    obj_weights['weight_dmax'] * dmax +
                    obj_weights['weight_nij'] * nij).mean()
            
            # --- F. 反向传播 ---
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / steps if steps > 0 else 0
        logger.info(f"Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(net.state_dict(), save_path)
            logger.info(f"New best model saved to {save_path}")

    logger.info("Training Complete.")

if __name__ == "__main__":
    args = parse_args()
    train(args)