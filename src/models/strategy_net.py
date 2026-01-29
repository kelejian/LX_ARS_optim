# src/models/strategy_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class StrategyNet(nn.Module):
    """
    策略网络 A_theta(s) -> a
    
    功能：
    接收工况状态 (State)，直接输出接近全局最优的控制参数 (Action)。
    该网络作为“摊销优化”(Amortized Optimization) 的载体。
    
    输入:
        x_continuous: (B, N_cont) 归一化的连续特征
        x_discrete: (B, N_disc) 离散特征索引
    
    输出:
        action: (B, N_ctrl) 归一化到 [-1, 1] 的控制参数
    """
    
    def __init__(self, 
                 continuous_dim: int, 
                 discrete_dims: List[int],
                 action_dim: int, 
                 hidden_dims: List[int] = [128, 128],
                 dropout: float = 0.0):
        """
        Args:
            continuous_dim: 连续状态特征数量 (e.g. 11)
            discrete_dims: 离散特征的类别数列表 (e.g. [2, 4] for is_driver, OT)
            action_dim: 输出动作维度 (e.g. 5)
            hidden_dims: 隐藏层维度列表
            dropout: Dropout概率
        """
        super(StrategyNet, self).__init__()
        
        # 1. 离散特征嵌入 (Embeddings)
        # 为每个离散特征建立 Embedding，维度设为 min(num_classes, 8)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, min(num_classes, 8)) 
            for num_classes in discrete_dims
        ])
        
        # 计算嵌入后的总维度
        # embedding_dim_sum = sum(min(c, 8))
        self.emb_out_dim = sum([e.embedding_dim for e in self.embeddings])
        
        # 2. MLP 主干
        input_dim = continuous_dim + self.emb_out_dim
        layers = []
        curr_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_dim = hidden_dim
            
        self.backbone = nn.Sequential(*layers)
        
        # 3. 输出头 (Action Head)
        # 输出范围 [-1, 1]，对应 ParamManager 的 Optimization Space
        self.action_head = nn.Sequential(
            nn.Linear(curr_dim, action_dim),
            nn.Tanh() 
        )
        
        # 4. 初始化权重
        self._init_weights()
        
        logger.info(f"StrategyNet initialized. "
                    f"Input: [Cont={continuous_dim}, Disc={discrete_dims} -> Emb={self.emb_out_dim}], "
                    f"Output: {action_dim}, Hidden: {hidden_dims}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_continuous: torch.Tensor, x_discrete: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_continuous: (B, 11)
            x_discrete: (B, 2) Int64 Tensor
        Returns:
            action: (B, 5) Range [-1, 1]
        """
        # 1. 处理离散特征
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            # x_discrete[:, i] shape: (B,)
            emb = emb_layer(x_discrete[:, i]) # (B, emb_dim)
            emb_list.append(emb)
        
        x_emb = torch.cat(emb_list, dim=1) # (B, total_emb_dim)
        
        # 2. 拼接输入
        x = torch.cat([x_continuous, x_emb], dim=1)
        
        # 3. MLP
        feat = self.backbone(x)
        
        # 4. 输出
        action = self.action_head(feat)
        
        return action