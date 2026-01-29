# src/utils/metrics.py

import numpy as np
import torch
from typing import Dict, List, Any, Union
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class MetricsTracker:
    """
    评估指标跟踪器 (Metrics Tracker)
    
    功能：
    1. 记录单次优化的关键指标 (初始损失、最终损失、耗时等)。
    2. 计算批量统计信息 (平均优化率、成功率)。
    3. 提供特定指标的计算函数。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有统计数据"""
        self.history = {
            "case_ids": [],
            "initial_loss": [],
            "final_loss": [],
            "improvement_rate": [], # (Init - Final) / Init
            "time_cost": [],
            "param_shift_l2": [],   # 参数调整的欧氏距离
            "steps_taken": []
        }
    
    def update(self, result: Dict[str, Any], case_id: int):
        """
        更新单次优化结果
        
        Args:
            result: optimizer.optimize() 返回的字典
            case_id: 当前案例 ID
        """
        init_loss = result['initial']['loss']
        final_loss = result['optimized']['loss']
        
        # 计算优化率 (防止除零)
        if abs(init_loss) > 1e-6:
            imp_rate = (init_loss - final_loss) / init_loss
        else:
            imp_rate = 0.0
            
        # 计算参数偏移量 (L2 Distance)
        # 此时 action_phys 已经是 list
        p0 = np.array(result['initial']['action_phys'])
        p_opt = np.array(result['optimized']['action_phys'])
        shift = np.linalg.norm(p_opt - p0)
        
        # 记录数据
        self.history["case_ids"].append(case_id)
        self.history["initial_loss"].append(init_loss)
        self.history["final_loss"].append(final_loss)
        self.history["improvement_rate"].append(imp_rate)
        self.history["time_cost"].append(result['time_cost'])
        self.history["param_shift_l2"].append(shift)
        self.history["steps_taken"].append(len(result['trajectory']))

    def compute_summary(self) -> Dict[str, float]:
        """计算当前所有记录的聚合统计信息"""
        count = len(self.history["case_ids"])
        if count == 0:
            return {}
            
        summary = {
            "total_samples": count,
            "avg_initial_loss": np.mean(self.history["initial_loss"]),
            "avg_final_loss": np.mean(self.history["final_loss"]),
            "avg_improvement_rate": np.mean(self.history["improvement_rate"]),
            "avg_time_ms": np.mean(self.history["time_cost"]) * 1000, # ms
            "avg_param_shift": np.mean(self.history["param_shift_l2"]),
            "avg_steps": np.mean(self.history["steps_taken"]),
            
            # 成功率统计：优化率 > 0 即视为有效优化
            "success_rate": np.mean(np.array(self.history["improvement_rate"]) > 1e-4)
        }
        return summary

    def log_summary(self):
        """打印统计摘要"""
        s = self.compute_summary()
        if not s:
            logger.warning("No metrics to log.")
            return

        logger.info("\n" + "="*40)
        logger.info(" [Evaluation Summary]")
        logger.info("="*40)
        logger.info(f" Total Samples      : {s['total_samples']}")
        logger.info(f" Avg Time Cost      : {s['avg_time_ms']:.2f} ms")
        logger.info(f" Avg Steps          : {s['avg_steps']:.1f}")
        logger.info("-" * 40)
        logger.info(f" Avg Initial Loss   : {s['avg_initial_loss']:.4f}")
        logger.info(f" Avg Final Loss     : {s['avg_final_loss']:.4f}")
        logger.info(f" Avg Improvement    : {s['avg_improvement_rate']*100:.2f}%")
        logger.info(f" Success Rate       : {s['success_rate']*100:.1f}%")
        logger.info(f" Avg Param Shift    : {s['avg_param_shift']:.4f} (L2)")
        logger.info("="*40 + "\n")

    @staticmethod
    def calculate_damage_reduction(initial: float, final: float) -> float:
        """静态辅助函数：计算单项损伤降低率"""
        if initial <= 1e-6: return 0.0
        return (initial - final) / initial