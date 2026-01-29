# run_evaluation.py

import argparse
import os
import yaml
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# 导入项目组件
from src.utils.logger import setup_logger
from src.interface.data_loader import ARSDataLoader
from src.core.optimizer import ARS_Optimizer
from src.utils.metrics import MetricsTracker

logger = setup_logger("RunEval")

def parse_args():
    parser = argparse.ArgumentParser(description="ARS 在线寻优与评估 (Step 3 + Step 4)")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="配置文件路径")
    parser.add_argument("--split", type=str, default="test", choices=['train', 'val', 'test', 'all'], help="评估数据集划分")
    parser.add_argument("--output_dir", type=str, default=None, help="结果输出目录 (默认使用 config 中定义的路径)")
    parser.add_argument("--max_samples", type=int, default=None, help="仅评估前N个样本 (用于快速调试)")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 加载配置
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
        
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. 准备输出环境
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = args.output_dir if args.output_dir else config['paths']['output_dir']
    run_out_dir = os.path.join(base_out_dir, f"eval_{args.split}_{timestamp}")
    
    os.makedirs(run_out_dir, exist_ok=True)
    
    # 备份本次运行配置，保证实验可复现
    with open(os.path.join(run_out_dir, "run_config.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    logger.info(f"Evaluation outputs will be saved to: {run_out_dir}")

    # 3. 初始化核心组件
    logger.info("Initializing system components...")
    
    # (A) 数据加载
    try:
        data_loader = ARSDataLoader(config, split=args.split)
    except Exception as e:
        logger.error(f"Failed to initialize DataLoader: {e}")
        return

    # (B) 优化器 (包含 ParamManager, StrategyNet, SurrogateAdapter)
    try:
        optimizer = ARS_Optimizer(config)
    except Exception as e:
        logger.error(f"Failed to initialize Optimizer: {e}")
        return
    
    # (C) 指标跟踪
    tracker = MetricsTracker()

    # 4. 执行评估循环
    total_samples = len(data_loader)
    if args.max_samples:
        total_samples = min(total_samples, args.max_samples)
        logger.info(f"DEBUG MODE: Limiting evaluation to first {total_samples} samples.")

    logger.info(f"Starting evaluation loop on '{args.split}' set ({total_samples} cases)...")
    
    results_list = []
    
    # TODO: [需确认] 当前为串行单样本评估。如果数据集极大(>10W)，可能需要重构 optimizer.optimize 以支持 Batch 并行寻优。
    # 考虑到 ARS 是针对每个事故工况的个性化寻优，串行模拟更符合实际车载 ECU 的处理逻辑。
    
    for i in tqdm(range(total_samples), desc="Optimizing"):
        # 获取单个工况数据
        try:
            case_data = data_loader[i]
        except Exception as e:
            logger.warning(f"Skipping sample index {i} due to loading error: {e}")
            continue
            
        case_id = case_data['case_id']
        state_dict = case_data['state_dict']
        waveform = case_data['waveform'] # (1, 2, 150)
        ground_truth = case_data['ground_truth']

        try:
            # --- 核心调用：执行寻优 ---
            # result 结构: { 'initial': ..., 'optimized': ..., 'trajectory': ..., 'time_cost': ... }
            opt_result = optimizer.optimize(state_dict, waveform)
            
            # --- 更新统计指标 ---
            tracker.update(opt_result, case_id)
            
            # --- 记录详细报表 ---
            # WARNING: [假设] 假设 optimizer.optimize 返回的 action_phys 列表顺序与 param_manager.control_names 严格一致。
            # 这是基于 ParamManager 内部实现逻辑的推断，若 ParamManager 变更需同步修改此处。
            ctrl_names = optimizer.param_manager.control_names
            opt_actions = opt_result['optimized']['action_phys']
            init_actions = opt_result['initial']['action_phys']
            
            record = {
                "case_id": case_id,
                # 性能指标
                "loss_init": opt_result['initial']['loss'],
                "loss_opt": opt_result['optimized']['loss'],
                "improvement": tracker.history["improvement_rate"][-1],
                "time_ms": opt_result['time_cost'] * 1000,
                "steps": len(opt_result['trajectory']),
                
                # 参考真值 (注意：优化目标是 Surrogate Loss，不是 GT MAIS，此项仅供离线参考)
                "gt_MAIS": ground_truth.get('MAIS', -1),
                
                # 参数详情 (Flattened)
                **{f"opt_{name}": val for name, val in zip(ctrl_names, opt_actions)},
                **{f"init_{name}": val for name, val in zip(ctrl_names, init_actions)}
            }
            results_list.append(record)
            
        except Exception as e:
            # 捕获优化过程中的计算错误 (如梯度爆炸、Surrogate 报错)
            logger.error(f"Optimization failed for Case {case_id}: {e}")
            # TODO: [需确认] 错误处理策略：当前策略为跳过并继续。
            # 是否需要将失败案例 ID 记录到单独的 error_log.txt 中以便后续排查？
            continue

    # 5. 生成最终报告
    if not results_list:
        logger.warning("No results generated. Please check dataset or optimizer configuration.")
        return

    logger.info("Generating evaluation reports...")
    
    # (A) 打印摘要到控制台
    tracker.log_summary()
    
    # (B) 保存 CSV 明细表
    try:
        df = pd.DataFrame(results_list)
        csv_path = os.path.join(run_out_dir, "eval_details.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Detailed CSV report saved: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV report: {e}")

    # (C) 保存 Metrics 摘要 JSON
    try:
        summary_dict = tracker.compute_summary()
        # 确保 numpy 类型转为 python native，否则 json dump 会报错
        summary_safe = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                        for k, v in summary_dict.items()}
        
        json_path = os.path.join(run_out_dir, "metrics_summary.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_safe, f, indent=4)
        logger.info(f"Metrics summary saved: {json_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON summary: {e}")

    logger.info("Evaluation Process Finished.")

if __name__ == "__main__":
    main()