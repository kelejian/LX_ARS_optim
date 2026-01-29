# src/utils/logger.py
import logging
import sys
import os

def setup_logger(name: str = "LX-ARS", log_file: str = None, level: int = logging.INFO):
    """
    配置全局统一的 Logger
    Args:
        name: Logger 名称
        log_file: 日志文件路径 (可选)
        level: 日志等级
    """
    # 1. 获取 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False # 防止重复打印

    # 如果已经有 handler，说明已经配置过，直接返回
    if logger.handlers:
        return logger

    # 2. 格式化
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 3. 控制台输出 (StreamHandler)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 4. 文件输出 (FileHandler) - 可选
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger