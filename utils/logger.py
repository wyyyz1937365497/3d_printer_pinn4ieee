"""
Logging utilities
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(name: str,
                log_dir: str = "logs",
                level: int = logging.INFO,
                console: bool = True) -> logging.Logger:
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        console: Whether to add console handler

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Clear existing handlers

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger or create new one

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, setup with defaults
    if not logger.handlers:
        return setup_logger(name)

    return logger


class Logger:
    """
    Convenience class for logging with context
    """

    def __init__(self, name: str, log_dir: str = "logs"):
        self.logger = setup_logger(name, log_dir)
        self.name = name

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)

    def log_metrics(self, metrics: dict, step: int, prefix: str = ""):
        """Log training/validation metrics"""
        message = f"Step {step}"
        if prefix:
            message = f"{prefix} - {message}"

        for key, value in metrics.items():
            message += f" | {key}: {value:.6f}"

        self.info(message)

    def log_model_summary(self, model):
        """Log model summary"""
        import torch.nn as nn

        self.info(f"Model: {model.__class__.__name__}")
        self.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        self.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Log layer-wise parameters
        for name, layer in model.named_children():
            if isinstance(layer, nn.Module):
                num_params = sum(p.numel() for p in layer.parameters())
                self.info(f"  {name}: {num_params:,} parameters")
