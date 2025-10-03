#!/usr/bin/env python3
"""
Utility functions for landmark navigation system
"""

import os
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def get_best_gpu():
    """
    Get the best available GPU device.
    
    Returns:
        tuple: (gpu_id, device_name, memory_gb) or (None, "CPU", 0) if no GPU available
    """
    if not torch.cuda.is_available():
        return None, "CPU", 0

    gpu_id = 0
    props = torch.cuda.get_device_properties(gpu_id)
    memory_gb = props.total_memory / 1024 ** 3

    return gpu_id, props.name, memory_gb


def set_gpu_device(gpu_id):
    """
    Set GPU device and clear CUDA cache.
    
    Args:
        gpu_id: GPU device ID to use, or None for CPU
        
    Returns:
        str: Device string (e.g., "cuda:0" or "cpu")
    """
    if torch.cuda.is_available() and gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        return f"cuda:{gpu_id}"
    return "cpu"


def get_safe_output_dir():
    """
    Create and return a timestamped output directory.
    
    Returns:
        Path: Path object pointing to the created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./results/landmark_navigation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def monitor_gpu_memory():
    """
    Monitor current GPU memory usage.
    
    Returns:
        dict: Dictionary containing GPU memory statistics, or None if CUDA unavailable
    """
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(current_device) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(current_device) / 1024 ** 3
        total = torch.cuda.get_device_properties(current_device).total_memory / 1024 ** 3

        return {
            'device_id': current_device,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'usage_percent': (reserved / total) * 100,
        }
    return None


def resize_image_if_needed(image: np.ndarray, min_size: int = 192, max_size: int = 384):
    """
    Resize image to fit within size constraints while maintaining aspect ratio.
    
    Args:
        image: Input image as numpy array
        min_size: Minimum dimension size (default: 192)
        max_size: Maximum dimension size (default: 384)
        
    Returns:
        tuple: (resized_image, scale_factor)
    """
    h, w = image.shape[:2]
    original_min = min(h, w)
    original_max = max(h, w)

    scale_factor = 1.0
    resize_needed = False

    if original_min < min_size:
        scale_factor = min_size / original_min
        resize_needed = True
    elif original_max > max_size:
        scale_factor = max_size / original_max
        resize_needed = True

    if resize_needed:
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)

        if scale_factor > 1.0:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_AREA

        resized_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        return resized_image, scale_factor
    else:
        return image, scale_factor


def validate_dataset_path(dataset_path: str):
    """
    Validate dataset path and retrieve image files, excluding checkpoint files.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        tuple: (is_valid, list_of_image_paths)
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        return False, []

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(list(dataset_path.glob(f"**/{ext}")))
        image_paths.extend(list(dataset_path.glob(f"**/{ext.upper()}")))

    filtered_paths = []
    for path in image_paths:
        if "checkpoint" not in path.stem.lower():
            filtered_paths.append(path)

    return len(filtered_paths) > 0, sorted(set(filtered_paths))


def setup_environment():
    """
    Configure environment variables for model caching and GPU visibility.
    """
    os.environ['HF_HOME'] = '/root/autodl-tmp/model/huggingface'
    os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/model/transformers'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'