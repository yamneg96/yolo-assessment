"""
Image preprocessing utilities for YOLO inference.

This module provides standardized preprocessing functions for both PyTorch and ONNX inference,
ensuring consistent input formatting across different inference backends.
"""

import cv2
import numpy as np
from typing import Tuple


def preprocess_image(image: np.ndarray, input_size: Tuple[int, int] = (640, 640), 
                    normalize: bool = True) -> Tuple[np.ndarray, float]:
    """
    Preprocess an image for YOLO inference.
    
    This function handles:
    - Image resizing with aspect ratio preservation
    - Padding to maintain aspect ratio
    - Normalization to [0, 1] range
    - Channel ordering (BGR to RGB if needed)
    
    Args:
        image: Input image as numpy array (BGR format from OpenCV)
        input_size: Target size (width, height) for the model
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Tuple of (preprocessed_image, scale_factor)
        - preprocessed_image: Processed image ready for model input
        - scale_factor: Scale factor used for resizing (for coordinate scaling)
    """
    # Get original dimensions
    original_height, original_width = image.shape[:2]
    target_width, target_height = input_size
    
    # Calculate scale factor to maintain aspect ratio
    scale = min(target_width / original_width, target_height / original_height)
    
    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Create a square canvas with padding
    canvas = np.full((target_height, target_width, 3), 114, dtype=np.uint8)  # Gray padding
    
    # Calculate padding
    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2
    
    # Place resized image on canvas
    canvas[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
    
    # Convert BGR to RGB (YOLO models typically expect RGB)
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    
    # Normalize if requested
    if normalize:
        canvas_rgb = canvas_rgb.astype(np.float32) / 255.0
    
    return canvas_rgb, scale


def letterbox_image(image: np.ndarray, new_shape: Tuple[int, int] = (640, 640), 
                   color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize and pad image to maintain aspect ratio (letterboxing).
    
    This is the standard YOLO preprocessing approach that maintains aspect ratio
    by adding padding instead of stretching the image.
    
    Args:
        image: Input image as numpy array
        new_shape: Target size (width, height)
        color: Padding color in RGB format
        
    Returns:
        Tuple of (letterboxed_image, scale_factor, padding)
        - letterboxed_image: Resized and padded image
        - scale_factor: Scale factor used for resizing
        - padding: (pad_width, pad_height) tuple
    """
    # Get original shape
    shape = image.shape[:2]  # current shape [height, width]
    new_shape = (new_shape[1], new_shape[0])  # convert to [height, width]
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    # Minimum padding
    dw /= 2
    dh /= 2
    
    # Resize image
    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # Add padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # Convert color from RGB to BGR for OpenCV padding
    color_bgr = color[::-1]
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color_bgr)
    
    return image, r, (dw, dh)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to [0, 1] range.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert image from BGR to RGB format.
    
    Args:
        image: Input image in BGR format (from OpenCV)
        
    Returns:
        Image in RGB format
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prepare_batch_input(images: list, input_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Prepare a batch of images for model inference.
    
    Args:
        images: List of input images
        input_size: Target size for resizing
        
    Returns:
        Batch tensor ready for model input (N, C, H, W)
    """
    batch = []
    
    for image in images:
        # Preprocess each image
        processed, _ = preprocess_image(image, input_size, normalize=True)
        
        # Convert to CHW format
        chw = np.transpose(processed, (2, 0, 1))
        batch.append(chw)
    
    # Stack into batch
    batch_array = np.stack(batch, axis=0)
    
    return batch_array


def calculate_padding_info(original_shape: Tuple[int, int], target_shape: Tuple[int, int]) -> Tuple[float, Tuple[int, int]]:
    """
    Calculate padding information for coordinate transformation.
    
    Args:
        original_shape: Original image shape (height, width)
        target_shape: Target model input shape (height, width)
        
    Returns:
        Tuple of (scale_factor, padding)
        - scale_factor: Scale factor used for resizing
        - padding: (pad_width, pad_height) tuple
    """
    # Calculate scale factor
    scale = min(target_shape[1] / original_shape[1], target_shape[0] / original_shape[0])
    
    # Calculate new dimensions after scaling
    new_width = int(original_shape[1] * scale)
    new_height = int(original_shape[0] * scale)
    
    # Calculate padding
    pad_width = (target_shape[1] - new_width) / 2
    pad_height = (target_shape[0] - new_height) / 2
    
    return scale, (pad_width, pad_height)
