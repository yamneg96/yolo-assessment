"""
Post-processing utilities for YOLO inference results.

This module provides functions for processing raw YOLO outputs, including
non-maximum suppression, coordinate transformations, and result formatting.
"""

import numpy as np
from typing import Tuple, List
import cv2


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, 
                       score_threshold: float = 0.25, iou_threshold: float = 0.45) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply non-maximum suppression to remove duplicate detections.
    
    Args:
        boxes: Array of bounding boxes in format [x1, y1, x2, y2]
        scores: Array of confidence scores
        score_threshold: Minimum confidence score to keep
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Tuple of (filtered_boxes, filtered_scores, filtered_labels)
    """
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Filter by confidence threshold
    valid_indices = scores > score_threshold
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Convert boxes to format for NMS (x1, y1, x2, y2 -> x1, y1, w, h)
    boxes_for_nms = np.copy(boxes)
    boxes_for_nms[:, 2] = boxes_for_nms[:, 2] - boxes_for_nms[:, 0]  # width
    boxes_for_nms[:, 3] = boxes_for_nms[:, 3] - boxes_for_nms[:, 1]  # height
    
    # Apply OpenCV NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_for_nms.tolist(), 
        scores.tolist(), 
        score_threshold, 
        iou_threshold
    )
    
    if len(indices) > 0:
        indices = indices.flatten()
        filtered_boxes = boxes[indices]
        filtered_scores = scores[indices]
        filtered_labels = np.arange(len(filtered_boxes))  # Placeholder labels
    else:
        filtered_boxes = np.array([])
        filtered_scores = np.array([])
        filtered_labels = np.array([])
    
    return filtered_boxes, filtered_scores, filtered_labels


def process_yolo_output(output: np.ndarray, original_shape: Tuple[int, int], 
                       input_shape: Tuple[int, int] = (640, 640),
                       conf_threshold: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process raw YOLO model output to extract detections.
    
    This function handles the standard YOLO output format and transforms
    coordinates from model input space back to original image space.
    
    Args:
        output: Raw model output tensor
        original_shape: Original image shape (height, width)
        input_shape: Model input shape (height, width)
        conf_threshold: Confidence threshold for filtering
        
    Returns:
        Tuple of (boxes, scores, labels)
        - boxes: Bounding boxes in format [x1, y1, x2, y2]
        - scores: Confidence scores
        - labels: Class labels
    """
    # Handle different output shapes
    if len(output.shape) == 3:
        output = output.squeeze(0)  # Remove batch dimension
    
    if len(output.shape) == 2:
        # Standard YOLO format: [num_detections, 5 + num_classes]
        # Format: [x_center, y_center, width, height, confidence, class_probs...]
        
        # Extract boxes and confidence
        boxes = output[:, :4]  # [x_center, y_center, width, height]
        obj_conf = output[:, 4]  # objectness confidence
        
        # Extract class probabilities and find best class
        class_probs = output[:, 5:]
        class_conf = np.max(class_probs, axis=1)
        class_labels = np.argmax(class_probs, axis=1)
        
        # Calculate final confidence
        final_conf = obj_conf * class_conf
        
        # Filter by confidence threshold
        valid_mask = final_conf > conf_threshold
        boxes = boxes[valid_mask]
        final_conf = final_conf[valid_mask]
        class_labels = class_labels[valid_mask]
        
        # Convert from center format to corner format
        # [x_center, y_center, width, height] -> [x1, y1, x2, y2]
        x_center, y_center, width, height = boxes.T
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        boxes = np.column_stack([x1, y1, x2, y2])
        
        # Scale coordinates back to original image size
        boxes = scale_boxes(boxes, input_shape, original_shape)
        
        return boxes, final_conf, class_labels
    
    else:
        # Unexpected output format
        return np.array([]), np.array([]), np.array([])


def scale_boxes(boxes: np.ndarray, input_shape: Tuple[int, int], 
               original_shape: Tuple[int, int]) -> np.ndarray:
    """
    Scale bounding boxes from model input space to original image space.
    
    Args:
        boxes: Bounding boxes in format [x1, y1, x2, y2] in input space
        input_shape: Model input shape (height, width)
        original_shape: Original image shape (height, width)
        
    Returns:
        Scaled bounding boxes in original image space
    """
    if len(boxes) == 0:
        return boxes
    
    # Calculate scale factors
    input_h, input_w = input_shape
    orig_h, orig_w = original_shape
    
    # Calculate the scaling ratio (maintain aspect ratio)
    scale = min(input_w / orig_w, input_h / orig_h)
    
    # Calculate padding
    pad_x = (input_w - orig_w * scale) / 2
    pad_y = (input_h - orig_h * scale) / 2
    
    # Scale boxes back to original image space
    scaled_boxes = boxes.copy()
    
    # Remove padding and scale
    scaled_boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale  # x coordinates
    scaled_boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale  # y coordinates
    
    # Clip to image boundaries
    scaled_boxes[:, [0, 2]] = np.clip(scaled_boxes[:, [0, 2]], 0, orig_w)
    scaled_boxes[:, [1, 3]] = np.clip(scaled_boxes[:, [1, 3]], 0, orig_h)
    
    return scaled_boxes


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculate IoU between two sets of bounding boxes.
    
    Args:
        boxes1: Array of bounding boxes [N, 4]
        boxes2: Array of bounding boxes [M, 4]
        
    Returns:
        IoU matrix of shape [N, M]
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.array([])
    
    # Expand dimensions for broadcasting
    boxes1_expanded = np.expand_dims(boxes1, axis=1)  # [N, 1, 4]
    boxes2_expanded = np.expand_dims(boxes2, axis=0)  # [1, M, 4]
    
    # Calculate intersection
    x1 = np.maximum(boxes1_expanded[:, :, 0], boxes2_expanded[:, :, 0])
    y1 = np.maximum(boxes1_expanded[:, :, 1], boxes2_expanded[:, :, 1])
    x2 = np.minimum(boxes1_expanded[:, :, 2], boxes2_expanded[:, :, 2])
    y2 = np.minimum(boxes1_expanded[:, :, 3], boxes2_expanded[:, :, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Calculate union
    area1 = (boxes1_expanded[:, :, 2] - boxes1_expanded[:, :, 0]) * \
            (boxes1_expanded[:, :, 3] - boxes1_expanded[:, :, 1])
    area2 = (boxes2_expanded[:, :, 2] - boxes2_expanded[:, :, 0]) * \
            (boxes2_expanded[:, :, 3] - boxes2_expanded[:, :, 1])
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    union = np.maximum(union, 1e-7)
    
    return intersection / union


def match_detections(pytorch_boxes: np.ndarray, onnx_boxes: np.ndarray, 
                    iou_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match detections between PyTorch and ONNX results based on IoU.
    
    Args:
        pytorch_boxes: Bounding boxes from PyTorch inference
        onnx_boxes: Bounding boxes from ONNX inference
        iou_threshold: IoU threshold for matching
        
    Returns:
        Tuple of (pytorch_matches, onnx_matches)
        - pytorch_matches: Indices of matched PyTorch boxes
        - onnx_matches: Indices of matched ONNX boxes
    """
    if len(pytorch_boxes) == 0 or len(onnx_boxes) == 0:
        return np.array([]), np.array([])
    
    # Calculate IoU matrix
    iou_matrix = calculate_iou_batch(pytorch_boxes, onnx_boxes)
    
    # Find best matches
    pytorch_matches = []
    onnx_matches = []
    
    used_pytorch = set()
    used_onnx = set()
    
    # Greedy matching based on highest IoU
    while True:
        # Find highest IoU pair
        if iou_matrix.size == 0:
            break
            
        max_iou = np.max(iou_matrix)
        if max_iou < iou_threshold:
            break
            
        max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        pt_idx, onnx_idx = max_idx
        
        # Check if indices are already used
        if pt_idx in used_pytorch or onnx_idx in used_onnx:
            iou_matrix[pt_idx, onnx_idx] = 0
            continue
        
        # Add to matches
        pytorch_matches.append(pt_idx)
        onnx_matches.append(onnx_idx)
        used_pytorch.add(pt_idx)
        used_onnx.add(onnx_idx)
        
        # Mark this pair as used
        iou_matrix[pt_idx, :] = 0
        iou_matrix[:, onnx_idx] = 0
    
    return np.array(pytorch_matches), np.array(onnx_matches)


def format_detection_results(boxes: np.ndarray, scores: np.ndarray, 
                           labels: np.ndarray) -> dict:
    """
    Format detection results into a standardized dictionary.
    
    Args:
        boxes: Bounding boxes [N, 4]
        scores: Confidence scores [N]
        labels: Class labels [N]
        
    Returns:
        Formatted detection results dictionary
    """
    return {
        'boxes': boxes,
        'scores': scores,
        'labels': labels,
        'num_detections': len(boxes)
    }
