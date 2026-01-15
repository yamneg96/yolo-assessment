"""
Visualization utilities for YOLO inference results.

This module provides functions to draw bounding boxes, labels, and confidence scores
on images, making it easy to visualize and debug detection results.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


def draw_bounding_boxes(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, 
                       labels: np.ndarray, class_names: Dict[int, str],
                       color: Tuple[int, int, int] = (0, 255, 0), 
                       thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes, labels, and confidence scores on an image.
    
    Args:
        image: Input image as numpy array (BGR format)
        boxes: Array of bounding boxes in format [x1, y1, x2, y2]
        scores: Array of confidence scores
        labels: Array of class labels
        class_names: Dictionary mapping class IDs to names
        color: Box color in BGR format
        thickness: Line thickness for boxes
        
    Returns:
        Image with drawn bounding boxes and labels
    """
    # Create a copy to avoid modifying the original
    annotated_image = image.copy()
    
    # Define colors for different classes (optional)
    colors = generate_colors(len(class_names))
    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        try:
            # Convert box coordinates to integers
            x1, y1, x2, y2 = map(int, box)
            
            # Get color for this class
            if isinstance(class_names, dict):
                class_color = colors[int(label) % len(colors)] if len(colors) > 0 else color
                class_name = class_names.get(int(label), f"class_{int(label)}")
            elif isinstance(class_names, list):
                class_color = colors[int(label) % len(colors)] if len(colors) > 0 else color
                class_name = class_names[int(label)] if int(label) < len(class_names) else f"class_{int(label)}"
            else:
                class_color = color
                class_name = f"class_{int(label)}"
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), class_color, thickness)
            
            # Prepare label text
            label_text = f"{class_name}: {score:.2f}"
            
            # Calculate text size and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
            
            # Draw background rectangle for text
            text_x = x1
            text_y = y1 - 10 if y1 > 30 else y1 + text_size[1] + 10
            
            cv2.rectangle(
                annotated_image,
                (text_x, text_y - text_size[1] - 5),
                (text_x + text_size[0], text_y + 5),
                class_color,
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                annotated_image,
                label_text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),  # White text
                font_thickness,
                cv2.LINE_AA
            )
            
        except Exception as e:
            print(f"Error drawing box {i}: {str(e)}")
            continue
    
    return annotated_image


def generate_colors(num_classes: int) -> List[Tuple[int, int, int]]:
    """
    Generate distinct colors for different classes.
    
    Args:
        num_classes: Number of classes to generate colors for
        
    Returns:
        List of RGB color tuples
    """
    colors = []
    for i in range(num_classes):
        # Generate colors using HSV color space for better distribution
        hue = i * 360 / num_classes
        saturation = 0.8
        value = 0.9
        
        # Convert HSV to RGB
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue/360, saturation, value)
        
        # Convert to BGR for OpenCV
        colors.append((int(b*255), int(g*255), int(r*255)))
    
    return colors


def visualize_detection_comparison(original_image: np.ndarray, 
                                 pytorch_results: Dict[str, Any],
                                 onnx_results: Dict[str, Any],
                                 class_names: Dict[int, str],
                                 save_path: str = None) -> None:
    """
    Visualize side-by-side comparison of PyTorch and ONNX detection results.
    
    Args:
        original_image: Original input image
        pytorch_results: PyTorch detection results
        onnx_results: ONNX detection results
        class_names: Dictionary mapping class IDs to names
        save_path: Path to save the comparison image
    """
    # Draw results for both methods
    pytorch_annotated = draw_bounding_boxes(
        original_image, 
        pytorch_results['boxes'], 
        pytorch_results['scores'], 
        pytorch_results['labels'], 
        class_names,
        color=(0, 255, 0)  # Green for PyTorch
    )
    
    onnx_annotated = draw_bounding_boxes(
        original_image, 
        onnx_results['boxes'], 
        onnx_results['scores'], 
        onnx_results['labels'], 
        class_names,
        color=(255, 0, 0)  # Blue for ONNX
    )
    
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Convert BGR to RGB for matplotlib
    pytorch_rgb = cv2.cvtColor(pytorch_annotated, cv2.COLOR_BGR2RGB)
    onnx_rgb = cv2.cvtColor(onnx_annotated, cv2.COLOR_BGR2RGB)
    
    ax1.imshow(pytorch_rgb)
    ax1.set_title(f'PyTorch Results ({len(pytorch_results["boxes"])} detections)')
    ax1.axis('off')
    
    ax2.imshow(onnx_rgb)
    ax2.set_title(f'ONNX Results ({len(onnx_results["boxes"])} detections)')
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
    
    plt.show()


def plot_detection_statistics(pytorch_results: List[Dict[str, Any]], 
                            onnx_results: List[Dict[str, Any]],
                            save_path: str = None) -> None:
    """
    Plot statistics comparing PyTorch and ONNX detection results.
    
    Args:
        pytorch_results: List of PyTorch detection results
        onnx_results: List of ONNX detection results
        save_path: Path to save the statistics plot
    """
    # Extract statistics
    pytorch_counts = [len(result['boxes']) for result in pytorch_results]
    onnx_counts = [len(result['boxes']) for result in onnx_results]
    
    # Calculate confidence statistics
    pytorch_confidences = []
    onnx_confidences = []
    
    for result in pytorch_results:
        pytorch_confidences.extend(result['scores'].tolist())
    
    for result in onnx_results:
        onnx_confidences.extend(result['scores'].tolist())
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Detection counts per image
    ax1.bar(range(len(pytorch_counts)), pytorch_counts, alpha=0.7, label='PyTorch')
    ax1.bar(range(len(onnx_counts)), onnx_counts, alpha=0.7, label='ONNX')
    ax1.set_xlabel('Image Index')
    ax1.set_ylabel('Number of Detections')
    ax1.set_title('Detection Counts per Image')
    ax1.legend()
    
    # Plot 2: Confidence score distribution
    ax2.hist(pytorch_confidences, bins=20, alpha=0.7, label='PyTorch', density=True)
    ax2.hist(onnx_confidences, bins=20, alpha=0.7, label='ONNX', density=True)
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Confidence Score Distribution')
    ax2.legend()
    
    # Plot 3: Total detections comparison
    total_pytorch = sum(pytorch_counts)
    total_onnx = sum(onnx_counts)
    ax3.bar(['PyTorch', 'ONNX'], [total_pytorch, total_onnx], color=['green', 'blue'])
    ax3.set_ylabel('Total Detections')
    ax3.set_title('Total Detections Comparison')
    
    # Plot 4: Average confidence per image
    avg_pytorch_conf = [np.mean(scores) if len(scores) > 0 else 0 for scores in 
                       [result['scores'] for result in pytorch_results]]
    avg_onnx_conf = [np.mean(scores) if len(scores) > 0 else 0 for scores in 
                    [result['scores'] for result in onnx_results]]
    
    ax4.plot(range(len(avg_pytorch_conf)), avg_pytorch_conf, 'o-', label='PyTorch')
    ax4.plot(range(len(avg_onnx_conf)), avg_onnx_conf, 'o-', label='ONNX')
    ax4.set_xlabel('Image Index')
    ax4.set_ylabel('Average Confidence')
    ax4.set_title('Average Confidence per Image')
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Statistics plot saved to: {save_path}")
    
    plt.show()


def create_detection_summary_image(image: np.ndarray, results: Dict[str, Any], 
                                 class_names: Dict[int, str], 
                                 method_name: str = "Detection") -> np.ndarray:
    """
    Create a summary image with detection results and statistics.
    
    Args:
        image: Original image
        results: Detection results
        class_names: Dictionary mapping class IDs to names
        method_name: Name of the detection method
        
    Returns:
        Summary image with annotations and statistics
    """
    # Draw bounding boxes
    annotated = draw_bounding_boxes(image, results['boxes'], results['scores'], 
                                  results['labels'], class_names)
    
    # Add statistics text
    num_detections = len(results['boxes'])
    avg_confidence = np.mean(results['scores']) if len(results['scores']) > 0 else 0
    
    # Count detections by class
    class_counts = {}
    for label in results['labels']:
        class_name = class_names.get(int(label), f"class_{int(label)}")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Create statistics text
    stats_text = [
        f"{method_name} Results:",
        f"Total Detections: {num_detections}",
        f"Average Confidence: {avg_confidence:.3f}",
        "",
        "Detections by Class:"
    ]
    
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        stats_text.append(f"  {class_name}: {count}")
    
    # Add text to image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    line_height = 25
    
    # Create semi-transparent background for text
    text_bg_height = len(stats_text) * line_height + 20
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (350, text_bg_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
    
    # Add text
    for i, line in enumerate(stats_text):
        y_pos = 30 + i * line_height
        cv2.putText(annotated, line, (20, y_pos), font, font_scale, 
                   (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    return annotated


def save_annotated_images_batch(image_paths: List[str], results_list: List[Dict[str, Any]], 
                               class_names: Dict[int, str], output_dir: str, 
                               prefix: str = "annotated") -> None:
    """
    Save a batch of annotated images.
    
    Args:
        image_paths: List of original image paths
        results_list: List of detection results
        class_names: Dictionary mapping class IDs to names
        output_dir: Directory to save annotated images
        prefix: Prefix for output filenames
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path, results in zip(image_paths, results_list):
        try:
            # Read original image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to read image: {img_path}")
                continue
            
            # Draw annotations
            annotated = draw_bounding_boxes(image, results['boxes'], results['scores'], 
                                         results['labels'], class_names)
            
            # Save annotated image
            filename = f"{prefix}_{os.path.basename(img_path)}"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, annotated)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
