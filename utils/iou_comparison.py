"""
IoU computation and comparison utilities for PyTorch vs ONNX results.

This module provides functions to compute Intersection over Union (IoU) between
detections and compare results from different inference backends.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path
import matplotlib.pyplot as plt


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of bounding boxes.
    
    Args:
        boxes1: First set of boxes [N, 4] in format [x1, y1, x2, y2]
        boxes2: Second set of boxes [M, 4] in format [x1, y1, x2, y2]
        
    Returns:
        IoU matrix of shape [N, M] where element (i, j) is IoU between boxes1[i] and boxes2[j]
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.array([])
    
    # Expand dimensions for broadcasting
    boxes1_expanded = np.expand_dims(boxes1, axis=1)  # [N, 1, 4]
    boxes2_expanded = np.expand_dims(boxes2, axis=0)  # [1, M, 4]
    
    # Calculate intersection coordinates
    x1 = np.maximum(boxes1_expanded[:, :, 0], boxes2_expanded[:, :, 0])
    y1 = np.maximum(boxes1_expanded[:, :, 1], boxes2_expanded[:, :, 1])
    x2 = np.minimum(boxes1_expanded[:, :, 2], boxes2_expanded[:, :, 2])
    y2 = np.minimum(boxes1_expanded[:, :, 3], boxes2_expanded[:, :, 3])
    
    # Calculate intersection area
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Calculate individual areas
    area1 = (boxes1_expanded[:, :, 2] - boxes1_expanded[:, :, 0]) * \
            (boxes1_expanded[:, :, 3] - boxes1_expanded[:, :, 1])
    area2 = (boxes2_expanded[:, :, 2] - boxes2_expanded[:, :, 0]) * \
            (boxes2_expanded[:, :, 3] - boxes2_expanded[:, :, 1])
    
    # Calculate union area
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    union = np.maximum(union, 1e-7)
    
    return intersection / union


def match_detections_greedy(boxes1: np.ndarray, boxes2: np.ndarray, 
                           iou_threshold: float = 0.5) -> Tuple[List[int], List[int], List[float]]:
    """
    Match detections between two sets of boxes using greedy IoU matching.
    
    Args:
        boxes1: First set of boxes [N, 4]
        boxes2: Second set of boxes [M, 4]
        iou_threshold: Minimum IoU threshold for matching
        
    Returns:
        Tuple of (indices1, indices2, ious)
        - indices1: Indices of matched boxes from first set
        - indices2: Indices of matched boxes from second set
        - ious: IoU values for each match
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return [], [], []
    
    # Compute IoU matrix
    iou_matrix = compute_iou_matrix(boxes1, boxes2)
    
    # Greedy matching
    matched_indices1 = []
    matched_indices2 = []
    matched_ious = []
    
    used1 = set()
    used2 = set()
    
    while True:
        # Find the highest IoU pair
        if iou_matrix.size == 0:
            break
            
        max_iou = np.max(iou_matrix)
        if max_iou < iou_threshold:
            break
            
        # Get indices of max IoU
        idx1, idx2 = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        
        # Skip if already used
        if idx1 in used1 or idx2 in used2:
            iou_matrix[idx1, idx2] = 0
            continue
        
        # Add to matches
        matched_indices1.append(idx1)
        matched_indices2.append(idx2)
        matched_ious.append(max_iou)
        
        # Mark as used
        used1.add(idx1)
        used2.add(idx2)
        
        # Zero out this row and column
        iou_matrix[idx1, :] = 0
        iou_matrix[:, idx2] = 0
    
    return matched_indices1, matched_indices2, matched_ious


def compare_detections(pytorch_results: List[Dict[str, Any]], 
                      onnx_results: List[Dict[str, Any]],
                      iou_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compare detection results between PyTorch and ONNX inference.
    
    Args:
        pytorch_results: List of PyTorch detection results
        onnx_results: List of ONNX detection results
        iou_threshold: IoU threshold for matching detections
        
    Returns:
        Dictionary containing comparison metrics and statistics
    """
    comparison_stats = {
        'total_images': len(pytorch_results),
        'iou_threshold': iou_threshold,
        'image_comparisons': [],
        'summary': {
            'total_pytorch_detections': 0,
            'total_onnx_detections': 0,
            'total_matches': 0,
            'average_iou': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    }
    
    all_ious = []
    total_pytorch_detections = 0
    total_onnx_detections = 0
    total_matches = 0
    
    for img_idx, (pt_result, onnx_result) in enumerate(zip(pytorch_results, onnx_results)):
        pt_boxes = pt_result['boxes']
        onnx_boxes = onnx_result['boxes']
        pt_scores = pt_result['scores']
        onnx_scores = onnx_result['scores']
        
        num_pt = len(pt_boxes)
        num_onnx = len(onnx_boxes)
        
        # Match detections
        pt_indices, onnx_indices, ious = match_detections_greedy(
            pt_boxes, onnx_boxes, iou_threshold
        )
        
        # Calculate image-level statistics
        img_stats = {
            'image_index': img_idx,
            'pytorch_detections': num_pt,
            'onnx_detections': num_onnx,
            'matches': len(pt_indices),
            'iou_matches': ious,
            'average_iou': np.mean(ious) if ious else 0.0,
            'precision': len(pt_indices) / num_onnx if num_onnx > 0 else 0.0,
            'recall': len(pt_indices) / num_pt if num_pt > 0 else 0.0
        }
        
        # Add confidence comparisons for matched detections
        confidence_diffs = []
        for pt_idx, onnx_idx in zip(pt_indices, onnx_indices):
            conf_diff = abs(pt_scores[pt_idx] - onnx_scores[onnx_idx])
            confidence_diffs.append(conf_diff)
        
        img_stats['confidence_differences'] = confidence_diffs
        img_stats['avg_confidence_difference'] = np.mean(confidence_diffs) if confidence_diffs else 0.0
        
        comparison_stats['image_comparisons'].append(img_stats)
        
        # Update totals
        total_pytorch_detections += num_pt
        total_onnx_detections += num_onnx
        total_matches += len(pt_indices)
        all_ious.extend(ious)
    
    # Calculate overall statistics
    comparison_stats['summary']['total_pytorch_detections'] = total_pytorch_detections
    comparison_stats['summary']['total_onnx_detections'] = total_onnx_detections
    comparison_stats['summary']['total_matches'] = total_matches
    comparison_stats['summary']['average_iou'] = np.mean(all_ious) if all_ious else 0.0
    
    # Calculate precision, recall, F1
    if total_onnx_detections > 0:
        comparison_stats['summary']['precision'] = total_matches / total_onnx_detections
    if total_pytorch_detections > 0:
        comparison_stats['summary']['recall'] = total_matches / total_pytorch_detections
    
    precision = comparison_stats['summary']['precision']
    recall = comparison_stats['summary']['recall']
    if precision + recall > 0:
        comparison_stats['summary']['f1_score'] = 2 * (precision * recall) / (precision + recall)
    
    return comparison_stats


def save_comparison_results(comparison_stats: Dict[str, Any], output_path: str) -> None:
    """
    Save comparison results to JSON file.
    
    Args:
        comparison_stats: Comparison statistics dictionary
        output_path: Path to save the results
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    # Deep copy and convert
    import copy
    json_stats = copy.deepcopy(comparison_stats)
    json_stats = json.loads(json.dumps(json_stats, default=convert_numpy))
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(json_stats, f, indent=2)
    
    print(f"Comparison results saved to: {output_path}")


def plot_comparison_metrics(comparison_stats: Dict[str, Any], save_path: str = None) -> None:
    """
    Plot comparison metrics between PyTorch and ONNX results.
    
    Args:
        comparison_stats: Comparison statistics dictionary
        save_path: Path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data for plotting
    image_comparisons = comparison_stats['image_comparisons']
    
    # Plot 1: Detection counts per image
    pt_counts = [img['pytorch_detections'] for img in image_comparisons]
    onnx_counts = [img['onnx_detections'] for img in image_comparisons]
    
    x = range(len(image_comparisons))
    ax1.plot(x, pt_counts, 'o-', label='PyTorch', color='green')
    ax1.plot(x, onnx_counts, 'o-', label='ONNX', color='blue')
    ax1.set_xlabel('Image Index')
    ax1.set_ylabel('Number of Detections')
    ax1.set_title('Detection Counts per Image')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: IoU distribution for matched detections
    all_ious = []
    for img in image_comparisons:
        all_ious.extend(img['iou_matches'])
    
    if all_ious:
        ax2.hist(all_ious, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('IoU Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'IoU Distribution for Matched Detections\n(Avg IoU: {np.mean(all_ious):.3f})')
        ax2.axvline(np.mean(all_ious), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_ious):.3f}')
        ax2.legend()
    
    # Plot 3: Precision and Recall per image
    precisions = [img['precision'] for img in image_comparisons]
    recalls = [img['recall'] for img in image_comparisons]
    
    ax3.plot(x, precisions, 'o-', label='Precision', color='purple')
    ax3.plot(x, recalls, 'o-', label='Recall', color='red')
    ax3.set_xlabel('Image Index')
    ax3.set_ylabel('Score')
    ax3.set_title('Precision and Recall per Image')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)
    
    # Plot 4: Summary statistics bar chart
    summary = comparison_stats['summary']
    metrics = ['Total PyTorch', 'Total ONNX', 'Total Matches']
    values = [summary['total_pytorch_detections'], 
              summary['total_onnx_detections'], 
              summary['total_matches']]
    colors = ['green', 'blue', 'orange']
    
    bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
    ax4.set_ylabel('Count')
    ax4.set_title('Overall Detection Summary')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}', ha='center', va='bottom')
    
    # Add precision/recall/F1 text
    info_text = f"Precision: {summary['precision']:.3f}\n"
    info_text += f"Recall: {summary['recall']:.3f}\n"
    info_text += f"F1 Score: {summary['f1_score']:.3f}\n"
    info_text += f"Avg IoU: {summary['average_iou']:.3f}"
    
    ax4.text(0.02, 0.98, info_text, transform=ax4.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()


def generate_comparison_report(comparison_stats: Dict[str, Any]) -> str:
    """
    Generate a text report summarizing the comparison results.
    
    Args:
        comparison_stats: Comparison statistics dictionary
        
    Returns:
        Formatted text report
    """
    summary = comparison_stats['summary']
    
    report = f"""
YOLO PyTorch vs ONNX Inference Comparison Report
===============================================

Summary Statistics:
- Total Images Processed: {comparison_stats['total_images']}
- IoU Threshold for Matching: {comparison_stats['iou_threshold']}

Detection Counts:
- Total PyTorch Detections: {summary['total_pytorch_detections']}
- Total ONNX Detections: {summary['total_onnx_detections']}
- Total Matches: {summary['total_matches']}

Performance Metrics:
- Average IoU (matched): {summary['average_iou']:.4f}
- Precision: {summary['precision']:.4f}
- Recall: {summary['recall']:.4f}
- F1 Score: {summary['f1_score']:.4f}

Per-Image Statistics:
"""
    
    # Add per-image details
    for i, img_stats in enumerate(comparison_stats['image_comparisons']):
        report += f"""
Image {i+1}:
- PyTorch: {img_stats['pytorch_detections']} detections
- ONNX: {img_stats['onnx_detections']} detections
- Matches: {img_stats['matches']}
- Avg IoU: {img_stats['average_iou']:.4f}
- Precision: {img_stats['precision']:.4f}
- Recall: {img_stats['recall']:.4f}
- Avg Confidence Diff: {img_stats['avg_confidence_difference']:.4f}
"""
    
    return report


def run_full_comparison(pytorch_json_path: str, onnx_json_path: str, 
                       output_dir: str, iou_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Run a complete comparison between PyTorch and ONNX results.
    
    Args:
        pytorch_json_path: Path to PyTorch results JSON
        onnx_json_path: Path to ONNX results JSON
        output_dir: Directory to save comparison results
        iou_threshold: IoU threshold for matching
        
    Returns:
        Comparison statistics dictionary
    """
    # Load results
    with open(pytorch_json_path, 'r') as f:
        pytorch_data = json.load(f)
    
    with open(onnx_json_path, 'r') as f:
        onnx_data = json.load(f)
    
    # Extract detection results
    pytorch_results = []
    onnx_results = []
    
    for item in pytorch_data:
        detections = item['detections']
        if detections:
            boxes = np.array([det['box'] for det in detections])
            scores = np.array([det['score'] for det in detections])
            labels = np.array([det['label'] for det in detections])
        else:
            boxes = np.array([])
            scores = np.array([])
            labels = np.array([])
        
        pytorch_results.append({
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        })
    
    for item in onnx_data:
        detections = item['detections']
        if detections:
            boxes = np.array([det['box'] for det in detections])
            scores = np.array([det['score'] for det in detections])
            labels = np.array([det['label'] for det in detections])
        else:
            boxes = np.array([])
            scores = np.array([])
            labels = np.array([])
        
        onnx_results.append({
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        })
    
    # Run comparison
    comparison_stats = compare_detections(pytorch_results, onnx_results, iou_threshold)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    json_path = output_path / "comparison_results.json"
    save_comparison_results(comparison_stats, str(json_path))
    
    # Save text report
    report = generate_comparison_report(comparison_stats)
    report_path = output_path / "comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save plots
    plot_path = output_path / "comparison_plots.png"
    plot_comparison_metrics(comparison_stats, str(plot_path))
    
    print(f"Comparison completed. Results saved to: {output_dir}")
    
    return comparison_stats
