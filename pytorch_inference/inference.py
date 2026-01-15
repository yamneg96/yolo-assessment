"""
PyTorch inference script for YOLO models.

This script loads a YOLO PyTorch model (.pt) and runs inference on multiple input images.
It processes images, performs object detection, and saves the results with bounding boxes.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# Add utils directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "utils"))
try:
    from visualization import draw_bounding_boxes
    from preprocessing import preprocess_image
except ImportError:
    # Fallback if utils not yet created
    def draw_bounding_boxes(image, boxes, scores, labels, class_names):
        """Fallback visualization function."""
        return image
    
    def preprocess_image(image, input_size=(640, 640)):
        """Fallback preprocessing function."""
        return cv2.resize(image, input_size), 1.0

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PyTorchInference:
    """
    PyTorch-based YOLO inference handler.
    
    This class encapsulates all PyTorch inference operations including:
    - Model loading
    - Image preprocessing
    - Inference execution
    - Result processing and saving
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.25, 
                 iou_threshold: float = 0.45):
        """
        Initialize the PyTorch inference handler.
        
        Args:
            model_path: Path to the YOLO .pt model file
            confidence_threshold: Minimum confidence score for detections
            iou_threshold: IoU threshold for non-maximum suppression
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.class_names = []
        
        logger.info(f"Initializing PyTorch inference with model: {model_path}")
        
    def load_model(self) -> None:
        """Load the YOLO model from disk."""
        try:
            logger.info("Loading PyTorch YOLO model...")
            
            # Validate model file exists
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load YOLO model using Ultralytics
            self.model = YOLO(str(self.model_path))
            
            # Extract class names from the model
            self.class_names = self.model.names
            
            logger.info(f"Model loaded successfully. Classes: {len(self.class_names)}")
            logger.info(f"Class names: {list(self.class_names.values())[:10]}...")  # Show first 10
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def preprocess_images(self, image_paths: List[str]) -> Tuple[List[np.ndarray], List[float]]:
        """
        Preprocess a list of images for inference.
        
        Args:
            image_paths: List of paths to input images
            
        Returns:
            Tuple of (preprocessed_images, scale_factors)
        """
        logger.info(f"Preprocessing {len(image_paths)} images...")
        
        preprocessed_images = []
        scale_factors = []
        
        for img_path in tqdm(image_paths, desc="Preprocessing images"):
            try:
                # Read image
                image = cv2.imread(img_path)
                if image is None:
                    logger.warning(f"Failed to read image: {img_path}")
                    continue
                
                # Get original dimensions for scaling
                original_shape = image.shape[:2]  # (height, width)
                
                # Preprocess image (resize, normalize, etc.)
                # For PyTorch YOLO, let Ultralytics handle preprocessing internally
                processed_image = image
                scale_factor = 1.0
                
                preprocessed_images.append(processed_image)
                scale_factors.append(scale_factor)
                
            except Exception as e:
                logger.error(f"Error preprocessing image {img_path}: {str(e)}")
                continue
        
        logger.info(f"Successfully preprocessed {len(preprocessed_images)} images")
        return preprocessed_images, scale_factors
    
    def run_inference(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Run inference on preprocessed images.
        
        Args:
            images: List of preprocessed images
            
        Returns:
            List of inference results containing boxes, scores, and labels
        """
        logger.info(f"Running PyTorch inference on {len(images)} images...")
        
        results = []
        
        for i, image in enumerate(tqdm(images, desc="Running inference")):
            try:
                # Run inference
                with torch.no_grad():
                    prediction = self.model(image, conf=self.confidence_threshold, 
                                          iou=self.iou_threshold, verbose=False)
                
                # Extract results
                result = prediction[0]  # Get first (and only) result
                
                # Convert to standard format
                detections = {
                    'boxes': result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([]),
                    'scores': result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([]),
                    'labels': result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.array([]),
                    'image_index': i
                }
                
                results.append(detections)
                
            except Exception as e:
                logger.error(f"Error during inference on image {i}: {str(e)}")
                # Add empty result for consistency
                results.append({
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'labels': np.array([]),
                    'image_index': i
                })
        
        logger.info("PyTorch inference completed")
        return results
    
    def save_results(self, image_paths: List[str], results: List[Dict[str, Any]], 
                    output_dir: str, scale_factors: List[float] = None) -> None:
        """
        Save inference results with annotated images.
        
        Args:
            image_paths: Original image paths
            results: Inference results
            output_dir: Directory to save annotated images
            scale_factors: Scale factors for each image
        """
        logger.info(f"Saving results to {output_dir}...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detection results as JSON
        import json
        detection_data = []
        
        for i, (img_path, result) in enumerate(zip(image_paths, results)):
            try:
                # Read original image
                image = cv2.imread(img_path)
                if image is None:
                    logger.warning(f"Failed to read image for saving: {img_path}")
                    continue
                
                # Scale boxes back to original image size if scale factors provided
                boxes = result['boxes'].copy()
                if scale_factors and i < len(scale_factors):
                    scale = scale_factors[i]
                    boxes = boxes / scale  # Scale back coordinates
                
                # Draw bounding boxes on image
                annotated_image = draw_bounding_boxes(
                    image, boxes, result['scores'], result['labels'], self.class_names
                )
                
                # Save annotated image
                output_filename = f"annotated_{Path(img_path).stem}.jpg"
                output_filepath = output_path / output_filename
                cv2.imwrite(str(output_filepath), annotated_image)
                
                # Prepare detection data for JSON
                detection_data.append({
                    'image_path': str(img_path),
                    'output_path': str(output_filepath),
                    'detections': [
                        {
                            'box': box.tolist(),
                            'score': float(score),
                            'label': int(label),
                            'class_name': self.class_names.get(int(label), 'unknown')
                        }
                        for box, score, label in zip(boxes, result['scores'], result['labels'])
                    ],
                    'num_detections': len(boxes)
                })
                
            except Exception as e:
                logger.error(f"Error saving results for image {img_path}: {str(e)}")
                continue
        
        # Save detection data to JSON
        json_output_path = output_path / "detections.json"
        with open(json_output_path, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        logger.info(f"Results saved. Annotated images: {len(detection_data)}")
        logger.info(f"Detection data saved to: {json_output_path}")


def main():
    """Main function to run PyTorch inference."""
    # Configuration
    MODEL_PATH = "yolo11n.pt"
    IMAGE_DIR = "images"
    OUTPUT_DIR = "outputs/pytorch"
    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    
    logger.info("Starting PyTorch YOLO inference pipeline")
    
    try:
        # Initialize inference handler
        inference = PyTorchInference(
            model_path=MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD
        )
        
        # Load model
        inference.load_model()
        
        # Get image paths
        image_dir = Path(IMAGE_DIR)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [
            str(p) for p in image_dir.iterdir() 
            if p.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            logger.warning(f"No images found in {IMAGE_DIR}")
            return
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        # Preprocess images
        preprocessed_images, scale_factors = inference.preprocess_images(image_paths)
        
        # Run inference
        results = inference.run_inference(preprocessed_images)
        
        # Save results
        inference.save_results(image_paths, results, OUTPUT_DIR, scale_factors)
        
        logger.info("PyTorch inference pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
