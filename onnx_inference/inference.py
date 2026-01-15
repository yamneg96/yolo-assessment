"""
ONNX Runtime inference script for YOLO models.

This script loads an ONNX YOLO model and runs inference using ONNX Runtime,
providing optimized performance for production deployments.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
import onnxruntime as ort
from tqdm import tqdm

# Add utils directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "utils"))
try:
    from visualization import draw_bounding_boxes
    from preprocessing import preprocess_image
    from postprocessing import non_max_suppression, process_yolo_output
except ImportError:
    # Fallback if utils not yet created
    def draw_bounding_boxes(image, boxes, scores, labels, class_names):
        """Fallback visualization function."""
        return image
    
    def preprocess_image(image, input_size=(640, 640)):
        """Fallback preprocessing function."""
        return cv2.resize(image, input_size), 1.0
    
    def non_max_suppression(boxes, scores, score_threshold=0.25, iou_threshold=0.45):
        """Fallback NMS function."""
        return boxes, scores, np.arange(len(boxes))
    
    def process_yolo_output(output, original_shape, input_shape=(640, 640)):
        """Fallback output processing."""
        return np.array([]), np.array([]), np.array([])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ONNXInference:
    """
    ONNX Runtime-based YOLO inference handler.
    
    This class encapsulates all ONNX Runtime inference operations including:
    - Model loading and session configuration
    - Image preprocessing
    - Optimized inference execution
    - Result processing and saving
    """
    
    def __init__(self, onnx_model_path: str, confidence_threshold: float = 0.25, 
                 iou_threshold: float = 0.45, providers: List[str] = None):
        """
        Initialize the ONNX inference handler.
        
        Args:
            onnx_model_path: Path to the ONNX model file
            confidence_threshold: Minimum confidence score for detections
            iou_threshold: IoU threshold for non-maximum suppression
            providers: ONNX Runtime execution providers (e.g., ['CPUExecutionProvider', 'CUDAExecutionProvider'])
        """
        self.onnx_model_path = Path(onnx_model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.providers = providers or ['CPUExecutionProvider']
        
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        self.class_names = []
        
        logger.info(f"Initializing ONNX Runtime inference with model: {onnx_model_path}")
        logger.info(f"Using execution providers: {self.providers}")
        
    def load_model(self) -> None:
        """Load the ONNX model and create inference session."""
        try:
            logger.info("Loading ONNX model...")
            
            # Validate model file exists
            if not self.onnx_model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.onnx_model_path}")
            
            # Create ONNX Runtime session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Enable performance profiling if needed
            # session_options.enable_profiling = True
            
            self.session = ort.InferenceSession(
                str(self.onnx_model_path),
                sess_options=session_options,
                providers=self.providers
            )
            
            # Get input and output information
            input_info = self.session.get_inputs()[0]
            output_info = self.session.get_outputs()[0]
            
            self.input_name = input_info.name
            self.output_name = output_info.name
            self.input_shape = input_info.shape
            self.output_shape = output_info.shape
            
            logger.info(f"Input: {self.input_name}, Shape: {self.input_shape}, Type: {input_info.type}")
            logger.info(f"Output: {self.output_name}, Shape: {self.output_shape}, Type: {output_info.type}")
            
            # Set default class names (COCO dataset - 80 classes)
            self.class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush'
            ]
            
            logger.info(f"Model loaded successfully. Default classes: {len(self.class_names)}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {str(e)}")
            raise
    
    def preprocess_images(self, image_paths: List[str]) -> Tuple[List[np.ndarray], List[float], List[Tuple[int, int]]]:
        """
        Preprocess a list of images for ONNX inference.
        
        Args:
            image_paths: List of paths to input images
            
        Returns:
            Tuple of (preprocessed_images, scale_factors, original_shapes)
        """
        logger.info(f"Preprocessing {len(image_paths)} images for ONNX inference...")
        
        preprocessed_images = []
        scale_factors = []
        original_shapes = []
        
        # Get expected input size from model
        if len(self.input_shape) == 4:
            input_height, input_width = self.input_shape[2], self.input_shape[3]
        else:
            input_height, input_width = 640, 640  # Default YOLO size
        
        for img_path in tqdm(image_paths, desc="Preprocessing images"):
            try:
                # Read image
                image = cv2.imread(img_path)
                if image is None:
                    logger.warning(f"Failed to read image: {img_path}")
                    continue
                
                # Store original shape
                original_shape = image.shape[:2]  # (height, width)
                original_shapes.append(original_shape)
                
                # Calculate scale factor
                scale_x = input_width / original_shape[1]
                scale_y = input_height / original_shape[0]
                scale_factor = min(scale_x, scale_y)
                scale_factors.append(scale_factor)
                
                # Resize image
                resized = cv2.resize(image, (input_width, input_height))
                
                # Convert to RGB and normalize to [0, 1]
                resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                normalized = resized_rgb.astype(np.float32) / 255.0
                
                # Convert to CHW format and add batch dimension
                input_tensor = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
                input_tensor = np.expand_dims(input_tensor, axis=0)   # Add batch dimension
                
                preprocessed_images.append(input_tensor)
                
            except Exception as e:
                logger.error(f"Error preprocessing image {img_path}: {str(e)}")
                continue
        
        logger.info(f"Successfully preprocessed {len(preprocessed_images)} images")
        return preprocessed_images, scale_factors, original_shapes
    
    def run_inference(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Run inference on preprocessed images using ONNX Runtime.
        
        Args:
            images: List of preprocessed images as numpy arrays
            
        Returns:
            List of raw inference outputs
        """
        logger.info(f"Running ONNX Runtime inference on {len(images)} images...")
        
        results = []
        
        for i, image in enumerate(tqdm(images, desc="Running ONNX inference")):
            try:
                # Run inference
                output = self.session.run(
                    [self.output_name],
                    {self.input_name: image}
                )
                
                results.append(output[0])  # Get first (and usually only) output
                
            except Exception as e:
                logger.error(f"Error during ONNX inference on image {i}: {str(e)}")
                # Add empty result for consistency
                results.append(np.array([]))
        
        logger.info("ONNX Runtime inference completed")
        return results
    
    def process_outputs(self, raw_outputs: List[np.ndarray], original_shapes: List[Tuple[int, int]], 
                        scale_factors: List[float]) -> List[Dict[str, Any]]:
        """
        Process raw ONNX outputs to extract detections.
        
        Args:
            raw_outputs: List of raw model outputs
            original_shapes: List of original image shapes
            scale_factors: List of scale factors for each image
            
        Returns:
            List of processed detection results
        """
        logger.info("Processing ONNX outputs...")
        
        processed_results = []
        
        for i, (output, original_shape, scale_factor) in enumerate(
            zip(raw_outputs, original_shapes, scale_factors)
        ):
            try:
                # Process YOLO output (this depends on the specific YOLO variant)
                boxes, scores, labels = process_yolo_output(
                    output, 
                    original_shape, 
                    input_shape=(self.input_shape[2], self.input_shape[3]) if len(self.input_shape) == 4 else (640, 640)
                )
                
                # Apply non-maximum suppression
                if len(boxes) > 0:
                    boxes, scores, labels = non_max_suppression(
                        boxes, scores, 
                        score_threshold=self.confidence_threshold,
                        iou_threshold=self.iou_threshold
                    )
                
                # Scale boxes back to original image coordinates
                if len(boxes) > 0:
                    boxes = boxes / scale_factor
                
                processed_results.append({
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels,
                    'image_index': i
                })
                
            except Exception as e:
                logger.error(f"Error processing output for image {i}: {str(e)}")
                # Add empty result for consistency
                processed_results.append({
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'labels': np.array([]),
                    'image_index': i
                })
        
        logger.info("Output processing completed")
        return processed_results
    
    def save_results(self, image_paths: List[str], results: List[Dict[str, Any]], 
                    output_dir: str) -> None:
        """
        Save inference results with annotated images.
        
        Args:
            image_paths: Original image paths
            results: Processed inference results
            output_dir: Directory to save annotated images
        """
        logger.info(f"Saving ONNX results to {output_dir}...")
        
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
                
                # Draw bounding boxes on image
                annotated_image = draw_bounding_boxes(
                    image, result['boxes'], result['scores'], result['labels'], self.class_names
                )
                
                # Save annotated image
                output_filename = f"onnx_annotated_{Path(img_path).stem}.jpg"
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
                            'class_name': self.class_names[int(label)] if int(label) < len(self.class_names) else 'unknown'
                        }
                        for box, score, label in zip(result['boxes'], result['scores'], result['labels'])
                    ],
                    'num_detections': len(result['boxes'])
                })
                
            except Exception as e:
                logger.error(f"Error saving results for image {img_path}: {str(e)}")
                continue
        
        # Save detection data to JSON
        json_output_path = output_path / "onnx_detections.json"
        with open(json_output_path, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        logger.info(f"ONNX results saved. Annotated images: {len(detection_data)}")
        logger.info(f"Detection data saved to: {json_output_path}")


def main():
    """Main function to run ONNX inference."""
    # Configuration
    ONNX_MODEL_PATH = "onnx_conversion/yolo11n.onnx"
    IMAGE_DIR = "images"
    OUTPUT_DIR = "outputs/onnx"
    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    
    # Check if CUDA is available for ONNX Runtime
    providers = ['CPUExecutionProvider']
    try:
        if ort.get_device() == 'GPU':
            providers.insert(0, 'CUDAExecutionProvider')
            logger.info("CUDA available for ONNX Runtime")
    except:
        logger.info("Using CPU for ONNX Runtime")
    
    logger.info("Starting ONNX Runtime YOLO inference pipeline")
    
    try:
        # Initialize inference handler
        inference = ONNXInference(
            onnx_model_path=ONNX_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            providers=providers
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
        preprocessed_images, scale_factors, original_shapes = inference.preprocess_images(image_paths)
        
        # Run inference
        raw_outputs = inference.run_inference(preprocessed_images)
        
        # Process outputs
        results = inference.process_outputs(raw_outputs, original_shapes, scale_factors)
        
        # Save results
        inference.save_results(image_paths, results, OUTPUT_DIR)
        
        logger.info("ONNX Runtime inference pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
