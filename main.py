"""
Main orchestration script for YOLO PyTorch to ONNX conversion and comparison.

This script orchestrates the complete workflow:
1. PyTorch inference on input images
2. Model conversion from PyTorch to ONNX
3. ONNX Runtime inference on the same images
4. Comparison of results using IoU metrics
5. Generation of comprehensive reports and visualizations

This is the main entry point for the YOLO assessment project.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import argparse

# Add project directories to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "pytorch_inference"))
sys.path.append(str(project_root / "onnx_conversion"))
sys.path.append(str(project_root / "onnx_inference"))
sys.path.append(str(project_root / "utils"))

# Import modules
from pytorch_inference.inference import PyTorchInference
from onnx_conversion.convert import YOLOToONNXConverter
from onnx_inference.inference import ONNXInference
from utils.iou_comparison import run_full_comparison
from utils.visualization import plot_detection_statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo_assessment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class YOLOAssessmentPipeline:
    """
    Main pipeline class that orchestrates the complete YOLO assessment workflow.
    
    This class manages:
    - Environment setup and validation
    - PyTorch inference execution
    - ONNX model conversion
    - ONNX Runtime inference
    - Result comparison and analysis
    - Report generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration dictionary with all parameters
        """
        self.config = config
        self.start_time = time.time()
        
        # Validate configuration
        self._validate_config()
        
        # Initialize components
        self.pytorch_inference = None
        self.onnx_converter = None
        self.onnx_inference = None
        
        logger.info("YOLO Assessment Pipeline initialized")
        logger.info(f"Configuration: {config}")
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        required_keys = [
            'model_path', 'image_dir', 'output_dir',
            'confidence_threshold', 'iou_threshold'
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Validate paths
        model_path = Path(self.config['model_path'])
        image_dir = Path(self.config['image_dir'])
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        logger.info("Configuration validation passed")
    
    def setup_pytorch_inference(self) -> None:
        """Initialize PyTorch inference component."""
        logger.info("Setting up PyTorch inference...")
        
        self.pytorch_inference = PyTorchInference(
            model_path=self.config['model_path'],
            confidence_threshold=self.config['confidence_threshold'],
            iou_threshold=self.config['iou_threshold']
        )
        
        self.pytorch_inference.load_model()
        logger.info("PyTorch inference setup completed")
    
    def run_pytorch_inference(self) -> List[str]:
        """
        Run PyTorch inference on all images.
        
        Returns:
            List of image paths that were processed
        """
        logger.info("Running PyTorch inference...")
        
        # Get image paths
        image_dir = Path(self.config['image_dir'])
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [
            str(p) for p in image_dir.iterdir() 
            if p.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            logger.warning("No images found for processing")
            return []
        
        logger.info(f"Found {len(image_paths)} images for PyTorch inference")
        
        # Preprocess images
        preprocessed_images, scale_factors = self.pytorch_inference.preprocess_images(image_paths)
        
        # Run inference
        results = self.pytorch_inference.run_inference(preprocessed_images)
        
        # Save results
        output_dir = Path(self.config['output_dir']) / "pytorch"
        self.pytorch_inference.save_results(image_paths, results, str(output_dir), scale_factors)
        
        logger.info(f"PyTorch inference completed for {len(image_paths)} images")
        return image_paths
    
    def convert_to_onnx(self) -> str:
        """
        Convert PyTorch model to ONNX format.
        
        Returns:
            Path to the converted ONNX model
        """
        logger.info("Converting PyTorch model to ONNX...")
        
        onnx_output_dir = Path(self.config['output_dir']) / "onnx_conversion"
        
        self.onnx_converter = YOLOToONNXConverter(
            pytorch_model_path=self.config['model_path'],
            output_dir=str(onnx_output_dir)
        )
        
        onnx_model_path = self.onnx_converter.convert(
            validate=self.config.get('validate_onnx', True),
            compare_outputs=self.config.get('compare_outputs', True)
        )
        
        logger.info(f"Model conversion completed: {onnx_model_path}")
        return onnx_model_path
    
    def setup_onnx_inference(self, onnx_model_path: str) -> None:
        """Initialize ONNX Runtime inference component."""
        logger.info("Setting up ONNX Runtime inference...")
        
        # Check for CUDA availability
        providers = ['CPUExecutionProvider']
        if self.config.get('use_cuda', False):
            try:
                import onnxruntime as ort
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers.insert(0, 'CUDAExecutionProvider')
                    logger.info("CUDA enabled for ONNX Runtime")
                else:
                    logger.warning("CUDA not available, using CPU")
            except Exception as e:
                logger.warning(f"Failed to check CUDA availability: {e}")
        
        self.onnx_inference = ONNXInference(
            onnx_model_path=onnx_model_path,
            confidence_threshold=self.config['confidence_threshold'],
            iou_threshold=self.config['iou_threshold'],
            providers=providers
        )
        
        self.onnx_inference.load_model()
        logger.info("ONNX Runtime inference setup completed")
    
    def run_onnx_inference(self, image_paths: List[str]) -> None:
        """
        Run ONNX Runtime inference on the same images.
        
        Args:
            image_paths: List of image paths to process
        """
        logger.info("Running ONNX Runtime inference...")
        
        # Preprocess images
        preprocessed_images, scale_factors, original_shapes = self.onnx_inference.preprocess_images(image_paths)
        
        # Run inference
        raw_outputs = self.onnx_inference.run_inference(preprocessed_images)
        
        # Process outputs
        results = self.onnx_inference.process_outputs(raw_outputs, original_shapes, scale_factors)
        
        # Save results
        output_dir = Path(self.config['output_dir']) / "onnx"
        self.onnx_inference.save_results(image_paths, results, str(output_dir))
        
        logger.info(f"ONNX Runtime inference completed for {len(image_paths)} images")
    
    def compare_results(self) -> None:
        """Compare PyTorch and ONNX results and generate comparison report."""
        logger.info("Comparing PyTorch and ONNX results...")
        
        # Paths to result files
        pytorch_json = Path(self.config['output_dir']) / "pytorch" / "detections.json"
        onnx_json = Path(self.config['output_dir']) / "onnx" / "onnx_detections.json"
        
        if not pytorch_json.exists():
            logger.error(f"PyTorch results not found: {pytorch_json}")
            return
        
        if not onnx_json.exists():
            logger.error(f"ONNX results not found: {onnx_json}")
            return
        
        # Run comparison
        comparison_output_dir = Path(self.config['output_dir']) / "comparison"
        comparison_stats = run_full_comparison(
            str(pytorch_json),
            str(onnx_json),
            str(comparison_output_dir),
            self.config.get('iou_threshold', 0.5)
        )
        
        # Log summary statistics
        summary = comparison_stats['summary']
        logger.info("=== Comparison Summary ===")
        logger.info(f"Total PyTorch detections: {summary['total_pytorch_detections']}")
        logger.info(f"Total ONNX detections: {summary['total_onnx_detections']}")
        logger.info(f"Total matches: {summary['total_matches']}")
        logger.info(f"Average IoU: {summary['average_iou']:.4f}")
        logger.info(f"Precision: {summary['precision']:.4f}")
        logger.info(f"Recall: {summary['recall']:.4f}")
        logger.info(f"F1 Score: {summary['f1_score']:.4f}")
    
    def generate_final_report(self) -> None:
        """Generate a final comprehensive report."""
        logger.info("Generating final report...")
        
        end_time = time.time()
        total_time = end_time - self.start_time
        
        report = f"""
YOLO Assessment Pipeline - Final Report
=====================================

Execution Summary:
- Total Execution Time: {total_time:.2f} seconds
- Model: {self.config['model_path']}
- Image Directory: {self.config['image_dir']}
- Output Directory: {self.config['output_dir']}

Configuration:
- Confidence Threshold: {self.config['confidence_threshold']}
- IoU Threshold: {self.config['iou_threshold']}
- ONNX Validation: {self.config.get('validate_onnx', True)}
- Output Comparison: {self.config.get('compare_outputs', True)}
- CUDA Usage: {self.config.get('use_cuda', False)}

Generated Outputs:
1. PyTorch Results: {Path(self.config['output_dir']) / 'pytorch'}
2. ONNX Model: {Path(self.config['output_dir']) / 'onnx_conversion'}
3. ONNX Results: {Path(self.config['output_dir']) / 'onnx'}
4. Comparison Analysis: {Path(self.config['output_dir']) / 'comparison'}

Files Created:
- PyTorch annotated images and detections.json
- ONNX model file (.onnx) and annotated images
- Comparison report with IoU analysis and visualizations
- Detailed log file: yolo_assessment.log

Pipeline completed successfully!
"""
        
        # Save report
        report_path = Path(self.config['output_dir']) / "final_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Final report saved to: {report_path}")
        print(report)
    
    def run(self) -> None:
        """Run the complete assessment pipeline."""
        logger.info("Starting YOLO Assessment Pipeline...")
        
        try:
            # Step 1: PyTorch inference
            self.setup_pytorch_inference()
            image_paths = self.run_pytorch_inference()
            
            if not image_paths:
                logger.error("No images to process. Pipeline terminated.")
                return
            
            # Step 2: ONNX conversion
            onnx_model_path = self.convert_to_onnx()
            
            # Step 3: ONNX inference
            self.setup_onnx_inference(onnx_model_path)
            self.run_onnx_inference(image_paths)
            
            # Step 4: Compare results
            self.compare_results()
            
            # Step 5: Generate final report
            self.generate_final_report()
            
            logger.info("✓ YOLO Assessment Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLO PyTorch to ONNX Conversion and Assessment Pipeline"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolo11n.pt",
        help="Path to YOLO PyTorch model file"
    )
    
    parser.add_argument(
        "--images", "-i",
        type=str,
        default="images",
        help="Directory containing input images"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.25,
        help="Confidence threshold for detections"
    )
    
    parser.add_argument(
        "--iou-threshold", "-t",
        type=float,
        default=0.45,
        help="IoU threshold for non-maximum suppression"
    )
    
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA for ONNX Runtime inference"
    )
    
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip ONNX model validation"
    )
    
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip output comparison during conversion"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Create configuration
    config = {
        'model_path': args.model,
        'image_dir': args.images,
        'output_dir': args.output,
        'confidence_threshold': args.confidence,
        'iou_threshold': args.iou_threshold,
        'use_cuda': args.cuda,
        'validate_onnx': not args.no_validation,
        'compare_outputs': not args.no_comparison
    }
    
    # Create output directory
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    pipeline = YOLOAssessmentPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
