"""
ONNX conversion script for YOLO models.

This script converts a PyTorch YOLO model to ONNX format, handling common pitfalls
and ensuring proper input/output configurations for optimal performance.
"""

import logging
import os
from pathlib import Path
import torch
import onnx
import onnxruntime as ort
from ultralytics import YOLO
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLOToONNXConverter:
    """
    Handles conversion of YOLO PyTorch models to ONNX format.
    
    This class manages the complete conversion process including:
    - Model loading and validation
    - Input shape configuration
    - ONNX export with proper parameters
    - ONNX model validation
    """
    
    def __init__(self, pytorch_model_path: str, output_dir: str = "onnx_conversion"):
        """
        Initialize the converter.
        
        Args:
            pytorch_model_path: Path to the PyTorch .pt model file
            output_dir: Directory to save ONNX model and logs
        """
        self.pytorch_model_path = Path(pytorch_model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Conversion parameters
        self.input_size = (640, 640)  # Default YOLO input size
        self.opset_version = 11  # Stable opset version for YOLO
        self.dynamic_axes = None  # Set to True for dynamic input shapes
        
        logger.info(f"Initialized converter for model: {pytorch_model_path}")
    
    def load_pytorch_model(self) -> YOLO:
        """
        Load the PyTorch YOLO model.
        
        Returns:
            Loaded YOLO model
        """
        logger.info("Loading PyTorch YOLO model...")
        
        if not self.pytorch_model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.pytorch_model_path}")
        
        try:
            model = YOLO(str(self.pytorch_model_path))
            logger.info("PyTorch model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {str(e)}")
            raise
    
    def prepare_dummy_input(self) -> torch.Tensor:
        """
        Prepare a dummy input tensor for export tracing.
        
        Returns:
            Dummy input tensor with proper shape and dtype
        """
        # YOLO models expect input shape: (batch_size, channels, height, width)
        # Using batch_size=1 for single image inference
        dummy_input = torch.randn(1, 3, *self.input_size)
        
        logger.info(f"Prepared dummy input with shape: {dummy_input.shape}")
        return dummy_input
    
    def export_to_onnx(self, model: YOLO, dummy_input: torch.Tensor) -> str:
        """
        Export the PyTorch model to ONNX format.
        
        Args:
            model: Loaded PyOLO model
            dummy_input: Dummy input tensor for tracing
            
        Returns:
            Path to the exported ONNX model
        """
        logger.info("Starting ONNX export...")
        
        # Output path for ONNX model
        onnx_model_path = self.output_dir / f"{self.pytorch_model_path.stem}.onnx"
        
        try:
            # Get the underlying PyTorch model from YOLO wrapper
            pytorch_model = model.model
            
            # Set model to evaluation mode
            pytorch_model.eval()
            
            # Configure dynamic axes if needed
            dynamic_axes = None
            if self.dynamic_axes:
                dynamic_axes = {
                    'input': {0: 'batch_size'},  # Dynamic batch size
                    'output': {0: 'batch_size'}  # Dynamic batch size
                }
                logger.info("Using dynamic axes for variable batch sizes")
            
            # Export to ONNX
            torch.onnx.export(
                pytorch_model,
                dummy_input,
                str(onnx_model_path),
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            logger.info(f"ONNX export completed: {onnx_model_path}")
            return str(onnx_model_path)
            
        except Exception as e:
            logger.error(f"ONNX export failed: {str(e)}")
            raise
    
    def validate_onnx_model(self, onnx_model_path: str) -> bool:
        """
        Validate the exported ONNX model.
        
        Args:
            onnx_model_path: Path to the ONNX model file
            
        Returns:
            True if validation passes, False otherwise
        """
        logger.info("Validating ONNX model...")
        
        try:
            # Load and check the ONNX model
            onnx_model = onnx.load(onnx_model_path)
            
            # Check the model
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model structure validation passed")
            
            # Test with ONNX Runtime
            ort_session = ort.InferenceSession(onnx_model_path)
            
            # Get input info
            input_info = ort_session.get_inputs()[0]
            output_info = ort_session.get_outputs()[0]
            
            logger.info(f"Input shape: {input_info.shape}")
            logger.info(f"Input type: {input_info.type}")
            logger.info(f"Output shape: {output_info.shape}")
            logger.info(f"Output type: {output_info.type}")
            
            # Run a quick inference test
            dummy_input = np.random.randn(*input_info.shape).astype(np.float32)
            ort_outputs = ort_session.run(None, {'input': dummy_input})
            
            logger.info(f"ONNX Runtime inference test passed. Output shape: {ort_outputs[0].shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"ONNX model validation failed: {str(e)}")
            return False
    
    def compare_pytorch_onnx_outputs(self, pytorch_model: YOLO, onnx_model_path: str, 
                                   test_input: torch.Tensor) -> bool:
        """
        Compare outputs between PyTorch and ONNX models for consistency.
        
        Args:
            pytorch_model: Original PyTorch model
            onnx_model_path: Exported ONNX model path
            test_input: Test input tensor
            
        Returns:
            True if outputs are similar, False otherwise
        """
        logger.info("Comparing PyTorch and ONNX outputs...")
        
        try:
            # PyTorch inference
            with torch.no_grad():
                pytorch_model.model.eval()
                pytorch_output = pytorch_model.model(test_input)
            
            # ONNX Runtime inference
            ort_session = ort.InferenceSession(onnx_model_path)
            onnx_output = ort_session.run(None, {'input': test_input.numpy()})
            
            # Convert to numpy for comparison
            pytorch_np = pytorch_output[0].numpy() if isinstance(pytorch_output, (list, tuple)) else pytorch_output.numpy()
            onnx_np = onnx_output[0]
            
            # Calculate difference
            max_diff = np.abs(pytorch_np - onnx_np).max()
            mean_diff = np.abs(pytorch_np - onnx_np).mean()
            
            logger.info(f"Max difference: {max_diff:.6f}")
            logger.info(f"Mean difference: {mean_diff:.6f}")
            
            # Check if differences are within acceptable range
            tolerance = 1e-5
            if max_diff < tolerance:
                logger.info("✓ PyTorch and ONNX outputs are consistent")
                return True
            else:
                logger.warning(f"⚠ Outputs differ significantly (max diff: {max_diff:.6f})")
                return False
                
        except Exception as e:
            logger.error(f"Output comparison failed: {str(e)}")
            return False
    
    def convert(self, validate: bool = True, compare_outputs: bool = True) -> str:
        """
        Perform the complete conversion process.
        
        Args:
            validate: Whether to validate the ONNX model
            compare_outputs: Whether to compare PyTorch and ONNX outputs
            
        Returns:
            Path to the converted ONNX model
        """
        logger.info("Starting YOLO to ONNX conversion pipeline")
        
        try:
            # Load PyTorch model
            model = self.load_pytorch_model()
            
            # Prepare dummy input
            dummy_input = self.prepare_dummy_input()
            
            # Export to ONNX
            onnx_model_path = self.export_to_onnx(model, dummy_input)
            
            # Validate ONNX model
            if validate:
                if not self.validate_onnx_model(onnx_model_path):
                    raise RuntimeError("ONNX model validation failed")
            
            # Compare outputs
            if compare_outputs:
                if not self.compare_pytorch_onnx_outputs(model, onnx_model_path, dummy_input):
                    logger.warning("Output comparison showed differences, but conversion completed")
            
            logger.info("✓ Conversion pipeline completed successfully")
            return onnx_model_path
            
        except Exception as e:
            logger.error(f"Conversion pipeline failed: {str(e)}")
            raise


def main():
    """Main function to run the conversion."""
    # Configuration
    PYTORCH_MODEL_PATH = "yolo11n.pt"
    OUTPUT_DIR = "onnx_conversion"
    
    logger.info("Starting YOLO to ONNX conversion")
    
    try:
        # Initialize converter
        converter = YOLOToONNXConverter(
            pytorch_model_path=PYTORCH_MODEL_PATH,
            output_dir=OUTPUT_DIR
        )
        
        # Perform conversion
        onnx_model_path = converter.convert(
            validate=True,
            compare_outputs=True
        )
        
        logger.info(f"Conversion completed. ONNX model saved to: {onnx_model_path}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
