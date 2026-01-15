# YOLO PyTorch to ONNX Assessment Project

A comprehensive Python project that demonstrates YOLO model conversion from PyTorch to ONNX format with detailed inference comparison and analysis.

## Project Structure

```
yolo-assessment/
├── pytorch_inference/
│   └── inference.py              # PyTorch-based YOLO inference
├── onnx_conversion/
│   └── convert.py                # PyTorch to ONNX conversion
├── onnx_inference/
│   └── inference.py              # ONNX Runtime inference
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py          # Image preprocessing utilities
│   ├── visualization.py          # Visualization and plotting utilities
│   ├── postprocessing.py         # Output processing and NMS
│   └── iou_comparison.py         # IoU computation and result comparison
├── outputs/
│   ├── pytorch/                  # PyTorch inference results
│   ├── onnx/                     # ONNX inference results
│   └── comparison/               # Comparison analysis results
├── images/                       # Input images for inference
├── requirements.txt              # Pinned dependencies
├── main.py                       # Main orchestration script
├── yolo11n.pt                    # Pre-trained YOLO model
└── README.md                     # This file
```

## Features

- **PyTorch Inference**: Load and run YOLO models using Ultralytics
- **ONNX Conversion**: Convert PyTorch models to ONNX with validation
- **ONNX Runtime Inference**: Optimized inference using ONNX Runtime
- **Result Comparison**: IoU-based comparison between PyTorch and ONNX results
- **Visualization**: Annotated images and comparison plots
- **Comprehensive Logging**: Detailed logging throughout the pipeline
- **Production Quality**: Clean, modular, and well-documented code

## Installation

1. Clone or download the project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

Run the complete pipeline with default settings:
```bash
python main.py
```

### Advanced Usage

Customize the pipeline with command-line arguments:
```bash
python main.py --model yolo11n.pt \
               --images images/ \
               --output outputs/ \
               --confidence 0.25 \
               --iou-threshold 0.45 \
               --cuda
```

### Individual Components

Run specific components separately:

1. **PyTorch Inference Only**:
   ```bash
   python pytorch_inference/inference.py
   ```

2. **ONNX Conversion Only**:
   ```bash
   python onnx_conversion/convert.py
   ```

3. **ONNX Inference Only**:
   ```bash
   python onnx_inference/inference.py
   ```

## Command Line Arguments

- `--model, -m`: Path to YOLO PyTorch model file (default: yolo11n.pt)
- `--images, -i`: Directory containing input images (default: images/)
- `--output, -o`: Output directory for results (default: outputs/)
- `--confidence, -c`: Confidence threshold for detections (default: 0.25)
- `--iou-threshold, -t`: IoU threshold for NMS (default: 0.45)
- `--cuda`: Use CUDA for ONNX Runtime inference
- `--no-validation`: Skip ONNX model validation
- `--no-comparison`: Skip output comparison during conversion

## Output Files

The pipeline generates several output files:

### PyTorch Results (`outputs/pytorch/`)
- `annotated_*.jpg`: Images with bounding boxes and labels
- `detections.json`: Detection results in JSON format

### ONNX Results (`outputs/onnx/`)
- `onnx_annotated_*.jpg`: ONNX inference results
- `onnx_detections.json`: ONNX detection results

### ONNX Model (`outputs/onnx_conversion/`)
- `yolo11n.onnx`: Converted ONNX model

### Comparison Analysis (`outputs/comparison/`)
- `comparison_results.json`: Detailed comparison metrics
- `comparison_report.txt`: Text summary of comparison
- `comparison_plots.png`: Visual comparison plots

### Pipeline Logs
- `yolo_assessment.log`: Detailed execution log
- `final_report.txt`: Summary of the entire pipeline

## Key Features Explained

### 1. Robust Preprocessing
- Aspect ratio preservation with letterboxing
- Consistent preprocessing for both PyTorch and ONNX
- Configurable input sizes and normalization

### 2. Comprehensive Validation
- ONNX model structure validation
- Output comparison between PyTorch and ONNX
- Numerical accuracy verification

### 3. Advanced Comparison Metrics
- IoU-based detection matching
- Precision, Recall, and F1 score calculation
- Confidence score comparison
- Per-image and aggregate statistics

### 4. Production-Ready Code
- Comprehensive error handling
- Detailed logging at each stage
- Modular and extensible architecture
- Type hints and documentation

### 5. Visualization Tools
- Side-by-side result comparison
- Statistical plots and charts
- Annotated image generation
- Summary visualizations

## Dependencies

All dependencies are pinned in `requirements.txt` for reproducibility:

- `torch==2.1.2` - PyTorch deep learning framework
- `torchvision==0.16.2` - Computer vision utilities
- `onnx==1.15.0` - ONNX model format support
- `onnxruntime==1.16.3` - ONNX Runtime inference engine
- `opencv-python==4.8.1.78` - Computer vision operations
- `matplotlib==3.8.2` - Plotting and visualization
- `ultralytics==8.0.206` - YOLO model implementation
- `numpy==1.24.4` - Numerical computations
- `Pillow==10.1.0` - Image processing
- `tqdm==4.66.1` - Progress bars

## Technical Details

### Model Conversion Process
1. **Model Loading**: Load PyTorch YOLO model using Ultralytics
2. **Input Preparation**: Create dummy input with proper shape
3. **ONNX Export**: Export with specified opset version and dynamic axes
4. **Validation**: Verify ONNX model structure and runtime compatibility
5. **Output Comparison**: Ensure numerical consistency

### Inference Pipeline
1. **Image Loading**: Read and validate input images
2. **Preprocessing**: Resize, normalize, and format images
3. **Model Inference**: Run detection with confidence thresholds
4. **Post-processing**: Apply NMS and format results
5. **Visualization**: Generate annotated images and plots

### Comparison Methodology
- **IoU Matching**: Greedy matching based on Intersection over Union
- **Metrics Calculation**: Precision, Recall, F1 score
- **Statistical Analysis**: Per-image and aggregate statistics
- **Visual Comparison**: Side-by-side annotated images

## Troubleshooting

### Common Issues

1. **CUDA Not Available**: Falls back to CPU automatically
2. **Memory Issues**: Reduce batch size or image resolution
3. **Model Loading Errors**: Verify model file path and format
4. **ONNX Validation Failures**: Check opset version compatibility

### Performance Tips

- Use CUDA for faster ONNX Runtime inference
- Optimize image sizes for your specific use case
- Adjust confidence and IoU thresholds for your dataset
- Monitor memory usage with large image batches

## Extension Points

The modular architecture allows easy extension:

- **New Model Formats**: Add conversion scripts for other formats
- **Custom Preprocessing**: Modify preprocessing for specific requirements
- **Additional Metrics**: Extend comparison with custom metrics
- **Different Visualizations**: Add new plotting functions
- **Batch Processing**: Optimize for large-scale processing

## License

This project is provided as-is for educational and assessment purposes.