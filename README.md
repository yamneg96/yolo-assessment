# 🚀 YOLO PyTorch → ONNX Inference Assessment

This project demonstrates a complete, reproducible workflow for:

* Running inference using a **YOLO PyTorch model**
* Converting the model to **ONNX**
* Validating the ONNX model
* Running inference again using **ONNX Runtime**
* Comparing outputs for consistency

The implementation emphasizes **clean environment setup**, **correctness**, **clear logging**, and **AI-assisted coding best practices**, following the provided assessment guidelines.

---

## 📌 Overview

**Key objectives covered in this project:**

* ✅ Fresh Python environment setup
* ✅ PyTorch inference using a YOLO `.pt` model
* ✅ ONNX model conversion and validation
* ✅ ONNX Runtime inference
* ✅ Human-readable outputs (console + annotated images)
* ✅ Optional output comparison for consistency

This repository is intentionally **Python-only** to keep the scope focused on model inference and validation.

---

## 🧠 Technologies Used

* 🐍 Python 3.x
* 🔥 PyTorch
* 📦 Ultralytics YOLO
* 🔁 ONNX
* ⚡ ONNX Runtime
* 🖼️ OpenCV / Matplotlib
* 🤖 AI-assisted coding via **Cursor Pro / Windsurf**

---

## 📂 Project Structure

```
yolo-assessment/
├── pytorch_inference/
│   └── run_pytorch.py        # PyTorch YOLO inference
├── onnx_conversion/
│   └── convert_to_onnx.py    # PyTorch → ONNX conversion
├── onnx_inference/
│   └── run_onnx.py           # ONNX Runtime inference
├── utils/
│   └── visualization.py     # Bounding box utilities
├── images/
│   ├── image1.png
│   ├── image2.png
│   └── image3.png
├── outputs/
│   ├── pytorch/
│   └── onnx/
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Setup

> A clean environment is created from scratch to ensure reproducibility.

```bash
python -m venv env
source env/bin/activate        # Linux / macOS
env\Scripts\activate           # Windows
pip install -r requirements.txt
```

---

## ▶️ PyTorch Inference

* Loads the YOLO `.pt` model
* Runs inference on the provided images
* Outputs:

  * Bounding box coordinates
  * Class labels
  * Confidence scores
  * Annotated images saved to disk

```bash
python pytorch_inference/run_pytorch.py
```

📌 **This step establishes the baseline output before ONNX conversion.**

---

## 🔄 Convert Model to ONNX

* Converts the PyTorch YOLO model to ONNX format
* Uses fixed input dimensions for stability
* Validates the exported ONNX graph

```bash
python onnx_conversion/convert_to_onnx.py
```

✔️ ONNX model validation is performed using `onnx.checker`.

---

## ⚡ ONNX Runtime Inference

* Loads the converted ONNX model
* Runs inference on the same input images
* Outputs are logged and saved for comparison

```bash
python onnx_inference/run_onnx.py
```

---

## 📊 Output Comparison (Optional)

* PyTorch and ONNX predictions are visually compared
* Bounding boxes are overlaid
* Minor numerical differences are expected due to floating-point precision

This step helps demonstrate **functional equivalence** between the two inference pipelines.

---

## 🤖 AI-Assisted Coding

Throughout the implementation, **Cursor Pro / Windsurf AI** was used to:

* Scaffold scripts quickly
* Validate ONNX export parameters
* Catch common YOLO/ONNX pitfalls
* Review code structure and robustness

AI tools were used intentionally as **engineering assistants**, not as black-box generators.

---

## 🎥 Video Walkthrough

A full **screen + audio recording** accompanies this project, covering:

* Environment setup
* PyTorch inference
* ONNX conversion
* ONNX Runtime inference
* Output validation and comparison
* Explanation of design decisions and AI tool usage

⏱️ Total runtime: under 60 minutes

---

## ✅ Key Takeaways

* Clean, reproducible environment setup is critical
* Always validate PyTorch inference **before** ONNX conversion
* ONNX Runtime provides portable, efficient inference
* AI-assisted tools improve productivity when used deliberately
* Clear logging and explanations matter as much as correct output

---

## 🙌 Final Notes

This project focuses on **correctness, clarity, and process**, mirroring real-world production workflows for ML inference pipelines.

Thank you for reviewing this submission.

— **Yamlak**

---