# Advanced Brain Tumor MRI Classification System

A comprehensive deep learning system for automated brain tumor detection and type classification using MRI scans.

## 🧠 Project Overview

This project implements a 4-class brain tumor classification system that demonstrates:

- **Medical AI Expertise**: Specialized MRI image preprocessing with CLAHE and edge enhancement
- **Technical Depth**: EfficientNet-B4 with medical evaluation metrics
- **Production Ready**: Complete web application with Streamlit
- **Clinical Relevance**: Real-world medical problem with clinical assessment

## 📊 Dataset

**Brain Tumor MRI Dataset**

- **Source**: Kaggle — Brain Tumor MRI Dataset (Masoud Nickparvar)
- **Total Images**: ~7,023 MRI images with expert annotations
- **Classes**: 4 tumor types
- **Quality**: High-quality, clinically validated annotations
- **Class Distribution**:
  - No Tumor: 1,595 images (22.7%)
  - Glioma: 1,621 images (23.1%)
  - Meningioma: 1,645 images (23.4%)
  - Pituitary: 1,757 images (25.0%)

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv mri_env

# Activate environment
# Windows:
mri_env\Scripts\activate
# Linux/Mac:
source mri_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download from Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Place it as:
```
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

### 3. Generate CSV

```bash
python generate_csv.py
```

## 🏗️ Project Structure

```
Brain-Tumor-MRI-Classification/
├── src/
│   ├── mri_processor.py         # MRI image preprocessing pipeline
│   ├── dataset.py               # Dataset handling & class imbalance
│   ├── model.py                 # EfficientNet-B4 model
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Medical evaluation metrics
│   ├── gradcam_utils.py         # Grad-CAM visualization
│   └── deployment.py            # ONNX export & deployment
├── models/                      # Saved model checkpoints
├── data/                        # Dataset splits
├── dataset/                     # Original dataset (MRI images)
├── generate_csv.py              # CSV generator from folder structure
├── requirements.txt
└── README.md
```

## 🔬 Technical Features

### MRI Image Preprocessing

- **CLAHE Enhancement**: Adaptive contrast enhancement (critical for MRI)
- **Edge Enhancement**: Laplacian-based tumor boundary sharpening
- **Skull Strip Crop**: Remove background regions via circular mask
- **Normalization**: Proper scaling for deep learning

### Class Imbalance Handling

- **Weighted Sampling**: Balanced training batches
- **Class Weights**: Computed automatically
- **Data Augmentation**: Albumentations pipeline
- **Stratified Splits**: Maintain class distribution

### Medical Evaluation Metrics

- **Quadratic Weighted Kappa (QWK)**: Primary metric (>0.85 target)
- **Tumor Detection Sensitivity**: Critical for detecting any tumor (>90% target)
- **Specificity**: Reduces false alarms
- **AUC Scores**: Per-class discriminative ability
- **Confusion Matrix**: Class-wise performance

### Model Interpretability

- **Grad-CAM Visualization**: Shows which brain regions influence the decision
- **Attention Maps**: Visualize focus areas on MRI scans
- **Clinical Interpretability**: Essential for medical AI trust

### Validation Strategy

- **Stratified K-Fold Cross-Validation**: Robust evaluation for imbalanced medical data
- **5-Fold Validation**: Ensures each fold maintains class distribution
- **External Validation**: Ready for testing on different datasets

### Deployment Optimization

- **ONNX Conversion**: Lightweight deployment format
- **TensorRT Optimization**: GPU inference acceleration
- **Model Size**: <100MB target
- **Inference Speed**: <200ms per image

## 📈 Expected Performance

- **Quadratic Weighted Kappa (QWK)**: >0.85 (primary metric)
- **Overall Accuracy**: >90%
- **Tumor Detection Sensitivity**: >90%
- **Processing Time**: <200ms per image
- **Model Size**: <100MB

## 🎯 What Makes This Stand Out

1. **Medical Domain Expertise**: Specialized MRI preprocessing
2. **Technical Sophistication**: EfficientNet-B4 + medical metrics
3. **Production Ready**: Complete web application
4. **Clinical Relevance**: Real-world brain tumor detection
5. **Class Imbalance Handling**: Proper medical data handling

## 📋 Implementation Status

### ✅ Phase 1: Foundation (Completed)

- [x] MRI image preprocessing pipeline
- [x] Dataset analysis and class imbalance handling
- [x] Data augmentation pipeline
- [x] PyTorch dataset implementation

### ✅ Phase 2: Model Development (Completed)

- [x] EfficientNet-B4 model implementation
- [x] Training pipeline with cross-validation
- [x] Medical evaluation metrics
- [x] Grad-CAM visualization

### ✅ Phase 3: Deployment (Completed)

- [x] ONNX export & inference script
- [x] Deployment benchmarking
- [x] Production-ready code

## 📚 Dependencies

See `requirements.txt` for complete list. Key dependencies:

- PyTorch & Torchvision
- EfficientNet-PyTorch
- OpenCV & Albumentations
- Streamlit & Plotly
- Scikit-learn & Pandas

## 📄 License

This project is for educational and research purposes.
