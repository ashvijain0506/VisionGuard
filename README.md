# VisionGuard

## Deep Learning Visual Defect Detection

VisionGuard is a computer vision project that uses deep learning and explainable AI to identify and classify visual defects in industrial components.

### Current Focus: Cable Inspection

The first model focuses on the **cable** category from the MVTec AD dataset. Rather than only predicting normal vs. defective, VisionGuard is being developed to recognize the specific defect class present in an image.

## Pipeline

```text
Cable Image
    ↓
Preprocessing & Augmentation
    ↓
CNN Feature Learning
    ↓
Defect Classification
    ↓
Prediction + Confidence
    ↓
Explainable AI (SHAP)
```

## Project Goals

- Explore and preprocess industrial inspection images
- Build a CNN baseline for cable defect classification
- Evaluate predictions on unseen test images
- Compare learned visual features with Random Forest and XGBoost classifiers
- Use SHAP to investigate model predictions
- Build a simple interactive demonstration

## Repository Structure

```text
VisionGuard/
├── data/
│   └── README.md
├── models/
│   └── README.md
├── notebooks/
│   └── README.md
├── src/
│   ├── dataset.py
│   ├── explore_cable.py
│   ├── explore_dataset.py
│   ├── transforms.py
│   └── README.md
├── requirements.txt
└── README.md
```

## Status

🚧 In development — dataset exploration and preprocessing are complete; model training is next.

## Dataset

VisionGuard uses the MVTec Anomaly Detection (MVTec AD) dataset. The dataset itself is not stored in this repository.

## Technologies

Python · PyTorch · scikit-learn · XGBoost · SHAP · Streamlit
