# AI Human Detection

![CI](https://github.com/helggaa/ai-human-detection/actions/workflows/python-ci.yml/badge.svg?branch=main)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-37%20Passed-success)
![Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Ruff](https://img.shields.io/badge/linter-Ruff-red)

> An end-to-end deep learning pipeline for distinguishing authentic human images from AI-generated human images using EfficientNetV2B0 and TensorFlow.

---

# Overview

AI-generated portraits are becoming increasingly realistic, making it difficult to distinguish synthetic images from authentic photographs.

This project implements a complete computer vision pipeline for binary image classification using transfer learning with EfficientNetV2B0. The project covers every stage of the workflow, from dataset cleaning to model deployment, following software engineering best practices including modular architecture, automated testing, code formatting, linting, Docker support, and CI.

---

## Live Demo

🔗 https://ai-or-realhuman-detection.streamlit.app/

## Demo

![Demo](docs/images/demo.png)


# Features

- End-to-end image classification pipeline
- EfficientNetV2B0 transfer learning
- Fine-tuning
- TensorFlow Dataset pipeline
- Automatic dataset cleaning
- Dataset preprocessing
- Training callbacks
- Performance evaluation
- Confusion matrix visualization
- ROC curve visualization
- Classification report generation
- Command-line inference
- Interactive Streamlit application
- Docker support
- Unit testing with Pytest
- Ruff linting
- Black code formatting
- GitHub Actions CI

---

# Project Architecture

```
AI_Human_Detection/

│
├── .github/
│   └── workflows/
│
├── .runtime/
│   └── python-version
│
├── docs/
│   └── images/
│
├── models/
│   ├── best_model.keras
│   └── final_model.keras
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_Preprocessing.ipynb
│   ├── 04_Training.ipynb
│   └── 05_Evaluation.ipynb
│
├── reports/
│   ├── figures/
│   ├── logs/
│   └── metrics/
│
├── src/
│   ├── cleaning.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── trainer.py
│   ├── predictor.py
│   ├── evaluator.py
│   ├── visualization.py
│   ├── config.py
│   ├── logger.py
│   └── utils.py
│
├── tests/
│
├── app.py
├── predict.py
├── Dockerfile
├── requirements.txt
├── README.md
└── runtime.txt
```

---

# Dataset

The dataset consists of two image classes.

| Class | Images |
|--------|-------:|
| Authentic Human | **515** |
| AI Generated Human | **537** |
| Total | **1052** |

Before training, every image passes through an automated cleaning pipeline including:

- Image validation
- Corrupted image detection
- RGB conversion
- JPEG conversion
- Sequential renaming
- Metadata logging
- MD5 checksum generation

---

# Model

## Backbone

EfficientNetV2B0

## Training Strategy

- Transfer Learning
- Fine-Tuning

## Framework

TensorFlow 2.20

## Loss Function

SparseCategoricalCrossentropy

## Optimizer

Adam

---

# Pipeline

```
Raw Dataset
      │
      ▼
Dataset Cleaning
      │
      ▼
Preprocessing
      │
      ▼
Transfer Learning
      │
      ▼
Fine-Tuning
      │
      ▼
Evaluation
      │
      ▼
Inference
      │
      ▼
Deployment
```

---

# Model Performance

| Metric | Score |
|--------|-------:|
| Accuracy | **81.65%** |
| Precision | **82.09%** |
| Recall | **81.65%** |
| F1-Score | **81.60%** |
| ROC-AUC | **TODO** |

Per-class performance:

| Class | Precision | Recall | F1-Score | Support |
|--------|-------:|-------:|-------:|-------:|
| Authentic Human | 78.16% | 87.18% | 82.42% | 78 |
| AI Generated Human | 85.92% | 76.25% | 80.79% | 80 |

Generated reports include:

- Classification Report
- Confusion Matrix
- ROC Curve
- Training History

---

# Installation

Clone the repository.

```bash
git clone https://github.com/helggaa/ai-human-detection.git

cd ai-human-detection
```

Create a virtual environment.

Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

Linux/macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Command-Line Inference

Predict a single image.

```bash
python predict.py path/to/image.jpg
```

Example output

```
Prediction

Authentic Human

Confidence

98.76%
```

---

# Streamlit Demo

Run the interactive web application.

```bash
streamlit run app.py
```

---

# Docker

Build the image.

```bash
docker build -t ai-human-detection .
```

Run the container.

```bash
docker run -p 8501:8501 ai-human-detection
```

Open http://localhost:8501

---

# Testing

Run all unit tests.

```bash
pytest
```

Current status

```
37 passed
```

---

# Code Quality

Lint

```bash
ruff check .
```

Formatter

```bash
black .
```

---

# Future Work

- ONNX export
- TensorRT optimization
- Explainable AI (Grad-CAM)
- Model quantization
- REST API with FastAPI
- Multi-class classification
- Larger datasets

---

# License

This project is licensed under the MIT License.