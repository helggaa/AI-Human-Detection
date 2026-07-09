# AI Human Detection

![CI](https://github.com/<USERNAME>/<REPOSITORY>/actions/workflows/ci.yml/badge.svg)

![Python](https://img.shields.io/badge/Python-3.12-blue)

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange)

![License](https://img.shields.io/badge/license-MIT-green)

A deep learning pipeline for binary image classification that distinguishes authentic human images from AI-generated human images using EfficientNetV2B0.

---

## Project Overview

This project implements an end-to-end computer vision pipeline including

- Dataset cleaning
- Dataset preprocessing
- TensorFlow Dataset pipeline
- Transfer learning
- Fine-tuning
- Model evaluation
- Visualization
- Model export

The project is designed using a modular architecture where each stage is implemented in an independent Python module.

---

## Features

- Modular source code
- EfficientNetV2B0 transfer learning
- TensorFlow Dataset pipeline
- Stratified dataset split
- Automatic logging
- Training callbacks
- Evaluation metrics
- Confusion matrix
- ROC Curve
- Classification report
- Training history export

---

## Project Structure

```
project/

├── dataset/
├── dataset_clean/
├── models/
├── notebooks/
│   ├── 01_Dataset_Exploration.ipynb
│   ├── 02_Dataset_Cleaning.ipynb
│   ├── 03_Preprocessing.ipynb
│   ├── 04_Model_Training.ipynb
│   └── 05_Evaluation.ipynb
│
├── reports/
│   ├── figures/
│   ├── logs/
│   └── metrics/
│
└── src/
    ├── config.py
    ├── logger.py
    ├── utils.py
    ├── cleaning.py
    ├── preprocessing.py
    ├── model.py
    ├── trainer.py
    ├── evaluator.py
    └── visualization.py
```

---

## Dataset

The dataset contains two image categories:

- Authentic Human Images
- AI-Generated Human Images

All images are cleaned before training.

Cleaning pipeline

- Image validation
- RGB conversion
- JPEG conversion
- Sequential renaming
- Metadata generation
- MD5 checksum generation

---

## Model

Backbone

- EfficientNetV2B0

Training

- Transfer Learning
- Fine Tuning

Loss

- SparseCategoricalCrossentropy

Optimizer

- Adam

Metrics

- Accuracy

---

## Pipeline

Dataset

↓

Cleaning

↓

Preprocessing

↓

Training

↓

Evaluation

↓

Visualization

---

## Evaluation

Generated outputs

- Classification Report
- Confusion Matrix
- ROC Curve
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## Installation

```bash
git clone https://github.com/username/AI-Human-Detection.git

cd AI-Human-Detection

pip install -r requirements.txt
```

---

## Usage

Cleaning

```bash
python src/cleaning.py
```

Preprocessing

```bash
python src/preprocessing.py
```

Training

```bash
python src/trainer.py
```

Evaluation

```bash
python src/evaluator.py
```

Visualization

```bash
python src/visualization.py
```

---

## Requirements

See

```
requirements.txt
```

---

## Future Improvements

- ONNX export
- TensorBoard integration
- Grad-CAM visualization
- Multi-class classification
- Hyperparameter tuning
- Docker support

---

## License

MIT License