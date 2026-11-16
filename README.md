# 🩺 GlycoGuard AI – Clinical Diabetes Risk Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-teal.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67+-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Pytest-8%20Passed-brightgreen.svg)](https://docs.pytest.org/)

An end-to-end, production-grade Artificial Intelligence and Machine Learning (AI/ML) healthcare decision support system built from the [`Prachita_AIML_Practical.ipynb`](./Prachita_AIML_Practical.ipynb) notebook.

GlycoGuard AI bridges academic predictive modeling with industry software engineering, providing:
1. **Leakage-Proof Machine Learning Pipeline**: Custom Scikit-Learn transformers, missing physiological zero imputation via `KNNImputer`, and hyperparameter-tuned ensemble classification.
2. **High-Performance FastAPI Backend**: RESTful microservice with Pydantic request validation, batch cohort screening, and what-if sensitivity simulations.
3. **Interactive Medical Web Dashboard**: A responsive single-page clinical application (Tailwind CSS, SVG gauge, Chart.js) with real-time risk stratification.
4. **Interactive Streamlit Data App**: An alternative, publication-ready data science exploration interface.
5. **Explainable AI (XAI)**: Local patient risk factor attribution vs. healthy cohort baselines.

---

## 📑 Table of Contents
- [Project Architecture](#-project-architecture)
- [Machine Learning Methodology](#-machine-learning-methodology)
  - [Clinical Data Handling & Leakage Prevention](#clinical-data-handling--leakage-prevention)
  - [Multi-Model Benchmarking & Selection](#multi-model-benchmarking--selection)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
- [Directory Structure](#-directory-structure)
- [Quick Start Guide](#-quick-start-guide)
  - [1. Setup Virtual Environment](#1-setup-virtual-environment)
  - [2. Train the Model Pipeline](#2-train-the-model-pipeline)
  - [3. Launch the Web Application](#3-launch-the-web-application)
  - [4. Launch the Streamlit Dashboard](#4-launch-the-streamlit-dashboard)
  - [5. Run Automated Tests](#5-run-automated-tests)
- [API Documentation](#-api-documentation)
- [Docker Deployment](#-docker-deployment)
- [Ethical & Clinical Disclaimer](#-ethical--clinical-disclaimer)

---

## 🏛️ Project Architecture

```
                  ┌─────────────────────────────────────────┐
                  │           User Interfaces               │
                  │  ┌───────────────────┐  ┌────────────┐  │
                  │  │ Tailwind Web App  │  │ Streamlit  │  │
                  │  │ (SVG Gauge, JS)   │  │ Dashboard  │  │
                  │  └─────────┬─────────┘  └─────┬──────┘  │
                  └────────────┼──────────────────┼─────────┘
                               │ HTTP / REST      │ Native
                               ▼                  ▼
                  ┌─────────────────────────────────────────┐
                  │             FastAPI Backend             │
                  │   /api/predict      /api/what-if        │
                  │   /api/model-info   /api/predict/batch  │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │       Production ML Pipeline            │
                  │ ┌─────────────────────────────────────┐ │
                  │ │ ZeroToNaNConverter (Domain Zeros)   │ │
                  │ ├─────────────────────────────────────┤ │
                  │ │ DiabetesFeatureEngineer (WHO BMI)   │ │
                  │ ├─────────────────────────────────────┤ │
                  │ │ KNNImputer (k=5) + StandardScaler   │ │
                  │ ├─────────────────────────────────────┤ │
                  │ │ OneHotEncoder (BMI, Age Groups)     │ │
                  │ ├─────────────────────────────────────┤ │
                  │ │ Tuned RandomForest Ensemble         │ │
                  │ └─────────────────────────────────────┘ │
                  └─────────────────────────────────────────┘
```

---

## 🔬 Machine Learning Methodology

### Clinical Data Handling & Leakage Prevention
In standard datasets like the Pima Indians Diabetes dataset, clinical features such as `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` contain zero values that represent **missing clinical tests** (a human cannot survive with zero blood glucose or blood pressure).
- **`ZeroToNaNConverter`**: Converts physiologically invalid zeros into `NaN` while keeping valid zeros (e.g., zero `Pregnancies`).
- **Pipeline Encapsulation**: Imputation (`KNNImputer(n_neighbors=5)`) and normalization (`StandardScaler`) are fit **strictly** on training cross-validation folds, guaranteeing zero data leakage.
- **Engineered Clinical Proxies**:
  - WHO BMI Risk Categories (`Underweight`, `Normal`, `Overweight`, `Obese`).
  - Age demographic intervals (`Young`, `Middle_Age`, `Senior`).
  - Glucose-to-Insulin metabolic interaction ratio proxy.

### Multi-Model Benchmarking & Selection
Seven classification algorithms were benchmarked under **Stratified 5-Fold Cross-Validation**:

| Model | CV ROC-AUC | Sensitivity (Recall) | Specificity | F1-Score | CV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** | **0.8360** | **0.7222** | **0.7400** | **0.6555** | **0.7338** |
| Logistic Regression | 0.8494 | 0.7475 | 0.7580 | 0.7006 | 0.7671 |
| Support Vector Machine (RBF) | 0.8333 | 0.7337 | 0.7511 | 0.6628 | 0.7426 |
| Extra Trees | 0.8202 | 0.5514 | 0.8380 | 0.5997 | 0.7394 |
| Gradient Boosting | 0.8148 | 0.6121 | 0.8240 | 0.6377 | 0.7508 |
| K-Nearest Neighbors | 0.8033 | 0.5749 | 0.8300 | 0.6234 | 0.7410 |
| Decision Tree | 0.6519 | 0.5564 | 0.7020 | 0.5483 | 0.6515 |

*Note: In medical diagnosis, high **Sensitivity (Recall)** is prioritized to minimize False Negatives (missed diabetic patients who need early intervention).*

### Hyperparameter Optimization
Using `GridSearchCV` on the ensemble pipeline:
- `n_estimators`: `100`
- `max_depth`: `5`
- `min_samples_split`: `2`
- `min_samples_leaf`: `2`
- `class_weight`: `'balanced'` (counters class imbalance: 65% non-diabetic vs. 35% diabetic)

---

## 📁 Directory Structure

```
PrachitaDIDI/
├── data/
│   └── diabetes.csv                     # Raw clinical dataset (768 records)
├── src/
│   ├── __init__.py
│   ├── transformers.py                  # Scikit-Learn custom transformers
│   └── train.py                         # Training, CV, grid search, and export
├── backend/
│   ├── __init__.py
│   ├── app.py                           # FastAPI application & endpoints
│   └── schemas.py                       # Pydantic v2 schemas
├── frontend/
│   ├── index.html                       # Modern medical UI (Tailwind CSS)
│   ├── app.js                           # Client-side reactivity & Chart.js logic
│   └── style.css                        # Glassmorphism & SVG gauge styling
├── models/
│   ├── glycoguard_pipeline.joblib       # Serialized production pipeline
│   ├── model_metadata.json              # Version, metrics & baseline stats
│   └── feature_importance.json          # Gini feature importances
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                 # Automated pytest suite (8 tests)
├── streamlit_app.py                     # Interactive Streamlit data app
├── run.py                               # Universal project runner
├── requirements.txt                     # Dependencies
├── Dockerfile                           # Containerization manifest
├── Prachita_AIML_Practical.ipynb        # Original exploratory notebook
└── README.md                            # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Setup Virtual Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Train the Model Pipeline
To re-run cross-validation, grid search, and serialize the pipeline:
```bash
python run.py --train
```

### 3. Launch the Web Application
Starts the FastAPI server and hosts the interactive medical UI:
```bash
python run.py
```
* Access the Web Dashboard: **[http://localhost:8000](http://localhost:8000)**
* Access Interactive Swagger API Docs: **[http://localhost:8000/docs](http://localhost:8000/docs)**
* Access ReDoc: **[http://localhost:8000/redoc](http://localhost:8000/redoc)**

### 4. Launch the Streamlit Dashboard
```bash
python run.py --streamlit
```
* Access Streamlit: **[http://localhost:8501](http://localhost:8501)**

### 5. Run Automated Tests
```bash
python run.py --test
# or directly:
pytest -v tests/test_pipeline.py
```

---

## 🔌 API Documentation

### `POST /api/predict`
Calculates patient diabetes risk, risk tier, local feature deviation, and clinical recommendations.

**Request Body:**
```json
{
  "Pregnancies": 3,
  "Glucose": 140.0,
  "BloodPressure": 75.0,
  "SkinThickness": 25.0,
  "Insulin": 110.0,
  "BMI": 31.2,
  "DiabetesPedigreeFunction": 0.55,
  "Age": 42
}
```

**Response:**
```json
{
  "prediction": 1,
  "outcome_label": "Diabetic Risk Indicated",
  "risk_probability": 0.642,
  "risk_percentage": 64.2,
  "risk_tier": "High Risk",
  "risk_color": "#EF4444",
  "confidence_score": 64.2,
  "local_explanations": [
    {
      "feature": "Glucose",
      "patient_value": 140.0,
      "healthy_baseline": 109.85,
      "deviation_pct": 27.4,
      "status": "elevated",
      "risk_level": "High Risk Factor"
    }
  ],
  "clinical_recommendations": [
    "Elevated blood glucose detected: Schedule a confirmatory Fasting Plasma Glucose (FPG) or HbA1c laboratory diagnostic panel.",
    "Classified as Obese (BMI >= 30): A structured 5-7% intentional body weight reduction program is strongly indicated."
  ]
}
```

### `POST /api/what-if`
Sweeps a continuous clinical feature across its physiological range to simulate how varying it changes risk.

**Request Body:**
```json
{
  "base_patient": { "Glucose": 120, "BMI": 28.5, ... },
  "variable_to_sweep": "Glucose",
  "steps": 35
}
```

### `POST /api/predict/batch`
Upload a CSV file containing patient cohorts for instant bulk screening and tabular manifest generation.

---

## 🐳 Docker Deployment

Build and run GlycoGuard AI in an isolated container:
```bash
# Build Docker image
docker build -t glycoguard-ai .

# Run container
docker run -p 8000:8000 -p 8501:8501 glycoguard-ai
```

---

## ⚠️ Ethical & Clinical Disclaimer
GlycoGuard AI is an automated decision support prototype developed for academic evaluation and clinical research demonstration. It is not an FDA-cleared diagnostic device and should not replace formal diagnostic medical testing or consultations with licensed healthcare providers.
# GlycoGuard-AI
