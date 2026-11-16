import os
import joblib
import pytest
import numpy as np
import pandas as pd

from src.transformers import ZeroToNaNConverter, DiabetesFeatureEngineer
from backend.app import (
    health_check, get_model_info, predict_patient, what_if_simulation
)
from backend.schemas import PatientInput, WhatIfRequest

def test_zero_to_nan_converter():
    df = pd.DataFrame({
        'Glucose': [0, 120, 140],
        'BloodPressure': [70, 0, 80],
        'Pregnancies': [0, 2, 4]  # Pregnancies should NOT be converted to NaN
    })
    converter = ZeroToNaNConverter(columns=['Glucose', 'BloodPressure'])
    res = converter.transform(df)
    assert np.isnan(res['Glucose'].iloc[0])
    assert res['Glucose'].iloc[1] == 120
    assert np.isnan(res['BloodPressure'].iloc[1])
    assert res['Pregnancies'].iloc[0] == 0  # 0 pregnancies is valid

def test_feature_engineer():
    df = pd.DataFrame({
        'BMI': [17.5, 22.0, 28.0, 34.0],
        'Age': [25, 35, 55, 60],
        'Glucose': [100, 120, 140, 160],
        'Insulin': [50, 0, 100, 150]
    })
    fe = DiabetesFeatureEngineer()
    res = fe.transform(df)
    assert list(res['BMI_Category']) == ['Underweight', 'Normal', 'Overweight', 'Obese']
    assert list(res['Age_Group']) == ['Young', 'Middle_Age', 'Senior', 'Senior']
    assert 'Glucose_Insulin_Ratio' in res.columns
    assert len(res) == 4

def test_model_pipeline_inference():
    pipeline_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'glycoguard_pipeline.joblib')
    assert os.path.exists(pipeline_path), "Serialized pipeline does not exist!"
    pipeline = joblib.load(pipeline_path)

    sample = pd.DataFrame([{
        'Pregnancies': 2,
        'Glucose': 120.0,
        'BloodPressure': 70.0,
        'SkinThickness': 20.0,
        'Insulin': 80.0,
        'BMI': 28.5,
        'DiabetesPedigreeFunction': 0.45,
        'Age': 35
    }])
    proba = pipeline.predict_proba(sample)
    pred = pipeline.predict(sample)

    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0
    assert pred[0] in [0, 1]

def test_api_health_endpoint():
    res = health_check()
    assert res["status"] == "healthy"
    assert res["model_loaded"] is True
    assert res["version"] == "1.0.0"

def test_api_model_info_endpoint():
    data = get_model_info()
    assert "RandomForest" in data.algorithm
    assert data.cv_roc_auc > 0.80
    assert len(data.benchmarks) >= 5
    assert len(data.feature_importances) > 0

def test_api_predict_endpoint_high_risk():
    patient = PatientInput(
        Pregnancies=6,
        Glucose=180.0,
        BloodPressure=85.0,
        SkinThickness=35.0,
        Insulin=220.0,
        BMI=38.0,
        DiabetesPedigreeFunction=0.85,
        Age=55
    )
    res = predict_patient(patient)
    assert res.prediction == 1
    assert res.risk_probability >= 0.50
    assert res.risk_tier in ["High Risk", "Critical Risk"]
    assert len(res.local_explanations) > 0
    assert len(res.clinical_recommendations) > 0

def test_api_predict_endpoint_healthy():
    patient = PatientInput(
        Pregnancies=1,
        Glucose=85.0,
        BloodPressure=68.0,
        SkinThickness=20.0,
        Insulin=65.0,
        BMI=22.4,
        DiabetesPedigreeFunction=0.25,
        Age=26
    )
    res = predict_patient(patient)
    assert res.prediction == 0
    assert res.risk_probability < 0.40
    assert res.risk_tier in ["Low Risk", "Moderate Risk"]

def test_api_what_if_endpoint():
    req = WhatIfRequest(
        base_patient=PatientInput(
            Pregnancies=2,
            Glucose=110.0,
            BloodPressure=70.0,
            SkinThickness=20.0,
            Insulin=80.0,
            BMI=28.0,
            DiabetesPedigreeFunction=0.40,
            Age=32
        ),
        variable_to_sweep="Glucose",
        steps=20
    )
    res = what_if_simulation(req)
    assert res.variable == "Glucose"
    assert len(res.curve) == 20
    assert res.summary is not None
