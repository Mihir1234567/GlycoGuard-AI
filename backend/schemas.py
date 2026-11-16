from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PatientInput(BaseModel):
    Pregnancies: int = Field(default=2, ge=0, le=25, description="Number of times pregnant")
    Glucose: float = Field(default=120.0, ge=0.0, le=300.0, description="Plasma glucose concentration (mg/dL)")
    BloodPressure: float = Field(default=70.0, ge=0.0, le=200.0, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(default=20.0, ge=0.0, le=120.0, description="Triceps skin fold thickness (mm)")
    Insulin: float = Field(default=80.0, ge=0.0, le=900.0, description="2-Hour serum insulin (mu U/ml)")
    BMI: float = Field(default=28.5, ge=0.0, le=80.0, description="Body Mass Index (weight in kg/(height in m)^2)")
    DiabetesPedigreeFunction: float = Field(default=0.45, ge=0.0, le=3.0, description="Diabetes pedigree genetic score")
    Age: int = Field(default=35, ge=1, le=120, description="Age in years")

class FeatureComparison(BaseModel):
    feature: str
    patient_value: float
    healthy_baseline: float
    deviation_pct: float
    status: str   # 'elevated', 'optimal', 'below'
    risk_level: str # 'High Risk Factor', 'Normal / Low Risk'

class PredictionResponse(BaseModel):
    prediction: int
    outcome_label: str
    risk_probability: float
    risk_percentage: float
    risk_tier: str
    risk_color: str
    confidence_score: float
    local_explanations: List[FeatureComparison]
    clinical_recommendations: List[str]

class BatchPredictionResponse(BaseModel):
    total_processed: int
    diabetic_count: int
    non_diabetic_count: int
    average_risk_percentage: float
    results: List[Dict[str, Any]]

class WhatIfRequest(BaseModel):
    base_patient: PatientInput
    variable_to_sweep: str = Field(default="Glucose", description="Clinical feature to sweep across range")
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    steps: int = Field(default=30, ge=10, le=100)

class CurvePoint(BaseModel):
    value: float
    risk_percentage: float
    risk_tier: str

class WhatIfResponse(BaseModel):
    variable: str
    baseline_value: float
    baseline_risk_percentage: float
    curve: List[CurvePoint]
    inflection_point: Optional[Dict[str, Any]] = None
    summary: str

class ModelInfoResponse(BaseModel):
    model_name: str
    algorithm: str
    best_params: Dict[str, Any]
    cv_roc_auc: float
    test_metrics: Dict[str, Any]
    benchmarks: List[Dict[str, Any]]
    feature_importances: List[Dict[str, Any]]
    healthy_baseline: Dict[str, float]
    diabetic_baseline: Dict[str, float]
    feature_ranges: Dict[str, Any]
