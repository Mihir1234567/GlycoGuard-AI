import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CLINICAL_ZERO_COLUMNS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

class ZeroToNaNConverter(BaseEstimator, TransformerMixin):
    """
    Converts physiologically invalid zero measurements into NaN values
    so that downstream imputation algorithms (e.g. KNNImputer) can reconstruct them.
    """
    def __init__(self, columns=None):
        self.columns = columns if columns is not None else CLINICAL_ZERO_COLUMNS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        if not isinstance(X_out, pd.DataFrame):
            X_out = pd.DataFrame(X_out)
        for col in self.columns:
            if col in X_out.columns:
                X_out[col] = X_out[col].replace(0, np.nan)
        return X_out


class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Engineers domain-specific clinical features:
    1. WHO BMI Risk Categories (Underweight, Normal, Overweight, Obese)
    2. Age Demographic Groups (Young, Middle_Age, Senior)
    3. Glucose-to-Insulin Metabolic Interaction Ratio
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        if not isinstance(X_out, pd.DataFrame):
            X_out = pd.DataFrame(X_out)

        # 1. WHO BMI Risk Categories
        bmi = X_out['BMI'].fillna(25.0)  # safe fallback for evaluation
        bmi_conditions = [
            (bmi < 18.5),
            (bmi >= 18.5) & (bmi < 25.0),
            (bmi >= 25.0) & (bmi < 30.0),
            (bmi >= 30.0)
        ]
        bmi_categories = ['Underweight', 'Normal', 'Overweight', 'Obese']
        X_out['BMI_Category'] = np.select(bmi_conditions, bmi_categories, default='Normal')

        # 2. Age Demographic Groups
        age = X_out['Age'].fillna(30.0)
        age_conditions = [
            (age < 30),
            (age >= 30) & (age < 50),
            (age >= 50)
        ]
        age_categories = ['Young', 'Middle_Age', 'Senior']
        X_out['Age_Group'] = np.select(age_conditions, age_categories, default='Young')

        # 3. Glucose / Insulin Interaction Ratio
        insulin_safe = X_out['Insulin'].fillna(0)
        glucose_safe = X_out['Glucose'].fillna(100.0)
        X_out['Glucose_Insulin_Ratio'] = glucose_safe / (insulin_safe + 1e-5)

        return X_out
