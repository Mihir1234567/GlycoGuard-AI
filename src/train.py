import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, make_scorer
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

from src.transformers import ZeroToNaNConverter, DiabetesFeatureEngineer, CLINICAL_ZERO_COLUMNS

def specificity_score(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

def build_preprocessing_pipeline():
    num_cols = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Glucose_Insulin_Ratio'
    ]
    cat_cols = ['BMI_Category', 'Age_Group']

    numerical_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    preprocessing_pipeline = Pipeline(steps=[
        ('nan_converter', ZeroToNaNConverter(columns=CLINICAL_ZERO_COLUMNS)),
        ('feature_engineering', DiabetesFeatureEngineer()),
        ('preprocessor', preprocessor)
    ])

    return preprocessing_pipeline, num_cols, cat_cols

def train_and_export():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'diabetes.csv')
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)

    print(f"[*] Loading clinical diabetes dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[+] Loaded {len(df)} records with {len(df.columns)} features.")

    target_col = 'Outcome'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Stratified Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Train set size: {X_train.shape}, Test set size: {X_test.shape}")

    preprocessing_pipeline, num_cols, cat_cols = build_preprocessing_pipeline()

    # 1. Benchmark 7 candidate algorithms
    candidate_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42, class_weight='balanced'),
        'Extra Trees': ExtraTreesClassifier(random_state=42, class_weight='balanced')
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'specificity': make_scorer(specificity_score)
    }

    print("\n[*] Evaluating Candidate Models with Stratified 5-Fold Cross-Validation:")
    benchmark_results = []
    for name, model in candidate_models.items():
        pipe = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('classifier', model)
        ])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        benchmark_results.append({
            'model': name,
            'cv_accuracy': float(np.round(scores['test_accuracy'].mean(), 4)),
            'cv_precision': float(np.round(scores['test_precision'].mean(), 4)),
            'cv_recall': float(np.round(scores['test_recall'].mean(), 4)),
            'cv_specificity': float(np.round(scores['test_specificity'].mean(), 4)),
            'cv_f1': float(np.round(scores['test_f1'].mean(), 4)),
            'cv_roc_auc': float(np.round(scores['test_roc_auc'].mean(), 4))
        })
        print(f"    - {name:<24}: ROC-AUC={benchmark_results[-1]['cv_roc_auc']:.4f}, Recall={benchmark_results[-1]['cv_recall']:.4f}, F1={benchmark_results[-1]['cv_f1']:.4f}")

    benchmark_df = pd.DataFrame(benchmark_results).sort_values(by='cv_roc_auc', ascending=False)

    # 2. Hyperparameter Tuning for the Ensemble Model (Random Forest)
    print("\n[*] Performing Systematic Hyperparameter Tuning (GridSearchCV) on Random Forest...")
    rf_base_pipe = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 150],
        'classifier__max_depth': [5, 8, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(
        rf_base_pipe,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=1
    )
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    print(f"[+] Best Hyperparameters: {grid_search.best_params_}")
    print(f"[+] Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    # 3. Final Evaluation on Held-out Test Set
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    test_metrics = {
        'accuracy': float(np.round(accuracy_score(y_test, y_pred), 4)),
        'precision': float(np.round(precision_score(y_test, y_pred), 4)),
        'recall': float(np.round(recall_score(y_test, y_pred), 4)),
        'specificity': float(np.round(tn / (tn + fp), 4)),
        'f1_score': float(np.round(f1_score(y_test, y_pred), 4)),
        'roc_auc': float(np.round(roc_auc_score(y_test, y_proba), 4)),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
    }
    print("\n[+] Test Set Evaluation Metrics:")
    for k, v in test_metrics.items():
        if k != 'confusion_matrix':
            print(f"    - {k:<15}: {v:.4f}")

    # 4. Extract Feature Importances
    preprocessor = best_pipeline.named_steps['preprocessing'].named_steps['preprocessor']
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_feature_names

    classifier = best_pipeline.named_steps['classifier']
    importances = classifier.feature_importances_
    feat_imp_list = [
        {'feature': feat, 'importance': float(np.round(imp, 5))}
        for feat, imp in sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)
    ]

    # 5. Baseline reference profiles for Explainable AI (healthy non-diabetic average vs diabetic average)
    healthy_baseline = df[df['Outcome'] == 0].drop(columns=['Outcome']).mean().to_dict()
    diabetic_baseline = df[df['Outcome'] == 1].drop(columns=['Outcome']).mean().to_dict()
    overall_ranges = {
        col: {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'median': float(df[col].median()),
            'std': float(df[col].std())
        }
        for col in X.columns
    }

    metadata = {
        'model_name': 'GlycoGuard AI Clinical Classifier',
        'algorithm': 'RandomForestClassifier (Ensemble)',
        'best_params': {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()},
        'cv_roc_auc': float(np.round(grid_search.best_score_, 4)),
        'test_metrics': test_metrics,
        'benchmarks': benchmark_results,
        'feature_names': all_feature_names,
        'healthy_baseline': {k: float(np.round(v, 2)) for k, v in healthy_baseline.items()},
        'diabetic_baseline': {k: float(np.round(v, 2)) for k, v in diabetic_baseline.items()},
        'ranges': overall_ranges,
        'dataset_records': len(df),
        'train_records': len(X_train),
        'test_records': len(X_test)
    }

    # 6. Save Model Pipeline and JSON Metadata
    pipeline_path = os.path.join(models_dir, 'glycoguard_pipeline.joblib')
    joblib.dump(best_pipeline, pipeline_path)
    print(f"\n[+] Successfully serialized pipeline to: {pipeline_path}")

    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved model metadata to: {metadata_path}")

    feat_path = os.path.join(models_dir, 'feature_importance.json')
    with open(feat_path, 'w', encoding='utf-8') as f:
        json.dump(feat_imp_list, f, indent=2)
    print(f"[+] Saved feature importances to: {feat_path}")

    print("\n[✓] Training & Pipeline Serialization Completed Successfully!")
    return best_pipeline, metadata

if __name__ == '__main__':
    train_and_export()
