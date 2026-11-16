import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Custom transformers needed for unpickling
from src.transformers import ZeroToNaNConverter, DiabetesFeatureEngineer

st.set_page_config(
    page_title="GlycoGuard AI – Clinical Diabetes Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PIPELINE_PATH = os.path.join(MODELS_DIR, "glycoguard_pipeline.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
FEATURE_IMP_PATH = os.path.join(MODELS_DIR, "feature_importance.json")

@st.cache_resource
def load_model():
    if not os.path.exists(PIPELINE_PATH):
        st.error("Model pipeline not found! Please run 'python -m src.train' first.")
        st.stop()
    return joblib.load(PIPELINE_PATH)

@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(FEATURE_IMP_PATH, "r", encoding="utf-8") as f:
        feat_imp = json.load(f)
    return meta, feat_imp

pipeline = load_model()
metadata, feat_importances = load_metadata()

# Header
st.title("🩺 GlycoGuard AI – Diabetes Risk Analysis Platform")
st.markdown(
    "*Clinical Machine Learning Pipeline with Explainable AI (XAI) & Sensitivity Simulation*"
)

# Sidebar
st.sidebar.header("Patient Clinical Inputs")

preset = st.sidebar.selectbox(
    "Load Clinical Case Preset:",
    ["Custom", "Healthy Baseline", "Borderline Prediabetic", "High-Risk Obese Diabetic", "High Gravidity Profile"]
)

preset_values = {
    "Healthy Baseline": {"preg": 1, "glu": 85, "bp": 68, "skin": 20, "ins": 65, "bmi": 22.4, "dpf": 0.25, "age": 26},
    "Borderline Prediabetic": {"preg": 3, "glu": 128, "bp": 78, "skin": 28, "ins": 110, "bmi": 28.7, "dpf": 0.48, "age": 44},
    "High-Risk Obese Diabetic": {"preg": 6, "glu": 175, "bp": 88, "skin": 38, "ins": 210, "bmi": 36.8, "dpf": 0.85, "age": 52},
    "High Gravidity Profile": {"preg": 8, "glu": 142, "bp": 82, "skin": 32, "ins": 140, "bmi": 31.5, "dpf": 0.62, "age": 38},
}

vals = preset_values.get(preset, {"preg": 2, "glu": 120, "bp": 70, "skin": 20, "ins": 80, "bmi": 28.5, "dpf": 0.45, "age": 35})

pregnancies = st.sidebar.slider("Pregnancies", 0, 17, vals["preg"])
glucose = st.sidebar.slider("Glucose (mg/dL)", 50, 250, vals["glu"])
bp = st.sidebar.slider("Diastolic Blood Pressure (mm Hg)", 40, 140, vals["bp"])
skin = st.sidebar.slider("Triceps Skin Fold (mm)", 0, 80, vals["skin"])
insulin = st.sidebar.slider("2-Hour Serum Insulin (mu U/ml)", 0, 600, vals["ins"])
bmi = st.sidebar.slider("Body Mass Index (BMI)", 15.0, 65.0, float(vals["bmi"]), step=0.1)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.05, 2.50, float(vals["dpf"]), step=0.01)
age = st.sidebar.slider("Age (years)", 20, 90, vals["age"])

patient_df = pd.DataFrame([{
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': bp,
    'SkinThickness': skin,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': dpf,
    'Age': age
}])

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Patient Risk Assessment",
    "📈 What-If Sensitivity Simulator",
    "📁 Batch Cohort Screening",
    "📊 Model Analytics & Benchmarks"
])

# TAB 1
with tab1:
    prob = float(pipeline.predict_proba(patient_df)[0, 1])
    pred = int(pipeline.predict(patient_df)[0])
    risk_pct = prob * 100

    if prob < 0.30:
        tier, color = "Low Risk", "green"
    elif prob < 0.50:
        tier, color = "Moderate Risk", "orange"
    elif prob < 0.75:
        tier, color = "High Risk", "red"
    else:
        tier, color = "Critical Risk", "darkred"

    col_res1, col_res2, col_res3 = st.columns([1.5, 1.5, 2])
    with col_res1:
        st.metric("Predicted Diabetes Risk", f"{risk_pct:.1f}%")
        st.caption(f"Risk Tier: :{color}[**{tier}**]")
    with col_res2:
        outcome = "Diabetic Risk Indicated" if pred == 1 else "Non-Diabetic / Low Risk"
        conf = prob if pred == 1 else (1.0 - prob)
        st.metric("Clinical Classification", outcome)
        st.caption(f"Model Confidence: {conf * 100:.1f}%")
    with col_res3:
        st.progress(prob)
        st.caption("Risk Probability Scale (0% to 100%)")

    st.divider()

    col_xai, col_rec = st.columns([1.2, 1])

    with col_xai:
        st.subheader("🔬 Local Explainability: Feature Deviation vs. Healthy Population")
        healthy = metadata.get("healthy_baseline", {})
        comp_data = []
        for feat in patient_df.columns:
            pval = patient_df[feat].iloc[0]
            bval = healthy.get(feat, 0)
            diff = pval - bval
            pct = (diff / bval * 100) if bval != 0 else 0
            comp_data.append({
                "Feature": feat,
                "Patient Value": pval,
                "Healthy Baseline": bval,
                "Deviation %": round(pct, 1),
                "Status": "Elevated" if pct > 15 else ("Below Avg" if pct < -15 else "Optimal")
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

    with col_rec:
        st.subheader("📋 Clinical Decision Support")
        if glucose >= 140:
            st.warning("⚠️ **Elevated Glucose**: Order fasting glucose or HbA1c lab panel.")
        elif glucose >= 100:
            st.info("ℹ️ **Borderline Glucose**: Recommend low-GI diet and monitoring.")
        if bmi >= 30:
            st.warning("⚠️ **Classified as Obese**: Recommend targeted weight loss to improve insulin sensitivity.")
        if bp >= 80:
            st.warning("⚠️ **Elevated Blood Pressure**: Monitor cardiovascular markers.")
        if prob >= 0.50:
            st.error("🚨 **High Risk Predicted**: Recommend physician or endocrinologist consultation.")
        else:
            st.success("✅ **Within Normal Limits**: Continue routine health screenings.")

# TAB 2: WHAT-IF SIMULATION
with tab2:
    st.subheader("📈 What-If Clinical Sensitivity Simulation")
    st.markdown("Isolate a single clinical parameter and observe how varying it impacts this patient's predicted risk score.")

    sweep_var = st.selectbox("Choose variable to sweep:", ["Glucose", "BMI", "Age", "BloodPressure", "Insulin", "DiabetesPedigreeFunction"])
    ranges = metadata.get("ranges", {}).get(sweep_var, {"min": 50, "max": 250})

    sweep_vals = np.linspace(ranges["min"], ranges["max"], 40)
    sim_rows = [patient_df.iloc[0].to_dict() for _ in range(len(sweep_vals))]
    for i, val in enumerate(sweep_vals):
        sim_rows[i][sweep_var] = val
    sim_df = pd.DataFrame(sim_rows)
    probs = pipeline.predict_proba(sim_df)[:, 1] * 100

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sweep_vals, probs, color="#0d9488", linewidth=2.5, marker="o", markersize=4, label="Simulated Risk Curve")
    ax.axhline(50, color="red", linestyle="--", alpha=0.7, label="High Risk Threshold (50%)")

    # Patient baseline intersection
    curr_val = patient_df[sweep_var].iloc[0]
    curr_risk = prob * 100
    ax.scatter([curr_val], [curr_risk], color="crimson", s=100, zorder=5, label=f"Current Patient ({curr_val:.1f}, {curr_risk:.1f}%)")

    ax.set_title(f"Sensitivity Analysis: Sweeping {sweep_var} for Current Patient", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"{sweep_var} Value")
    ax.set_ylabel("Predicted Risk Probability (%)")
    ax.set_ylim(-2, 102)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    st.pyplot(fig)

# TAB 3: BATCH
with tab3:
    st.subheader("📁 Batch Patient Screening")
    st.markdown("Upload a CSV with patient records to predict risk across an entire cohort.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        req_cols = list(patient_df.columns)
        if all(c in batch_df.columns for c in req_cols):
            b_probs = pipeline.predict_proba(batch_df[req_cols])[:, 1]
            b_preds = pipeline.predict(batch_df[req_cols])

            batch_df["Risk_Score_%"] = np.round(b_probs * 100, 1)
            batch_df["Classification"] = np.where(b_preds == 1, "Diabetic Risk", "Non-Diabetic")

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Patients", len(batch_df))
            c2.metric("Diabetic Cases Flagged", int(np.sum(b_preds == 1)))
            c3.metric("Cohort Average Risk", f"{np.mean(b_probs)*100:.1f}%")

            st.dataframe(batch_df, use_container_width=True)

            csv_out = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Scored Cohort CSV", csv_out, "scored_patients.csv", "text/csv")
        else:
            st.error(f"Missing required columns. Must contain: {req_cols}")

# TAB 4: MODEL INSIGHTS
with tab4:
    st.subheader("📊 Cross-Validation Benchmarks & Model Architecture")
    bench_df = pd.DataFrame(metadata.get("benchmarks", []))
    st.dataframe(bench_df, use_container_width=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("Top Feature Importances")
        feat_df = pd.DataFrame(feat_importances).head(10)
        fig_feat, ax_f = plt.subplots(figsize=(6, 4))
        ax_f.barh(feat_df["feature"], feat_df["importance"], color="#0d9488")
        ax_f.invert_yaxis()
        ax_f.set_xlabel("Gini Importance")
        st.pyplot(fig_feat)

    with col_m2:
        st.subheader("Test Set Performance")
        tm = metadata.get("test_metrics", {})
        st.write(f"- **Accuracy**: `{tm.get('accuracy', 0):.4f}`")
        st.write(f"- **Sensitivity (Recall)**: `{tm.get('recall', 0):.4f}`")
        st.write(f"- **Specificity**: `{tm.get('specificity', 0):.4f}`")
        st.write(f"- **F1-Score**: `{tm.get('f1_score', 0):.4f}`")
        st.write(f"- **ROC-AUC**: `{tm.get('roc_auc', 0):.4f}`")

st.divider()
st.caption("GlycoGuard AI © 2026 – Advanced Clinical Machine Learning System. For educational and research demonstration.")
