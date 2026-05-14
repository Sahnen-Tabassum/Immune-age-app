import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Immune Age Dashboard", layout="wide")
st.title("🧬 Immune Age Prediction Dashboard")

# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    X_columns = joblib.load("X_columns.pkl")
    explainer = joblib.load("explainer.pkl")
    return model, X_columns, explainer

model, X_columns, explainer = load_assets()

#--- NEW: HISTORY LOGGING FUNCTION ---
def save_to_history(df_to_save, filename="patient_history.csv"):
    if not os.path.isfile(filename):
        df_to_save.to_csv(filename, index=False)
    else:
        df_to_save.to_csv(filename, mode='a', index=False, header=False)

# =========================
# UPLOAD FILE
# =========================
uploaded_file = st.file_uploader("Upload Patient CSV File", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin analysis.")
    st.stop()

# =========================
# DATA PROCESSING
# =========================
df_raw = pd.read_csv(uploaded_file)

# Focus on a single person
if len(df_raw) > 1:
    st.warning("Multiple rows detected. Analyzing the first patient entry only.")
    df_raw = df_raw.iloc[[0]]

# 1. Capture Name for UI (Define this before using it)
patient_name = df_raw["Name"].iloc[0] if "Name" in df_raw.columns else "The Patient"

# 2. Create numeric copy for the model
df_model_input = df_raw.copy()

if "gender" in df_model_input.columns:
    gender_map = {'male': 1, 'female': 0}
    gender_val = str(df_model_input["gender"].iloc[0]).lower()
    # Replace text with numeric code in the model copy only
    df_model_input["gender"] = gender_map.get(gender_val, 0)

if "age" not in df_raw.columns:
    st.error("❌ CSV must contain an 'age' column.")
    st.stop()

chrono_age = float(df_raw["age"].iloc[0])

# 3. Align features for the model (ensures only numeric X_columns are passed)
df_model = df_model_input.reindex(columns=X_columns)
df_model = df_model.apply(pd.to_numeric, errors="coerce")
df_model = df_model.fillna(0) 

# =========================
# PREDICTION
# =========================
immune_age = float(model.predict(df_model)[0])
aging_gap = immune_age - chrono_age

# =========================
# 3. RISK STRATIFICATION (METER)
# =========================
if aging_gap <= 0:
    risk_label = "Low Risk (Healthy Aging)"
    risk_color = "green"
elif 0 < aging_gap <= 3:
    risk_label = "Moderate Risk (Early Senescence)"
    risk_color = "orange"
else:
    risk_label = "High Risk (Accelerated Aging)"
    risk_color = "red"

# =========================
# 1. PRIMARY METRICS
# =========================
st.subheader(f"🧬 Results for {patient_name}")
m1, m2, m3 = st.columns(3)

m1.metric("Chronological Age", f"{chrono_age:.0f} yrs")
m2.metric("Predicted Immune Age", f"{immune_age:.1f} yrs")
m3.metric("Aging Gap", f"{aging_gap:+.1f} yrs", delta=f"{aging_gap:+.1f}", delta_color="inverse")

#Saving Patient results to history
if st.button("💾 Save Results to History"):
    history_entry = df_raw.copy()
    # Check to avoid duplicate column if Patient_Name was already inserted elsewhere
    if "Patient_Name" not in history_entry.columns:
        history_entry.insert(0, "Patient_Name", patient_name)
    
    history_entry["predicted_immune_age"] = round(immune_age, 2)
    history_entry["aging_gap"] = round(aging_gap, 2)
    history_entry["risk_category"] = risk_label
    history_entry["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    
    save_to_history(history_entry)
    st.success(f"Successfully saved {patient_name}'s record to the History section!")

# =========================
# 2. PATIENT REPORT SUMMARY
# =========================
st.divider()
st.subheader(f"📋 Patient Report Summary")

# Use df_raw for display to keep original 'Male/Female' formatting
gender_display = "Male"
if "gender" in df_raw.columns:
    gender_display = str(df_raw["gender"].iloc[0]).capitalize()

st.markdown(f"""
This analysis is for **{patient_name}**, a **{chrono_age:.0f}-year-old {gender_display}**. 
The model has processed the uploaded blood biomarkers to determine the biological state of the immune system.
""")

# Display key markers from df_raw
c1, c2, c3, c4 = st.columns(4)
with c1: st.write(f"**CRP:** {df_raw['crp'].iloc[0] if 'crp' in df_raw.columns else 'N/A'}")
with c2: st.write(f"**Glucose:** {df_raw['Glucose'].iloc[0] if 'Glucose' in df_raw.columns else 'N/A'}")
with c3: st.write(f"**BMI:** {df_raw['bmi'].iloc[0] if 'bmi' in df_raw.columns else 'N/A'}")
with c4: st.write(f"**IL-6:** {df_raw['il6'].iloc[0] if 'il6' in df_raw.columns else 'N/A'}")
st.divider()

fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = aging_gap,
    title = {'text': f"Risk Status: {risk_label}", 'font': {'size': 20}},
    gauge = {
        'axis': {'range': [-10, 10], 'tickwidth': 1},
        'bar': {'color': "black"},
        'steps': [
            {'range': [-10, 0], 'color': "rgba(0, 255, 0, 0.3)"},
            {'range': [0, 3], 'color': "rgba(255, 255, 0, 0.3)"},
            {'range': [3, 10], 'color': "rgba(255, 0, 0, 0.3)"}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': aging_gap
        }
    }
))
fig_gauge.update_layout(height=300, margin=dict(t=50, b=0))
st.plotly_chart(fig_gauge, use_container_width=True)

# =========================
# 4. VISUAL INSIGHTS (SHAP)
# =========================
st.subheader("📊 Biological Drivers (SHAP Analysis)")
col1, col2 = st.columns([2, 1])

with col1:
    shap_values = explainer(df_model)
    fig_shap, ax_shap = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("Biomarker Impact on Immune Age")
    st.pyplot(fig_shap)

with col2:
    st.markdown("### 🧠 Clinical Interpretation")
    vals = shap_values.values[0]
    top_indices = np.argsort(np.abs(vals))[::-1][:3]

    for i in top_indices:
        feature = X_columns[i]
        impact = vals[i]
        direction = "increased ↗️" if impact > 0 else "decreased ↘️"
        st.info(f"**{feature}**\n\nThis marker {direction} immune age by **{abs(impact):.2f} years**.")

# =========================
# 5. DOWNLOAD REPORT
# =========================
st.divider()
df_final = df_raw.copy()
df_final.insert(0, "Patient_Name", patient_name)
df_final["predicted_immune_age"] = immune_age
df_final["aging_gap"] = aging_gap
df_final["risk_category"] = risk_label

csv = df_final.to_csv(index=False).encode('utf-8')
st.download_button(f"Download Clinical Report for {patient_name}", csv, f"{patient_name}_report.csv", "text/csv")