import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np

st.set_page_config(page_title="Patient Analytics", layout="wide")
st.title("📊 Advanced Patient Analytics")

# Path to your database
file_path = "patient_history.csv"

if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    df_history = pd.read_csv(file_path)
    
    # 🚨 FIXES THE STRING ERROR: Force vital columns to be numeric types
    numeric_cols = ['age', 'predicted_immune_age', 'aging_gap', 'crp', 'Glucose', 'bmi', 'il6']
    for col in numeric_cols:
        if col in df_history.columns:
            df_history[col] = pd.to_numeric(df_history[col], errors='coerce')
    
    # 1. Sidebar Selection
    st.sidebar.header("Select Patient")
    patient_list = df_history['Patient_Name'].unique()
    selected_name = st.sidebar.selectbox("Choose a record:", patient_list)
    
    # Get the latest record for that patient
    patient_data = df_history[df_history['Patient_Name'] == selected_name].iloc[-1]
    
    # ==========================================
    # ROW 1: THE BIG PICTURE
    # ==========================================
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🧬 Immune Age Gauge")
        # Ensure we don't pass a nan value to the gauge
        immune_age_val = float(patient_data['predicted_immune_age']) if pd.notna(patient_data['predicted_immune_age']) else 0.0
        chrono_age_val = float(patient_data['age']) if pd.notna(patient_data['age']) else 0.0
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = immune_age_val,
            delta = {'reference': chrono_age_val, 'position': "top"},
            title = {'text': "Immune Age vs Chrono Age"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightcyan"},
                    {'range': [40, 70], 'color': "royalblue"},
                    {'range': [70, 100], 'color': "midnightblue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': chrono_age_val
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader("🕸️ Biomarker Balance (Radar Chart)")
        
        categories = ['CRP (Inflammation)', 'Glucose (Metabolic)', 'BMI', 'IL-6 (Cytokine)', 'Age Impact']
        
        # Pull values safely and default to 0 if missing/NaN
        crp_raw = float(patient_data.get('crp', 0)) if pd.notna(patient_data.get('crp', 0)) else 0.0
        glu_raw = float(patient_data.get('Glucose', 0)) if pd.notna(patient_data.get('Glucose', 0)) else 0.0
        bmi_raw = float(patient_data.get('bmi', 0)) if pd.notna(patient_data.get('bmi', 0)) else 0.0
        il6_raw = float(patient_data.get('il6', 0)) if pd.notna(patient_data.get('il6', 0)) else 0.0
        gap_raw = float(patient_data.get('aging_gap', 0)) if pd.notna(patient_data.get('aging_gap', 0)) else 0.0
        
        # Simple normalization logic for the visual
        crp_val = min(crp_raw / 10, 1.0)
        glu_val = min(glu_raw / 150, 1.0)
        bmi_val = min(bmi_raw / 40, 1.0)
        il6_val = min(il6_raw / 10, 1.0)
        gap_val = min(abs(gap_raw) / 10, 1.0) # Now safe from string errors!
        
        values = [crp_val, glu_val, bmi_val, il6_val, gap_val]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
              r=values,
              theta=categories,
              fill='toself',
              name=selected_name,
              line_color='teal'
        ))

        fig_radar.update_layout(
          polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
          showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ==========================================
    # ROW 2: HISTORICAL TREND
    # ==========================================
    st.divider()
    st.subheader(f"📈 Aging Trend for {selected_name}")
    
    patient_history = df_history[df_history['Patient_Name'] == selected_name]
    
    if len(patient_history) > 1:
        fig_trend = px.line(patient_history, x='timestamp', y='aging_gap', 
                            markers=True, title="Aging Gap Over Time")
        fig_trend.update_traces(line_color='red')
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Trend analysis will appear once this patient has multiple saved records.")

else:
    st.warning("Please save patient data in the Dashboard first to view analytics.")