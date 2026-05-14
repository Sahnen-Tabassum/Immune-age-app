import streamlit as st

st.set_page_config(page_title="About Immune Age", layout="wide")

st.title("📘 About Immune Age Prediction")

st.markdown("""
## 🧬 What is Immune Age?

Immune age is a biological measure of how old your immune system behaves,
which may differ from your actual age.

---

## 🔬 Biomarkers Used in Prediction:

The AI model analyzes blood and health markers such as:

- CRP (Inflammation marker)
- Glucose (Metabolic health)
- Creatinine (Kidney function)
- Albumin (Nutrition status)
- White Blood Cells (Immune response)
- T-cell counts (Immune strength)
- Cholesterol levels
- BMI & lifestyle factors

---

## 🤖 How AI works:

1. User uploads biomarker data (CSV)
2. Data is cleaned and processed
3. XGBoost model predicts immune age
4. SHAP explains feature contributions
5. Results are visualized for interpretation

---

This is a research-driven AI health analytics tool.
""")