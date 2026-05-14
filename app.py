import streamlit as st

st.set_page_config(page_title="Immune Age AI", layout="wide")

st.markdown("""
<style>

.neon {
    font-size: 70px;
    text-align: center;
    font-weight: bold;
    color: #00f7ff;
    font-family: Arial;
    
    text-shadow:
         0 0 2px #00f7ff,
         0 0 6px #00f7ff;       
}
</style>

<div class="neon">IMMUNE AGE AI</div>
""", unsafe_allow_html=True)

#dark theme styling
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Layout
left, right = st.columns([1.3, 1])

with left:
 st.title("🧬 Immune Age Prediction AI")

 st.markdown("""
 ## Welcome 👋

 **Immune Age** is a biological age estimate that reflects how your immune system is functioning,
 rather than your actual chronological age.

 Your immune system can age faster or slower depending on:
 - Inflammation levels
 - Metabolic health
 - Organ function
 - Blood biomarkers

 ---

 ### 🔬 What our app does:
 - You upload patient biomarker data (CSV)
 - AI predicts immune age using XGBoost
 - SHAP explains why prediction was made
 - Visual insights are given for clinical interpretation

 ---

 ### 🚀 Get started!
 👉 Use the sidebar to navigate to **Predict Immune Age**""")

with right:
    st.image("assets/immsys.png", use_container_width=True)