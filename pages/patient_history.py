import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Patient History", layout="wide")
st.title("📜 Patient History Database")

# Verify the file exists and is not empty before reading
if os.path.exists("patient_history.csv") and os.path.getsize("patient_history.csv") > 0:
    # Read the data
    history = pd.read_csv("patient_history.csv")
    
    # Optional: If you added the timestamp in the other script, sort it here
    if "timestamp" in history.columns:
        history = history.sort_values(by="timestamp", ascending=False)

    # Show the total count of patients
    st.write(f"Total Records: **{len(history)}**")

    # Display the table
    st.dataframe(history, use_container_width=True)
    
    # Add a refresh button in case you just saved a new patient
    if st.button("🔄 Refresh Data"):
        st.rerun()
else:
    st.info("The history file is currently empty. Results will appear here once you save them from the Dashboard.")