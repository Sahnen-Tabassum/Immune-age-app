import streamlit as st

st.set_page_config(page_title="Contact", layout="wide")

st.title("📬 Contact & Feedback")

st.markdown("""
Have questions, suggestions, or feedback?

📧 Email: Immuneteam@gmail.com

---

## 💬 Feedback Form
""")

name = st.text_input("Your Name")
email = st.text_input("Your Email")
message = st.text_area("Message")

if st.button("Submit"):
    st.success("Thank you for your feedback! We'll get back to you soon.")