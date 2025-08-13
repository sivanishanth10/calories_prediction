"""
Main entrypoint. Streamlit will detect `pages/` automatically.
This file provides a lightweight landing page.
"""
import streamlit as st

st.set_page_config(page_title="Calorie Burn Prediction", layout="wide")
st.title("🔥 Calorie Burn Prediction — Dashboard")
st.markdown(
    """
Welcome — use the left page navigation to:
- Explore the dataset (Page 1)
- Train and evaluate models (Page 2)
- Compare models and choose the best (Page 3)
- Make personalized predictions (Page 4)
"""
)
st.sidebar.info("This app trains RandomForest and GradientBoosting models and saves them under `models/`.")
