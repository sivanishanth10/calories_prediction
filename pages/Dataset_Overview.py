"""Page 1 — Dataset Overview"""

import streamlit as st
from utils.data_loader import load_raw, merge_datasets
from utils.visualization import corr_heatmap, scatter_matrix_plot
import pandas as pd

st.set_page_config(page_title="Dataset Overview", layout="wide")
st.title("1 — Dataset Overview")

try:
    ex_df, cal_df = load_raw()
    st.subheader("Raw data preview")
    c1, c2 = st.columns(2)
    with c1:
        st.write("exercise.csv")
        st.dataframe(ex_df.head(10))
    with c2:
        st.write("calories.csv")
        st.dataframe(cal_df.head(10))

    st.markdown("---")
    st.subheader("Merged & preprocessed dataset")
    merged = merge_datasets(ex_df, cal_df)
    st.dataframe(merged.head(15))

    st.markdown("### Summary statistics")
    st.dataframe(merged.describe().T)

    st.markdown("### Correlation heatmap")
    corr_heatmap(merged)

    st.markdown("### Scatter matrix (interactive)")
    scatter_matrix_plot(merged)

except Exception as exc:
    st.error(f"Error loading or visualizing data: {exc}")
    st.stop()
