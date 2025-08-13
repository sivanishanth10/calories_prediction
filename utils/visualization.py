"""Small helpers for plots used in Streamlit pages."""

from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st


def corr_heatmap(df: pd.DataFrame, figsize: Tuple[int, int] = (8, 6)) -> None:
    """Render correlation heatmap using seaborn + matplotlib."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)


def scatter_matrix_plot(df: pd.DataFrame) -> None:
    """Interactive scatter matrix using plotly for numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] <= 1:
        st.info("Not enough numeric columns to render scatter matrix.")
        return
    fig = px.scatter_matrix(numeric, dimensions=numeric.columns.tolist(), title="Scatter matrix (interactive)")
    st.plotly_chart(fig, use_container_width=True)


def actual_vs_pred_plot(y_true, y_pred) -> None:
    """Scatter plot actual vs predicted with identity line."""
    import numpy as np
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.6)
    lo = min(min(y_true), min(y_pred))
    hi = max(max(y_true), max(y_pred))
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)


def feature_importance_bar(fi_df):
    """Barplot for permutation importance; fi_df must have 'feature' and 'importance_mean'."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=fi_df, x="importance_mean", y="feature", orient="h", ax=ax)
    ax.set_title("Permutation Feature Importance")
    st.pyplot(fig)
