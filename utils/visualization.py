"""Small helpers for plots used in Streamlit pages."""

from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Try to import plotly with fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def corr_heatmap(df: pd.DataFrame, figsize: Tuple[int, int] = (8, 6)) -> None:
    """Render correlation heatmap using seaborn + matplotlib."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)


def scatter_matrix_plot(df: pd.DataFrame) -> None:
    """Enhanced interactive scatter matrix using plotly for better clarity."""
    if not PLOTLY_AVAILABLE:
        st.error("⚠️ Plotly not available. Please install with: pip install plotly")
        st.info("Showing basic correlation heatmap instead...")
        corr_heatmap(df)
        return
    
    # Select only numeric columns and handle the data properly
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    if len(numeric_cols) <= 1:
        st.info("Not enough numeric columns to render scatter matrix.")
        return
    
    # Limit to top 6 most important features for clarity
    if len(numeric_cols) > 6:
        # Keep the most important features (Calories, Duration, Heart_Rate, etc.)
        important_features = ['Calories', 'Duration_min', 'Heart_Rate', 'Age', 'Weight_kg', 'Height_cm']
        available_features = [col for col in important_features if col in numeric_cols]
        if len(available_features) < 6:
            # Add other features if needed
            other_features = [col for col in numeric_cols if col not in available_features]
            available_features.extend(other_features[:6-len(available_features)])
        numeric_cols = available_features[:6]
    
    # Create a cleaner dataset for visualization
    viz_df = df[numeric_cols].copy()
    
    # Remove outliers for better visualization (optional)
    for col in viz_df.columns:
        Q1 = viz_df[col].quantile(0.25)
        Q3 = viz_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        viz_df = viz_df[(viz_df[col] >= lower_bound) & (viz_df[col] <= upper_bound)]
    
    # Sample data if too many points for performance
    if len(viz_df) > 2000:
        viz_df = viz_df.sample(n=2000, random_state=42)
    
    # Create enhanced scatter matrix
    fig = px.scatter_matrix(
        viz_df,
        dimensions=numeric_cols,
        title="Interactive Scatter Matrix - Feature Relationships",
        color='Calories' if 'Calories' in numeric_cols else numeric_cols[0],
        color_continuous_scale='viridis',
        opacity=0.7,
        size='Calories' if 'Calories' in numeric_cols else numeric_cols[0],
        size_max=15
    )
    
    # Update layout for better clarity
    fig.update_layout(
        height=800,
        showlegend=True,
        title_x=0.5,
        title_font_size=16
    )
    
    # Update all traces for better visibility
    for trace in fig.data:
        trace.marker.opacity = 0.6
        trace.marker.line.width = 0.5
        trace.marker.line.color = 'white'
    
    # Update axes for better readability
    for axis in fig.layout:
        if 'xaxis' in axis:
            fig.layout[axis].title.font.size = 12
            fig.layout[axis].tickfont.size = 10
        if 'yaxis' in axis:
            fig.layout[axis].title.font.size = 12
            fig.layout[axis].tickfont.size = 10
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights about the relationships
    st.markdown("### 💡 **Scatter Matrix Insights**")
    
    # Calculate correlations for insights
    corr_matrix = viz_df.corr()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Strong Positive Correlations (>0.5):**")
        strong_pos = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val > 0.5:
                    strong_pos.append(f"• {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.2f}")
        
        if strong_pos:
            for insight in strong_pos[:3]:  # Show top 3
                st.write(insight)
        else:
            st.write("• No strong positive correlations found")
    
    with col2:
        st.info("**Strong Negative Correlations (<-0.3):**")
        strong_neg = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val < -0.3:
                    strong_neg.append(f"• {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.2f}")
        
        if strong_neg:
            for insight in strong_neg[:3]:  # Show top 3
                st.write(insight)
        else:
            st.write("• No strong negative correlations found")
    
    # Add usage instructions
    st.markdown("""
    **🎯 How to Use This Scatter Matrix:**
    
    • **Hover** over points to see exact values
    • **Click and drag** to zoom into specific regions
    • **Double-click** to reset the view
    • **Color coding** shows relationship with Calories (or primary feature)
    • **Size** indicates intensity of the primary feature
    • **Diagonal** shows distribution of each feature
    """)


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
