"""Page 2 — Model Evaluation & Analysis"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from pathlib import Path
from utils.data_loader import load_raw, merge_datasets

st.set_page_config(page_title="Model Evaluation", layout="wide")
st.title("2 — Model Evaluation & Analysis")

# Check for trained models
models_dir = Path("models")
model_files = list(models_dir.glob("*.pkl")) if models_dir.exists() else []

if not model_files:
    st.warning("⚠️ No trained models found! Please run 'py train_models.py' in PowerShell first.")
    st.info("💡 The training script will create models in the 'models/' directory.")
    st.stop()

# Load data for evaluation
try:
    ex_df, cal_df = load_raw()
    merged_df = merge_datasets(ex_df, cal_df)
    st.success("✅ Data loaded successfully for evaluation!")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# Load model comparison results
comparison_file = models_dir / "model_comparison.csv"
if comparison_file.exists():
    comparison_df = pd.read_csv(comparison_file, index_col=0)
    st.success("✅ Model evaluation data loaded!")
else:
    st.error("❌ Model evaluation file not found. Please run the training script first.")
    st.stop()

st.markdown("---")

## 1. MODEL PERFORMANCE SUMMARY
st.subheader("🏆 Model Performance Summary")

# Create performance overview
col1, col2, col3, col4 = st.columns(4)

with col1:
    best_r2 = comparison_df['R²'].max()
    best_r2_model = comparison_df['R²'].idxmax()
    st.metric("🥇 Best R² Score", f"{best_r2:.3f}", best_r2_model)

with col2:
    best_mae = comparison_df['MAE'].min()
    best_mae_model = comparison_df['MAE'].idxmin()
    st.metric("📏 Best MAE", f"{best_mae:.1f}", best_mae_model)

with col3:
    best_rmse = comparison_df['RMSE'].min()
    best_rmse_model = comparison_df['RMSE'].idxmin()
    st.metric("⚡ Best RMSE", f"{best_rmse:.1f}", best_rmse_model)

with col4:
    avg_r2 = comparison_df['R²'].mean()
    st.metric("📊 Average R²", f"{avg_r2:.3f}")

st.markdown("---")

## 2. DETAILED MODEL EVALUATION
st.subheader("📊 Detailed Model Evaluation")

# Create comprehensive evaluation table
evaluation_df = comparison_df.copy()
evaluation_df['Performance_Rank'] = evaluation_df['R²'].rank(ascending=False)
evaluation_df['MAE_Rank'] = evaluation_df['MAE'].rank(ascending=True)
evaluation_df['RMSE_Rank'] = evaluation_df['RMSE'].rank(ascending=True)
evaluation_df['Overall_Score'] = (evaluation_df['R²'] + (1/evaluation_df['MAE']) + (1/evaluation_df['RMSE'])) / 3

# Sort by overall score
evaluation_df = evaluation_df.sort_values('Overall_Score', ascending=False)

st.dataframe(evaluation_df, use_container_width=True)

st.markdown("---")

## 3. MODEL COMPARISON VISUALIZATIONS
st.subheader("📈 Model Comparison Visualizations")

# Create comparison charts
fig_comparison = make_subplots(
    rows=2, cols=2,
    subplot_titles=('R² Score Comparison', 'MAE Comparison', 'RMSE Comparison', 'Overall Performance'),
    specs=[[{"type": "bar"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "bar"}]]
)

# R² Score comparison
fig_comparison.add_trace(
    go.Bar(
        x=evaluation_df.index,
        y=evaluation_df['R²'],
        name="R² Score",
        marker_color='lightblue',
        text=[f"{v:.3f}" for v in evaluation_df['R²']],
        textposition='auto'
    ),
    row=1, col=1
)

# MAE comparison
fig_comparison.add_trace(
    go.Bar(
        x=evaluation_df.index,
        y=evaluation_df['MAE'],
        name="MAE",
        marker_color='lightcoral',
        text=[f"{v:.1f}" for v in evaluation_df['MAE']],
        textposition='auto'
    ),
    row=1, col=2
)

# RMSE comparison
fig_comparison.add_trace(
    go.Bar(
        x=evaluation_df.index,
        y=evaluation_df['RMSE'],
        name="RMSE",
        marker_color='lightgreen',
        text=[f"{v:.1f}" for v in evaluation_df['RMSE']],
        textposition='auto'
    ),
    row=2, col=1
)

# Overall performance score
fig_comparison.add_trace(
    go.Bar(
        x=evaluation_df.index,
        y=evaluation_df['Overall_Score'],
        name="Overall Score",
        marker_color='gold',
        text=[f"{v:.3f}" for v in evaluation_df['Overall_Score']],
        textposition='auto'
    ),
    row=2, col=2
)

fig_comparison.update_layout(height=700, title="Comprehensive Model Comparison")
st.plotly_chart(fig_comparison, use_container_width=True)

st.markdown("---")

## 4. MODEL SELECTION ANALYSIS
st.subheader("🎯 Model Selection Analysis")

# Create model selection criteria
criteria = st.multiselect(
    "Select evaluation criteria:",
    ["R² Score", "MAE", "RMSE", "Overall Performance"],
    default=["R² Score", "Overall Performance"]
)

if criteria:
    # Create radar chart for selected criteria
    fig_radar = go.Figure()
    
    # Normalize values for radar chart
    for model in evaluation_df.index:
        values = []
        for criterion in criteria:
            if criterion == "R² Score":
                values.append(evaluation_df.loc[model, 'R²'])
            elif criterion == "MAE":
                # Invert MAE (lower is better)
                values.append(1 / (1 + evaluation_df.loc[model, 'MAE'] / 100))
            elif criterion == "RMSE":
                # Invert RMSE (lower is better)
                values.append(1 / (1 + evaluation_df.loc[model, 'RMSE'] / 100))
            elif criterion == "Overall Performance":
                values.append(evaluation_df.loc[model, 'Overall_Score'])
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=criteria,
            fill='toself',
            name=model,
            line_color=px.colors.qualitative.Set3[list(evaluation_df.index).index(model)]
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Radar Chart (Selected Criteria)",
        height=500
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

## 5. FEATURE IMPORTANCE ANALYSIS
st.subheader("🔍 Feature Importance Analysis")

# Load Random Forest model for feature importance
rf_model_path = models_dir / "random_forest.pkl"
if rf_model_path.exists():
    try:
        rf_model = joblib.load(rf_model_path)
        
        # Get feature importance
        feature_names = ["Gender", "Age", "Height_cm", "Weight_kg", "Duration_min", "Heart_Rate", "Body_Temp_C"]
        importance = rf_model.feature_importances_
        
        # Create feature importance visualization
        fig_importance = px.bar(
            x=feature_names,
            y=importance,
            title="Feature Importance Analysis (Random Forest)",
            labels={'x': 'Features', 'y': 'Importance Score'},
            color=importance,
            color_continuous_scale='viridis'
        )
        
        fig_importance.update_layout(
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Feature importance insights
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("💡 **Top 3 Most Important Features:**")
            for i, (_, row) in enumerate(feature_importance_df.head(3).iterrows()):
                st.write(f"{i+1}. **{row['Feature']}**: {row['Importance']:.3f}")
        
        with col2:
            st.info("💡 **Feature Categories:**")
            st.write("• **Exercise Factors**: Duration, Heart Rate, Body Temperature")
            st.write("• **Personal Factors**: Age, Height, Weight, Gender")
            
    except Exception as e:
        st.warning(f"⚠️ Could not load Random Forest model for feature importance: {e}")
else:
    st.info("ℹ️ Random Forest model not found. Feature importance analysis unavailable.")

st.markdown("---")

## 6. MODEL RECOMMENDATIONS
st.subheader("💡 Model Recommendations & Use Cases")

# Create recommendations based on different criteria
recommendations = {
    "Production Use": best_r2_model,
    "High Precision": best_mae_model,
    "Consistent Performance": best_rmse_model,
    "Overall Best": evaluation_df.index[0]
}

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 **Model Recommendations by Use Case**")
    for use_case, model in recommendations.items():
        if use_case == "Production Use":
            st.success(f"**{use_case}**: {model} (R²: {comparison_df.loc[model, 'R²']:.3f})")
        elif use_case == "High Precision":
            st.info(f"**{use_case}**: {model} (MAE: {comparison_df.loc[model, 'MAE']:.1f})")
        elif use_case == "Consistent Performance":
            st.warning(f"**{use_case}**: {model} (RMSE: {comparison_df.loc[model, 'RMSE']:.1f})")
        else:
            st.metric(f"**{use_case}**:", model)

with col2:
    st.markdown("### 📊 **Performance Insights**")
    
    # Performance ranges
    r2_range = f"{comparison_df['R²'].min():.3f} - {comparison_df['R²'].max():.3f}"
    mae_range = f"{comparison_df['MAE'].min():.1f} - {comparison_df['MAE'].max():.1f}"
    rmse_range = f"{comparison_df['RMSE'].min():.1f} - {comparison_df['RMSE'].max():.1f}"
    
    st.info(f"""
    **Performance Ranges:**
    
    • **R² Score**: {r2_range}
    • **MAE**: {mae_range} calories
    • **RMSE**: {rmse_range} calories
    
    **Model Count**: {len(comparison_df)} models evaluated
    """)

# Final recommendation
st.markdown("---")
st.success(f"""
## 🏆 **Final Recommendation**

**Primary Model**: **{best_r2_model}** - Best overall performance with R² score of {best_r2:.3f}

**Model Characteristics**:
• **Accuracy**: R² = {comparison_df.loc[best_r2_model, 'R²']:.3f}
• **Precision**: MAE = {comparison_df.loc[best_r2_model, 'MAE']:.1f} calories
• **Consistency**: RMSE = {comparison_df.loc[best_r2_model, 'RMSE']:.1f} calories

**Next Steps**: 
1. Use the **Model Comparison** page for detailed analysis
2. Use the **Prediction** page to make actual predictions
3. Consider retraining if performance drops below acceptable thresholds
""")

st.markdown("---")
st.info("💡 **Note**: Models were trained using 'py train_models.py' in PowerShell. To retrain or update models, run the training script again.")
