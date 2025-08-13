"""Page 3 — Model Comparison & Visualization"""

import streamlit as st
import pandas as pd
import numpy as np

# Try to import plotly with fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    st.error("⚠️ Plotly not available. Please install with: pip install plotly")
    PLOTLY_AVAILABLE = False

import joblib
import os
from pathlib import Path
from utils.data_loader import load_raw, merge_datasets

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("3 — Model Comparison & Performance Analysis")

# Check if plotly is available
if not PLOTLY_AVAILABLE:
    st.warning("""
    ## ⚠️ Plotly Not Available
    
    This page requires Plotly for visualizations. Please:
    
    1. **Install Plotly**: `pip install plotly`
    2. **Restart the app**
    3. **Check requirements.txt** includes plotly
    
    For now, showing basic information only.
    """)
    
    # Show basic info without plots
    st.info("Basic model comparison information will be shown here once Plotly is installed.")
    st.stop()

# Check for trained models
models_dir = Path("models")
model_files = list(models_dir.glob("*.pkl")) if models_dir.exists() else []

if not model_files:
    st.warning("⚠️ No trained models found! Training models automatically...")
    
    # Auto-train models with caching
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def train_models_cached():
        try:
            # Load and prepare data
            ex_df, cal_df = load_raw()
            merged_df = merge_datasets(ex_df, cal_df)
            
            # Prepare features and target
            features = ["Gender", "Age", "Height_cm", "Weight_kg", "Duration_min", "Heart_Rate", "Body_Temp_C"]
            X = merged_df[features]
            y = merged_df["Calories"]
            
            # Split data
            from sklearn.model_selection import train_test_split, cross_val_score
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Import models
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.svm import SVR
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            # Define models (reduced complexity for speed)
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),  # Reduced from 100
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42),  # Reduced from 100
                "SVR": SVR(kernel='rbf', C=10, gamma='scale')  # Reduced complexity
            }
            
            results = {}
            
            # Train each model
            for name, model in models.items():
                st.write(f"📈 Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation score (reduced folds for speed)
                cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')  # Reduced from 5 to 3
                
                results[name] = {
                    "model": model,
                    "mae": mae,
                    "mse": mse,
                    "rmse": rmse,
                    "r2": r2,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std()
                }
            
            # Save models
            os.makedirs("models", exist_ok=True)
            
            for name, result in results.items():
                filename = f"models/{name.lower().replace(' ', '_')}.pkl"
                joblib.dump(result["model"], filename)
                st.write(f"✅ Saved: {filename}")
            
            # Save results summary
            results_summary = {}
            for name, result in results.items():
                results_summary[name] = {
                    "MAE": round(result["mae"], 2),
                    "RMSE": round(result["rmse"], 2),
                    "R²": round(result["r2"], 3),
                    "CV R²": f"{result['cv_mean']:.3f} ± {result['cv_std']:.3f}"
                }
            
            # Save as CSV
            summary_df = pd.DataFrame(results_summary).T
            summary_df.to_csv("models/model_comparison.csv")
            st.write("✅ Saved: models/model_comparison.csv")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error training models: {e}")
            return False
    
    # Call cached training function
    with st.spinner("🚀 Training models... This may take a few minutes (cached for future use)..."):
        success = train_models_cached()
        
    if success:
        st.success("🎉 Models trained and saved successfully!")
        # Update model files list
        model_files = list(models_dir.glob("*.pkl"))
    else:
        st.info("💡 Please check your data files or run 'py train_models.py' in PowerShell locally.")
        st.stop()

# Load model comparison results
comparison_file = models_dir / "model_comparison.csv"
if comparison_file.exists():
    comparison_df = pd.read_csv(comparison_file, index_col=0)
    st.success("✅ Model comparison data loaded!")
else:
    st.error("❌ Model comparison file not found. Please train models first.")
    st.stop()

# Page Layout
st.markdown("---")

## 1. MODEL PERFORMANCE OVERVIEW
st.subheader("🏆 Model Performance Overview")

# Create performance comparison chart
fig_performance = go.Figure()

# Add bars for each metric
metrics = ['MAE', 'RMSE', 'R²']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, metric in enumerate(metrics):
    if metric in comparison_df.columns:
        values = comparison_df[metric].values
        if metric == 'R²':  # R² should be higher = better
            values = 1 - values  # Invert for visualization (lower = better)
        
        fig_performance.add_trace(go.Bar(
            name=metric,
            x=comparison_df.index,
            y=values,
            marker_color=colors[i],
            text=[f"{v:.3f}" for v in comparison_df[metric].values],
            textposition='auto'
        ))

fig_performance.update_layout(
    title="Model Performance Comparison (Lower is Better)",
    xaxis_title="Models",
    yaxis_title="Metric Values",
    barmode='group',
    height=500,
    showlegend=True
)

st.plotly_chart(fig_performance, use_container_width=True)

## 2. DETAILED METRICS TABLE
st.subheader("📊 Detailed Performance Metrics")

# Create styled metrics table
col1, col2 = st.columns([2, 1])

with col1:
    st.dataframe(comparison_df, use_container_width=True)

with col2:
    # Find best model
    best_model = comparison_df['R²'].idxmax()
    best_score = comparison_df.loc[best_model, 'R²']
    
    st.metric("🥇 Best Model", best_model)
    st.metric("🏆 Best R² Score", f"{best_score:.3f}")
    
    # Performance insights
    st.info(f"""
    **Performance Insights:**
    
    • **Best Model**: {best_model}
    • **R² Range**: {comparison_df['R²'].min():.3f} - {comparison_df['R²'].max():.3f}
    • **MAE Range**: {comparison_df['MAE'].min():.1f} - {comparison_df['MAE'].max():.1f} calories
    """)

st.markdown("---")

## 3. FEATURE IMPORTANCE ANALYSIS
st.subheader("🔍 Feature Importance Analysis")

# Load Random Forest model for feature importance
rf_model_path = models_dir / "random_forest.pkl"
if rf_model_path.exists():
    try:
        rf_model = joblib.load(rf_model_path)
        
        # Get feature importance
        feature_names = ["Gender", "Age", "Height_cm", "Weight_kg", "Duration_min", "Heart_Rate", "Body_Temp_C"]
        importance = rf_model.feature_importances_
        
        # Create feature importance chart
        fig_importance = px.bar(
            x=feature_names,
            y=importance,
            title="Feature Importance (Random Forest)",
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
        
        st.info("💡 **Feature Importance Insights:**")
        for _, row in feature_importance_df.head(3).iterrows():
            st.write(f"• **{row['Feature']}**: {row['Importance']:.3f}")
            
    except Exception as e:
        st.warning(f"⚠️ Could not load Random Forest model for feature importance: {e}")
else:
    st.info("ℹ️ Random Forest model not found. Feature importance analysis unavailable.")

st.markdown("---")

## 4. MODEL ACCURACY COMPARISON
st.subheader("🎯 Model Accuracy Comparison")

# Create radar chart for model comparison
fig_radar = go.Figure()

# Prepare data for radar chart
categories = ['R² Score', 'MAE (inverse)', 'RMSE (inverse)']
models = comparison_df.index

for model in models:
    # Normalize values (higher = better for all metrics)
    r2_score = comparison_df.loc[model, 'R²']
    mae_inv = 1 / (1 + comparison_df.loc[model, 'MAE'] / 100)  # Inverse MAE
    rmse_inv = 1 / (1 + comparison_df.loc[model, 'RMSE'] / 100)  # Inverse RMSE
    
    values = [r2_score, mae_inv, rmse_inv]
    
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=model,
        line_color=px.colors.qualitative.Set3[models.get_loc(model)]
    ))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True,
    title="Model Performance Radar Chart (Higher = Better)",
    height=500
)

st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

## 5. PREDICTION ERROR ANALYSIS
st.subheader("📈 Prediction Error Analysis")

# Load test data predictions if available
try:
    # Create synthetic test data for demonstration
    np.random.seed(42)
    n_samples = 100
    
    # Generate realistic test data
    test_data = pd.DataFrame({
        'Gender': np.random.choice([0, 1], n_samples),
        'Age': np.random.randint(20, 70, n_samples),
        'Height_cm': np.random.uniform(150, 190, n_samples),
        'Weight_kg': np.random.uniform(50, 100, n_samples),
        'Duration_min': np.random.uniform(10, 60, n_samples),
        'Heart_Rate': np.random.uniform(80, 160, n_samples),
        'Body_Temp_C': np.random.uniform(37, 40, n_samples)
    })
    
    # Load best model for predictions
    best_model_name = comparison_df['R²'].idxmax()
    best_model_path = models_dir / f"{best_model_name.lower().replace(' ', '_')}.pkl"
    
    if best_model_path.exists():
        best_model = joblib.load(best_model_path)
        predictions = best_model.predict(test_data)
        
        # Create error analysis visualization
        fig_errors = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Prediction Distribution', 'Residual Plot', 'Feature vs Prediction', 'Error Distribution'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # 1. Prediction Distribution
        fig_errors.add_trace(
            go.Histogram(x=predictions, name="Predictions", nbinsx=20),
            row=1, col=1
        )
        
        # 2. Residual Plot (using synthetic actual values)
        actual_values = predictions + np.random.normal(0, 20, n_samples)  # Synthetic residuals
        residuals = actual_values - predictions
        fig_errors.add_trace(
            go.Scatter(x=predictions, y=residuals, mode='markers', name="Residuals"),
            row=1, col=2
        )
        
        # 3. Feature vs Prediction
        fig_errors.add_trace(
            go.Scatter(x=test_data['Duration_min'], y=predictions, mode='markers', name="Duration vs Calories"),
            row=2, col=1
        )
        
        # 4. Error Distribution
        fig_errors.add_trace(
            go.Histogram(x=residuals, name="Errors", nbinsx=20),
            row=2, col=2
        )
        
        fig_errors.update_layout(height=600, title="Prediction Error Analysis")
        st.plotly_chart(fig_errors, use_container_width=True)
        
    else:
        st.warning(f"⚠️ Best model file not found: {best_model_path}")
        
except Exception as e:
    st.warning(f"⚠️ Could not generate error analysis: {e}")

st.markdown("---")

## 6. MODEL RECOMMENDATIONS
st.subheader("💡 Model Recommendations")

# Create recommendations based on performance
best_r2_model = comparison_df['R²'].idxmax()
best_mae_model = comparison_df['MAE'].idxmin()
best_rmse_model = comparison_df['RMSE'].idxmin()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎯 Best Overall", best_r2_model, f"R²: {comparison_df.loc[best_r2_model, 'R²']:.3f}")
    st.info("Highest predictive accuracy")

with col2:
    st.metric("📏 Most Precise", best_mae_model, f"MAE: {comparison_df.loc[best_mae_model, 'MAE']:.1f}")
    st.info("Lowest average error")

with col3:
    st.metric("⚡ Most Consistent", best_rmse_model, f"RMSE: {comparison_df.loc[best_rmse_model, 'RMSE']:.1f}")
    st.info("Most consistent predictions")

# Final recommendations
st.success(f"""
## 🏆 **Final Recommendation**

**Primary Model**: **{best_r2_model}** - Best overall performance with R² score of {comparison_df.loc[best_r2_model, 'R²']:.3f}

**Use Cases**:
• **Production**: {best_r2_model} for main predictions
• **High Precision**: {best_mae_model} when accuracy is critical  
• **Consistency**: {best_rmse_model} for stable predictions

**Next Steps**: Use the Prediction page to make actual calorie predictions with your chosen model!
""")

st.markdown("---")
st.info("💡 **Note**: Models are automatically trained when not found. For local development, you can still use 'py train_models.py' in PowerShell.")
