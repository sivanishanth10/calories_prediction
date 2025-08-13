"""Page 4 — Make Predictions with Strong Visualizations"""

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

st.set_page_config(page_title="Calorie Prediction", layout="wide")
st.title("4 — Calorie Burn Prediction")

# Check if plotly is available
if not PLOTLY_AVAILABLE:
    st.warning("""
    ## ⚠️ Plotly Not Available
    
    This page requires Plotly for visualizations. Please:
    
    1. **Install Plotly**: `pip install plotly`
    2. **Restart the app**
    3. **Check requirements.txt** includes plotly
    
    For now, showing basic prediction interface only.
    """)
    
    # Show basic prediction interface without plots
    st.info("Basic prediction interface will be shown here once Plotly is installed.")
    st.stop()

# Try to load existing model
MODEL_PATH = Path("models/calorie_model.pkl")
model_loaded = False

if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        model_loaded = True
        st.success("✅ Pre-trained model loaded successfully!")
    except Exception as e:
        st.warning(f"⚠️ Could not load existing model: {e}")
        model_loaded = False
else:
    st.info("ℹ️ No pre-trained model found. Please run 'py train_models.py' in PowerShell first.")
    st.stop()

# Load data for visualizations
try:
    ex_df, cal_df = load_raw()
    merged_df = merge_datasets(ex_df, cal_df)
    st.success("✅ Data loaded for visualizations!")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# Prediction Interface
if model_loaded:
    st.subheader("🎯 Make Your Prediction")
    st.markdown("Enter your details below to predict calories burned:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["male", "female"])
        age = st.number_input("Age", min_value=10, max_value=100, value=30)
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    
    with col2:
        duration = st.number_input("Exercise Duration (minutes)", min_value=1, max_value=300, value=30)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=50, max_value=200, value=120)
        body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)
    
    # Convert inputs to model format
    gender_encoded = 1 if gender == "male" else 0
    
    # Create prediction button
    if st.button("🚀 Predict Calories Burned", type="primary"):
        try:
            # Prepare input data
            input_data = pd.DataFrame({
                "Gender": [gender_encoded],
                "Age": [age],
                "Height_cm": [height],
                "Weight_kg": [weight],
                "Duration_min": [duration],
                "Heart_Rate": [heart_rate],
                "Body_Temp_C": [body_temp]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.markdown("---")
            st.subheader("🎯 Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Calories", f"{prediction:.1f}", "calories")
            with col2:
                st.metric("Exercise Duration", f"{duration} min")
            with col3:
                st.metric("Heart Rate", f"{heart_rate} bpm")
            
            # Additional insights
            st.markdown("### 💡 Insights")
            if prediction > 200:
                st.success("🔥 High intensity workout! Great calorie burn!")
            elif prediction > 100:
                st.info("⚡ Moderate intensity - good workout!")
            else:
                st.warning("💪 Light activity - consider increasing intensity or duration")
            
            # Show input summary
            st.markdown("### 📊 Your Input Summary")
            input_summary = pd.DataFrame({
                "Feature": ["Gender", "Age", "Height", "Weight", "Duration", "Heart Rate", "Body Temperature"],
                "Value": [gender, f"{age} years", f"{height} cm", f"{weight} kg", f"{duration} min", f"{heart_rate} bpm", f"{body_temp}°C"]
            })
            st.dataframe(input_summary, use_container_width=True)
            
            # STRONG PREDICTION VISUALIZATIONS
            st.markdown("---")
            st.subheader("📈 Advanced Prediction Analysis")
            
            ## 1. PREDICTION CONFIDENCE INTERVAL
            st.markdown("### 🎯 Prediction Confidence Analysis")
            
            # Generate confidence intervals using model uncertainty
            np.random.seed(42)
            n_samples = 1000
            
            # Create variations of input data
            input_variations = pd.DataFrame({
                "Gender": [gender_encoded] * n_samples,
                "Age": np.random.normal(age, age * 0.1, n_samples),
                "Height_cm": np.random.normal(height, height * 0.02, n_samples),
                "Weight_kg": np.random.normal(weight, weight * 0.05, n_samples),
                "Duration_min": np.random.normal(duration, duration * 0.1, n_samples),
                "Heart_Rate": np.random.normal(heart_rate, heart_rate * 0.05, n_samples),
                "Body_Temp_C": np.random.normal(body_temp, body_temp * 0.01, n_samples)
            })
            
            # Clip to reasonable ranges
            input_variations["Age"] = np.clip(input_variations["Age"], 10, 100)
            input_variations["Height_cm"] = np.clip(input_variations["Height_cm"], 100, 250)
            input_variations["Weight_kg"] = np.clip(input_variations["Weight_kg"], 30, 200)
            input_variations["Duration_min"] = np.clip(input_variations["Duration_min"], 1, 300)
            input_variations["Heart_Rate"] = np.clip(input_variations["Heart_Rate"], 50, 200)
            input_variations["Body_Temp_C"] = np.clip(input_variations["Body_Temp_C"], 35, 42)
            
            # Make predictions
            predictions_variations = model.predict(input_variations)
            
            # Calculate confidence intervals
            confidence_95 = np.percentile(predictions_variations, [2.5, 97.5])
            confidence_68 = np.percentile(predictions_variations, [16, 84])
            
            # Create confidence interval visualization
            fig_confidence = go.Figure()
            
            # Add main prediction
            fig_confidence.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Calories Burned"},
                delta={'reference': np.mean(predictions_variations)},
                gauge={
                    'axis': {'range': [None, max(predictions_variations) * 1.1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, confidence_68[0]], 'color': "lightgray"},
                        {'range': [confidence_68[0], confidence_68[1]], 'color': "lightblue"},
                        {'range': [confidence_68[1], confidence_95[1]], 'color': "lightgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': prediction
                    }
                }
            ))
            
            fig_confidence.update_layout(
                title="Prediction Confidence Gauge",
                height=400
            )
            
            st.plotly_chart(fig_confidence, use_container_width=True)
            
            # Confidence interval metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("95% Confidence", f"{confidence_95[0]:.1f} - {confidence_95[1]:.1f}")
            with col2:
                st.metric("68% Confidence", f"{confidence_68[0]:.1f} - {confidence_68[1]:.1f}")
            with col3:
                st.metric("Standard Deviation", f"{np.std(predictions_variations):.1f}")
            
            ## 2. FEATURE SENSITIVITY ANALYSIS
            st.markdown("### 🔍 Feature Sensitivity Analysis")
            
            # Analyze how each feature affects the prediction
            base_input = np.array([gender_encoded, age, height, weight, duration, heart_rate, body_temp])
            feature_names = ["Gender", "Age", "Height_cm", "Weight_kg", "Duration_min", "Heart_Rate", "Body_Temp_C"]
            
            sensitivity_results = []
            
            for i, feature in enumerate(feature_names):
                # Vary each feature by ±20%
                variations = np.linspace(base_input[i] * 0.8, base_input[i] * 1.2, 10)
                predictions_var = []
                
                for var in variations:
                    test_input = base_input.copy()
                    test_input[i] = var
                    pred = model.predict([test_input])[0]
                    predictions_var.append(pred)
                
                sensitivity_results.append({
                    'Feature': feature,
                    'Variations': variations,
                    'Predictions': predictions_var,
                    'Sensitivity': np.std(predictions_var)
                })
            
            # Create sensitivity visualization
            fig_sensitivity = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Duration Sensitivity', 'Heart Rate Sensitivity', 'Age Sensitivity', 'Weight Sensitivity'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # Plot top 4 most sensitive features
            top_features = sorted(sensitivity_results, key=lambda x: x['Sensitivity'], reverse=True)[:4]
            
            for i, result in enumerate(top_features):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                fig_sensitivity.add_trace(
                    go.Scatter(
                        x=result['Variations'],
                        y=result['Predictions'],
                        mode='lines+markers',
                        name=result['Feature'],
                        line=dict(width=3)
                    ),
                    row=row, col=col
                )
            
            fig_sensitivity.update_layout(height=600, title="Feature Sensitivity Analysis")
            st.plotly_chart(fig_sensitivity, use_container_width=True)
            
            ## 3. COMPARISON WITH DATASET
            st.markdown("### 📊 Comparison with Dataset Distribution")
            
            # Compare user's prediction with dataset statistics
            fig_comparison = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Your Prediction vs Dataset', 'Feature Comparison'),
                specs=[[{"type": "histogram"}, {"type": "bar"}]]
            )
            
            # Histogram of dataset calories
            fig_comparison.add_trace(
                go.Histogram(
                    x=merged_df['Calories'],
                    name="Dataset Calories",
                    nbinsx=30,
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Add user's prediction as vertical line
            fig_comparison.add_vline(
                x=prediction,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Your Prediction: {prediction:.1f}",
                row=1, col=1
            )
            
            # Feature comparison bars
            user_features = [age, height, weight, duration, heart_rate, body_temp]
            dataset_means = [
                merged_df['Age'].mean(),
                merged_df['Height_cm'].mean(),
                merged_df['Weight_kg'].mean(),
                merged_df['Duration_min'].mean(),
                merged_df['Heart_Rate'].mean(),
                merged_df['Body_Temp_C'].mean()
            ]
            
            feature_labels = ['Age', 'Height', 'Weight', 'Duration', 'Heart Rate', 'Body Temp']
            
            fig_comparison.add_trace(
                go.Bar(
                    x=feature_labels,
                    y=user_features,
                    name="Your Values",
                    marker_color="red"
                ),
                row=1, col=2
            )
            
            fig_comparison.add_trace(
                go.Bar(
                    x=feature_labels,
                    y=dataset_means,
                    name="Dataset Average",
                    marker_color="blue"
                ),
                row=1, col=2
            )
            
            fig_comparison.update_layout(height=500, title="Dataset Comparison")
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            ## 4. OPTIMIZATION RECOMMENDATIONS
            st.markdown("### 💡 Optimization Recommendations")
            
            # Find optimal values for better calorie burn
            optimal_duration = duration
            optimal_heart_rate = heart_rate
            
            # Test different durations
            duration_range = np.linspace(duration * 0.5, duration * 2, 20)
            duration_predictions = []
            
            for dur in duration_range:
                test_input = base_input.copy()
                test_input[4] = dur  # Duration index
                pred = model.predict([test_input])[0]
                duration_predictions.append(pred)
            
            # Test different heart rates
            heart_rate_range = np.linspace(heart_rate * 0.8, heart_rate * 1.2, 20)
            heart_rate_predictions = []
            
            for hr in heart_rate_range:
                test_input = base_input.copy()
                test_input[5] = hr  # Heart rate index
                pred = model.predict([test_input])[0]
                heart_rate_predictions.append(pred)
            
            # Create optimization visualization
            fig_optimization = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Duration Optimization', 'Heart Rate Optimization'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            fig_optimization.add_trace(
                go.Scatter(
                    x=duration_range,
                    y=duration_predictions,
                    mode='lines+markers',
                    name="Duration vs Calories",
                    line=dict(width=3)
                ),
                row=1, col=1
            )
            
            fig_optimization.add_vline(
                x=duration,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Current: {duration} min",
                row=1, col=1
            )
            
            fig_optimization.add_trace(
                go.Scatter(
                    x=heart_rate_range,
                    y=heart_rate_predictions,
                    mode='lines+markers',
                    name="Heart Rate vs Calories",
                    line=dict(width=3)
                ),
                row=1, col=2
            )
            
            fig_optimization.add_vline(
                x=heart_rate,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Current: {heart_rate} bpm",
                row=1, col=2
            )
            
            fig_optimization.update_layout(height=500, title="Optimization Analysis")
            st.plotly_chart(fig_optimization, use_container_width=True)
            
            # Optimization insights
            best_duration_idx = np.argmax(duration_predictions)
            best_heart_rate_idx = np.argmax(heart_rate_predictions)
            
            best_duration = duration_range[best_duration_idx]
            best_heart_rate = heart_rate_range[best_heart_rate_idx]
            max_calories = max(duration_predictions)
            
            st.success(f"""
            ## 🚀 **Optimization Insights**
            
            **Best Duration**: {best_duration:.1f} minutes (vs your {duration} min)
            **Best Heart Rate**: {best_heart_rate:.1f} bpm (vs your {heart_rate} bpm)
            **Maximum Calories**: {max_calories:.1f} calories
            
            **Improvement**: +{max_calories - prediction:.1f} calories ({((max_calories - prediction) / prediction * 100):.1f}% increase)
            """)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")
    
    # Show model info
    st.markdown("---")
    st.subheader("ℹ️ Model Information")
    st.info("""
    This model uses machine learning algorithms trained on exercise data including:
    - **Personal factors**: Gender, Age, Height, Weight
    - **Exercise factors**: Duration, Heart Rate, Body Temperature
    
    The model predicts calories burned based on these physiological and exercise parameters.
    """)
    
    # Show sample predictions
    st.markdown("### 📈 Sample Predictions")
    sample_data = [
        {"Gender": "Male", "Age": 25, "Height": 175, "Weight": 70, "Duration": 30, "Heart_Rate": 140, "Body_Temp": 38.5},
        {"Gender": "Female", "Age": 30, "Height": 165, "Weight": 60, "Duration": 45, "Heart_Rate": 150, "Body_Temp": 39.0},
        {"Gender": "Male", "Age": 40, "Height": 180, "Weight": 80, "Duration": 60, "Heart_Rate": 160, "Body_Temp": 39.5}
    ]
    
    sample_df = pd.DataFrame(sample_data)
    sample_df["Gender_Encoded"] = sample_df["Gender"].map({"Male": 1, "Female": 0})
    
    # Make predictions for sample data
    sample_inputs = sample_df[["Gender_Encoded", "Age", "Height", "Weight", "Duration", "Heart_Rate"]].copy()
    sample_inputs.columns = ["Gender", "Age", "Height_cm", "Weight_kg", "Duration_min", "Heart_Rate"]
    sample_inputs["Body_Temp_C"] = sample_df["Body_Temp"]
    
    try:
        sample_predictions = model.predict(sample_inputs)
        sample_df["Predicted_Calories"] = sample_predictions.round(1)
        st.dataframe(sample_df[["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Predicted_Calories"]], use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate sample predictions: {e}")
