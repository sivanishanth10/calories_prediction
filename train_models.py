"""
Enhanced Model Training Script
Trains multiple models and saves them for the Streamlit app
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from pathlib import Path

def load_and_prepare_data():
    """Load and prepare the dataset"""
    print("📊 Loading data...")
    
    # Load datasets
    exercise_df = pd.read_csv("data/exercise.csv")
    calories_df = pd.read_csv("data/calories.csv")
    
    print(f"✅ Exercise data: {exercise_df.shape}")
    print(f"✅ Calories data: {calories_df.shape}")
    
    # Merge datasets
    df = pd.merge(exercise_df, calories_df, on="User_ID")
    print(f"✅ Merged data: {df.shape}")
    
    # Rename columns for consistency
    df.rename(columns={
        "Height": "Height_cm",
        "Weight": "Weight_kg", 
        "Duration": "Duration_min"
    }, inplace=True)
    
    # Convert Body_Temp to Body_Temp_C (assuming it's already in Celsius)
    df["Body_Temp_C"] = df["Body_Temp"]
    
    # Encode Gender (male=1, female=0)
    df["Gender"] = df["Gender"].astype(str).str.lower().map(lambda s: 1 if s.startswith("m") else 0)
    
    # Select features and target
    features = ["Gender", "Age", "Height_cm", "Weight_kg", "Duration_min", "Heart_Rate", "Body_Temp_C"]
    X = df[features]
    y = df["Calories"]
    
    print(f"✅ Features: {features}")
    print(f"✅ Target: Calories")
    
    return X, y, df

def train_and_evaluate_models(X, y):
    """Train multiple models and evaluate them"""
    print("\n🚀 Training models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel='rbf', C=100, gamma='scale')
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n📈 Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        results[name] = {
            "model": model,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std()
        }
        
        print(f"   MAE: {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   R²: {r2:.3f}")
        print(f"   CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return results, X_test, y_test

def save_models(results):
    """Save all trained models"""
    print("\n💾 Saving models...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Save each model
    for name, result in results.items():
        filename = f"models/{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(result["model"], filename)
        print(f"✅ Saved: {filename}")
    
    # Save results summary
    results_summary = {}
    for name, result in results.items():
        results_summary[name] = {
            "MAE": round(result["mae"], 2),
            "RMSE": round(result["rmse"], 2),
            "R²": round(result["r2"], 3),
            "CV R²": f"{result['cv_mean']:.3f} ± {result['cv_std']:.3f}"
        }
    
    # Save as CSV for easy viewing
    summary_df = pd.DataFrame(results_summary).T
    summary_df.to_csv("models/model_comparison.csv")
    print("✅ Saved: models/model_comparison.csv")
    
    return results_summary

def print_summary(results_summary):
    """Print a nice summary of all results"""
    print("\n" + "="*80)
    print("🏆 MODEL TRAINING SUMMARY")
    print("="*80)
    
    # Find best model by R² score
    best_model = max(results_summary.items(), key=lambda x: float(x[1]["R²"]))
    
    print(f"\n🥇 BEST MODEL: {best_model[0]}")
    print(f"   R² Score: {best_model[1]['R²']}")
    print(f"   MAE: {best_model[1]['MAE']} calories")
    print(f"   RMSE: {best_model[1]['RMSE']} calories")
    
    print(f"\n📊 ALL MODELS COMPARISON:")
    print("-" * 60)
    
    for name, metrics in results_summary.items():
        print(f"{name:20} | R²: {metrics['R²']:6} | MAE: {metrics['MAE']:6} | RMSE: {metrics['RMSE']:6}")
    
    print(f"\n💡 Models saved in 'models/' directory")
    print(f"📈 Use 'models/model_comparison.csv' for detailed comparison")
    print("="*80)

def main():
    """Main training function"""
    try:
        print("🔥 CALORIE PREDICTION MODEL TRAINING")
        print("="*50)
        
        # Load and prepare data
        X, y, df = load_and_prepare_data()
        
        # Train and evaluate models
        results, X_test, y_test = train_and_evaluate_models(X, y)
        
        # Save models
        results_summary = save_models(results)
        
        # Print summary
        print_summary(results_summary)
        
        print("\n🎉 Training completed successfully!")
        print("🚀 You can now run the Streamlit app with trained models!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
