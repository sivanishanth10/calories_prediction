import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "pages", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

EXERCISE_CSV = os.path.join(DATA_DIR, "exercise.csv")
CALORIES_CSV = os.path.join(DATA_DIR, "calories.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "calories_model.pkl")

try:
    # Load datasets
    exercise_df = pd.read_csv(EXERCISE_CSV)
    calories_df = pd.read_csv(CALORIES_CSV)

    # Standardize column names
    exercise_df.columns = exercise_df.columns.str.strip().str.replace(" ", "_")
    calories_df.columns = calories_df.columns.str.strip().str.replace(" ", "_")

    # Merge on User_ID
    df = pd.merge(exercise_df, calories_df, on="User_ID")

    # Check required columns
    required_cols = ["Gender", "Age", "Height_cm", "Weight_kg", "Duration_min", "Heart_Rate", "Body_Temp", "Calories"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Encode Gender
    df["Gender"] = df["Gender"].map({"male": 0, "female": 1})

    # Features and target
    X = df[["Gender", "Age", "Height_cm", "Weight_kg", "Duration_min", "Heart_Rate", "Body_Temp"]]
    y = df["Calories"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.2f}")

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

except Exception as e:
    print(f"❌ Error: {e}")
