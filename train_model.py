import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

# File paths
 
calories_path = "data/calories.csv"
exercise_path = "data/exercise.csv"

# Load datasets
try:
    exercise_df = pd.read_csv(exercise_path)
    calories_df = pd.read_csv(calories_path)
except FileNotFoundError as e:
    print(f"❌ File not found: {e.filename}")
    exit()

# Merge datasets
df = pd.merge(exercise_df, calories_df, on="User_ID")

# ✅ Rename columns for clarity (optional but keeps consistency)
df.rename(columns={
    "Height": "Height_cm",
    "Weight": "Weight_kg",
    "Duration": "Duration_min"
}, inplace=True)

# ✅ Convert Body_Temp (Fahrenheit → Celsius)
if df["Body_Temp"].max() > 45:  # Simple check: if temps are too high, likely in °F
    df["Body_Temp_C"] = (df["Body_Temp"] - 32) * 5.0 / 9.0
else:
    df["Body_Temp_C"] = df["Body_Temp"]

# ✅ Convert Gender to numeric (male=1, female=0)
df["Gender"] = df["Gender"].astype(str).str.lower().map(lambda s: 1 if s.startswith("m") else 0)

# Features and target
X = df[["Gender", "Age", "Height_cm", "Weight_kg", "Duration_min", "Heart_Rate", "Body_Temp_C"]]
y = df["Calories"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
with open("models/calorie_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")
print(f"Training Score: {model.score(X_train, y_train):.2f}")
print(f"Test Score: {model.score(X_test, y_test):.2f}")
