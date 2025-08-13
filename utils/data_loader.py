"""Data loading, merging, preprocessing and helpers."""

from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import streamlit as st

# More robust path calculation
ROOT = Path(__file__).resolve().parents[1]  # project root /calories_prediction
DATA_DIR = ROOT / "data"

# Fallback: if the above doesn't work, try relative to current working directory
if not DATA_DIR.exists():
    DATA_DIR = Path("data")

EXERCISE_CSV = DATA_DIR / "exercise.csv"
CALORIES_CSV = DATA_DIR / "calories.csv"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    """Read CSV and raise clear errors."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File not found: {path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise RuntimeError(f"CSV at {path} is empty or invalid.") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV {path}: {exc}") from exc


@st.cache_data(ttl=3600)
def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the two CSVs and return them raw.
    """
    exercise = _safe_read_csv(EXERCISE_CSV)
    calories = _safe_read_csv(CALORIES_CSV)
    return exercise, calories


def merge_datasets(exercise_df: pd.DataFrame, calories_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge exercise and calories into a single canonical dataframe.

    This function attempts to handle common column-name variations. If your CSV
    uses different names, extend the COLUMN_MAP below.
    """
    COLUMN_MAP = {
        # possible exercise.csv columns -> canonical
        "Height": "Height_cm",
        "Height_cm": "Height_cm",
        "height_cm": "Height_cm",
        "Weight": "Weight_kg",
        "Weight_kg": "Weight_kg",
        "weight_kg": "Weight_kg",
        "Duration": "Duration_min",
        "Duration_min": "Duration_min",
        "duration_min": "Duration_min",
        "Body_Temp": "Body_Temp_C",
        "Body_Temperature": "Body_Temp_C",
        "Body_Temp_C": "Body_Temp_C",
        "BodyTemp": "Body_Temp_C",
        "Heart_Rate": "Heart_Rate",
        "HeartRate": "Heart_Rate",
        "Age": "Age",
        "Gender": "Gender",
        # calories file
        "Calories": "Calories",
        "calories": "Calories",
        "Burned_Calories": "Calories",
    }

    ex = exercise_df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in exercise_df.columns})
    cal = calories_df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in calories_df.columns})

    # find calories column in cal
    calories_candidates = [c for c in cal.columns if c.lower().strip() in ("calories", "burned_calories", "calorie")]
    if not calories_candidates:
        raise KeyError(f"Could not find 'Calories' column in calories.csv. Available: {list(cal.columns)}")
    calories_col = calories_candidates[0]
    cal = cal.rename(columns={calories_col: "Calories"})

    # merging strategy
    if len(ex) == len(cal):
        merged = pd.concat([ex.reset_index(drop=True), cal["Calories"].reset_index(drop=True)], axis=1)
    else:
        # try to find a shared identifier (rare). else align by index with warning.
        common = set(ex.columns).intersection(set(cal.columns)) - {"Calories"}
        if common:
            key = list(common)[0]
            merged = ex.merge(cal, on=key, how="inner")
        else:
            # best effort: align by index
            merged = pd.concat([ex.reset_index(drop=True), cal["Calories"].reindex(ex.index).reset_index(drop=True)], axis=1)

    # ensure required columns present
    required = ["Gender", "Age", "Height_cm", "Weight_kg", "Duration_min", "Heart_Rate", "Body_Temp_C", "Calories"]
    missing = [c for c in required if c not in merged.columns]
    if missing:
        raise KeyError(f"Missing required columns after merge: {missing}. Available columns: {list(merged.columns)}")

    # dtype conversions
    merged["Gender"] = merged["Gender"].astype(str).str.lower().map(lambda s: 1 if s.startswith("m") else 0)
    for col in ["Age", "Height_cm", "Weight_kg", "Duration_min", "Heart_Rate", "Body_Temp_C", "Calories"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged = merged.dropna(subset=required)
    merged.reset_index(drop=True, inplace=True)
    return merged


def get_feature_target(df: pd.DataFrame):
    """Return X (features) and y (target)."""
    features = ["Gender", "Age", "Height_cm", "Weight_kg", "Duration_min", "Heart_Rate", "Body_Temp_C"]
    X = df[features].copy()
    y = df["Calories"].copy()
    return X, y


def train_test_split_df(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Wrapper around sklearn.model_selection.train_test_split but avoids importing sklearn here."""
    from sklearn.model_selection import train_test_split
    X, y = get_feature_target(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
