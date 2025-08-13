"""Model training, evaluation, saving/loading and importance utilities."""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import logging

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


def train_two_models(
    X: pd.DataFrame,
    y: pd.Series,
    rf_params: Dict[str, Any] | None = None,
    gb_params: Dict[str, Any] | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train RandomForest and GradientBoosting, compute CV metrics, and return fitted models and metrics.
    """
    rf_params = rf_params or {"n_estimators": 200, "random_state": random_state, "n_jobs": -1}
    gb_params = gb_params or {"n_estimators": 200, "random_state": random_state}

    rf = RandomForestRegressor(**rf_params)
    gb = GradientBoostingRegressor(**gb_params)

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = {}

    for name, model in [("RandomForest", rf), ("GradientBoosting", gb)]:
        # cross-validated negative MSE => RMSE
        try:
            neg_mse = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf, n_jobs=-1)
            r2 = cross_val_score(model, X, y, scoring="r2", cv=kf, n_jobs=-1)
            rmse_cv = float(np.sqrt(-neg_mse).mean())
            r2_cv = float(r2.mean())
        except Exception as exc:
            logger.warning("Cross-val failed: %s", exc)
            rmse_cv = float("nan")
            r2_cv = float("nan")

        # fit on full data
        model.fit(X, y)
        results[name] = {"model": model, "rmse_cv": rmse_cv, "r2_cv": r2_cv}

    return results


def evaluate_on_holdout(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Return MAE, MSE, RMSE, R2 on holdout set."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    return {"MAE": float(mae), "MSE": float(mse), "RMSE": rmse, "R2": r2}


def save_model(model, name: str) -> Path:
    """Save model to models/<name>.pkl and return the path."""
    path = MODELS_DIR / f"{name}.pkl"
    dump(model, path)
    return path


def load_model(name: str):
    """Load model from models/<name>.pkl or raise FileNotFoundError."""
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return load(path)


def permutation_feature_importance(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
    """
    Return a DataFrame with features and their permutation importance.
    """
    perm = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    fi = pd.DataFrame(
        {"feature": X.columns.tolist(), "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return fi


def select_best(results: Dict[str, Any]) -> Tuple[str, Any]:
    """
    Select best model (lowest rmse_cv, tie-breaker highest r2_cv). Returns (name, model).
    """
    best = None
    best_score = None
    for name, info in results.items():
        try:
            score = (info.get("rmse_cv", float("inf")), -info.get("r2_cv", float("-inf")))
        except Exception:
            score = (float("inf"), float("-inf"))
        if best_score is None or score < best_score:
            best_score = score
            best = (name, info["model"])
    return best
