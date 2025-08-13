"""Page 3 — Model Comparison"""

import streamlit as st
from utils.data_loader import load_raw, merge_datasets, train_test_split_df, get_feature_target
from utils.model_utils import train_two_models, evaluate_on_holdout, select_best, save_model
import pandas as pd

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("3 — Model Comparison")

ex_df, cal_df = load_raw()
try:
    merged = merge_datasets(ex_df, cal_df)
except Exception as exc:
    st.error(f"Failed to merge data: {exc}")
    st.stop()

test_size = st.slider("Holdout test size", 0.05, 0.5, 0.2)
random_state = int(st.number_input("Random state", value=42, step=1))
cv_folds = int(st.slider("CV folds", 3, 10, 5))

X_train, X_test, y_train, y_test = train_test_split_df(merged, test_size=test_size, random_state=random_state)

if st.button("Train & Compare models"):
    with st.spinner("Training and comparing..."):
        results = train_two_models(X_train, y_train, cv_folds=cv_folds, random_state=random_state)

    rows = []
    for name, info in results.items():
        eval_metrics = evaluate_on_holdout(info["model"], X_test, y_test)
        rows.append({
            "Model": name,
            "RMSE_CV": info.get("rmse_cv"),
            "R2_CV": info.get("r2_cv"),
            "MAE_holdout": eval_metrics["MAE"],
            "RMSE_holdout": eval_metrics["RMSE"],
            "R2_holdout": eval_metrics["R2"],
        })

    comp_df = pd.DataFrame(rows).set_index("Model")
    st.subheader("Comparison table")
    st.dataframe(comp_df)

    st.markdown("### Best model selection")
    best_name, best_model = select_best(results)
    st.success(f"Selected best model: **{best_name}**")
    if st.button("Save best model as `best_model.pkl`"):
        save_model(best_model, "best_model")
        st.success("Saved best_model.pkl")
