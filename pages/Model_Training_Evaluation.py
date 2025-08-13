"""Page 2 — Model Training and Evaluation"""

import streamlit as st
from utils.data_loader import load_raw, merge_datasets, train_test_split_df, get_feature_target
from utils.model_utils import train_two_models, evaluate_on_holdout, save_model, permutation_feature_importance
from utils.visualization import actual_vs_pred_plot, feature_importance_bar
import pandas as pd

st.set_page_config(page_title="Model Training & Evaluation", layout="wide")
st.title("2 — Model Training & Evaluation")

ex_df, cal_df = load_raw()
try:
    merged = merge_datasets(ex_df, cal_df)
except Exception as exc:
    st.error(f"Failed to merge datasets: {exc}")
    st.stop()

st.write("## Training configuration")
test_size = st.slider("Holdout test size", 0.05, 0.5, 0.2, step=0.05)
random_state = int(st.number_input("Random state", value=42, step=1))
cv_folds = int(st.slider("Cross-validation folds", 3, 10, 5))

X, y = get_feature_target(merged)
X_train, X_test, y_train, y_test = train_test_split_df(merged, test_size=test_size, random_state=random_state)

if st.button("Train models (RandomForest + GradientBoosting)"):
    with st.spinner("Training models... this may take a moment"):
        results = train_two_models(X_train, y_train, cv_folds=cv_folds, random_state=random_state)

    # show CV metrics
    rows = []
    for name, info in results.items():
        rows.append({"Model": name, "RMSE_CV": info["rmse_cv"], "R2_CV": info["r2_cv"]})
    st.subheader("Cross-validated metrics")
    st.table(pd.DataFrame(rows).set_index("Model"))

    st.markdown("---")
    st.subheader("Holdout evaluation")
    for name, info in results.items():
        eval_metrics = evaluate_on_holdout(info["model"], X_test, y_test)
        st.write(f"**{name}**")
        st.json(eval_metrics)
        # actual vs predicted
        y_pred = info["model"].predict(X_test)
        actual_vs_pred_plot(y_test, y_pred)

    st.markdown("---")
    st.subheader("Feature importance (permutation)")
    for name, info in results.items():
        st.write(f"**{name}**")
        fi = permutation_feature_importance(info["model"], X_train, y_train, n_repeats=10, random_state=random_state)
        st.dataframe(fi)
        feature_importance_bar(fi)

    st.markdown("---")
    st.subheader("Save models")
    if st.button("Save all models to disk"):
        for name, info in results.items():
            path = save_model(info["model"], name.lower())
            st.write(f"Saved {name} -> {path}")
        # save best as best_model.pkl
        best_name, best_model = sorted(results.items(), key=lambda kv: (kv[1]["rmse_cv"], -kv[1]["r2_cv"]))[0]
        save_model(results[best_name]["model"], "best_model")
        st.success("Saved models and best_model.pkl")
