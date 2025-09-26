import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_shap_analysis(model, X_train, X_test, task_name, model_name, target):
    """
    Run SHAP analysis for any sklearn model and save:
    - CSV of mean absolute SHAP importances
    - PNG summary plot
    """
    try:
        # Detect explainer type
        if hasattr(model, "feature_importances_"):  # tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:  # linear models / pipelines
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)

        # Compute mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": X_train.columns if hasattr(X_train, "columns") else [f"f{i}" for i in range(X_train.shape[1])],
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)

        # Save CSV
        out_csv = f"results/shap_importance_{task_name}_{model_name}_{target}.csv"
        importance_df.to_csv(out_csv, index=False)
        print(f"SHAP importances saved: {out_csv}")

        # Save summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_test, feature_names=X_train.columns, show=False)
        out_png = f"results/shap_summary_{task_name}_{model_name}_{target}.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"SHAP summary plot saved: {out_png}")

    except Exception as e:
        print(f"SHAP analysis skipped for {model_name} ({target}): {e}")
