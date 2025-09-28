import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_shap_analysis(model, X_train, X_test, task_name, model_name, target, top_k=20):
    """
    Run SHAP analysis for sklearn models (pipeline or direct estimator).
    If PCA is used, SHAP values are back-projected to original feature space.
    Saves:
      - CSV of mean absolute SHAP importances (all features)
      - PNG SHAP summary plot (top-k features)
      - PNG bar plot of mean absolute SHAP (top-k features)
    """
    
    try:
        # Handle pipeline models
        if hasattr(model, "named_steps"):
            final_model = model.named_steps["model"]
            X_train_transformed = model[:-1].transform(X_train)
            X_test_transformed = model[:-1].transform(X_test)
            
            # Check if PCA exists
            if "pca" in model.named_steps:
                pca = model.named_steps["pca"]
            else:
                pca = None    
        else:
            final_model = model  
            X_train_transformed = X_train  
            X_test_transformed = X_test
            pca = None

        # --- Choose SHAP explainer ---
        if hasattr(final_model, "feature_importances_"):  # tree-based models
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(X_test_transformed)
        else:  # linear models / pipelines
            explainer = shap.Explainer(final_model, X_train_transformed)
            shap_values = explainer(X_test_transformed)
            
        # --- Back-project SHAP if PCA was used ---
        if pca is not None:
            # shap_values.values shape: (n_samples, n_components)
            shap_matrix = shap_values.values @ pca.components_
            feature_names = list(X_train.columns)
        else:
            shap_matrix = shap_values.values
            feature_names = (
                list(X_train.columns)
                if hasattr(X_train, "columns")
                else [f"f{i}" for i in range(X_train_transformed.shape[1])]
            )     

        # Mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)

        # Save CSV
        out_csv = f"results/shap_importance_{task_name}_{model_name}_{target}.csv"
        importance_df.to_csv(out_csv, index=False)
        print(f"SHAP importances saved: {out_csv}")
        
        # --- Select top-k features for plotting ---
        top_features = importance_df.head(top_k)
        top_idx = [feature_names.index(f) for f in top_features["feature"]]

        shap_matrix_top = shap_matrix[:, top_idx]
        X_test_top = X_test[top_features["feature"]]

        # --- Save summary plot (top-k only) ---
        plt.figure()
        shap.summary_plot(
            shap_matrix_top, X_test_top,
            feature_names=top_features["feature"].tolist(),
            show=False
        )
        out_png = f"results/shap_summary_{task_name}_{model_name}_{target}_top{top_k}.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"SHAP summary plot saved (top {top_k}): {out_png}")
        
        # --- Bar plot of mean absolute SHAP (top-k) ---
        plt.figure(figsize=(8, 6))
        top_features[::-1].plot(
            x="feature", y="mean_abs_shap",
            kind="barh", legend=False, color="skyblue"
        )
        plt.xlabel("Mean |SHAP value|")
        plt.title(f"Top {top_k} SHAP Feature Importances\n{model_name} - {target}")
        plt.tight_layout()
        out_png_bar = f"results/shap_bar_{task_name}_{model_name}_{target}_top{top_k}.png"
        plt.savefig(out_png_bar, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"SHAP bar plot saved (top {top_k}): {out_png_bar}")

    except Exception as e:
        print(f"SHAP analysis skipped for {model_name} ({target}): {e}")