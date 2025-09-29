import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import Pipeline

def _is_tree_model(est):
    # crude but works well for sklearn trees/ensembles
    return hasattr(est, "feature_importances_")

def _unwrap_pipeline(model):
    """Return (transformer, final_estimator, pca_or_None)."""
    if isinstance(model, Pipeline):
        transformer = model[:-1] if len(model.steps) > 1 else None
        final_est = model.named_steps["model"]
        pca = model.named_steps.get("pca", None)
        return transformer, final_est, pca
    else:
        return None, model, None

def _ensure_2d(X):
    # SHAP expects 2D arrays/DataFrames
    if hasattr(X, "values"):
        return X.values
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

def _to_numpy(shap_out):
    """Normalize SHAP output to a numpy array (n_samples, n_features)."""
    # TreeExplainer often returns np.ndarray directly
    if isinstance(shap_out, np.ndarray):
        return shap_out
    # Newer SHAP returns Explanation objects
    if hasattr(shap_out, "values"):
        return shap_out.values
    # Fallback
    return np.asarray(shap_out)

def run_shap_analysis(model, X_train, X_test, task_name, model_name, target, top_k=20, results_dir="results"):
    """
    Run SHAP for a single-output sklearn model or Pipeline.
    If PCA is present in the pipeline, SHAP values are back-projected to the original feature space.
    Saves:
      - CSV of mean |SHAP|
      - summary plot (top-k)
      - bar plot (top-k)
    """
    os.makedirs(results_dir, exist_ok=True)

    # 1) Unwrap pipeline and transform data if needed
    transformer, final_model, pca = _unwrap_pipeline(model)

    if transformer is not None:
        X_train_tf = transformer.transform(X_train)
        X_test_tf = transformer.transform(X_test)
    else:
        X_train_tf = X_train
        X_test_tf = X_test

    X_train_tf = _ensure_2d(X_train_tf)
    X_test_tf = _ensure_2d(X_test_tf)

    # 2) Choose explainer
    if _is_tree_model(final_model):
        explainer = shap.TreeExplainer(final_model)
        shap_raw = explainer.shap_values(X_test_tf)
    else:
        # Linear / other models
        # Using shap.Explainer covers LinearExplainer/KernelExplainer internally
        explainer = shap.Explainer(final_model, X_train_tf)
        shap_raw = explainer(X_test_tf)

    shap_raw = _to_numpy(shap_raw)  # (n_samples, n_features_in_tf)

    # 3) If PCA present, back-project SHAP values to original feature space
    #    Back-projection: SHAP_PCA (n_samples, n_components) @ pca.components_ (n_components, n_features_original)
    if pca is not None:
        shap_matrix = shap_raw @ pca.components_
        feature_names = list(X_train.columns) if hasattr(X_train, "columns") else [f"f{i}" for i in range(shap_matrix.shape[1])]
    else:
        shap_matrix = shap_raw
        if hasattr(X_train, "columns"):
            feature_names = list(X_train.columns)
        else:
            feature_names = [f"f{i}" for i in range(shap_matrix.shape[1])]

    # 4) Mean |SHAP| in the **same space we will plot** (i.e., shap_matrix)
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)

    out_csv = os.path.join(results_dir, f"shap_importance_{task_name}_{model_name}_{target}.csv")
    importance_df.to_csv(out_csv, index=False)
    print(f"SHAP importances saved: {out_csv}")

    # 5) Plot top-k
    top = importance_df.head(top_k).copy()
    top_features = top["feature"].tolist()
    top_idx = [feature_names.index(f) for f in top_features]

    shap_top = shap_matrix[:, top_idx]

    # Build a DataFrame for summary plot colors (need test data for these features)
    if hasattr(X_test, "loc"):
        X_test_top = X_test.loc[:, top_features]
    else:
        # If X_test is numpy, just pass the array; SHAP will show without feature values
        X_test_top = None

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_top, X_test_top, feature_names=top_features, show=False)
    out_png = os.path.join(results_dir, f"shap_summary_{task_name}_{model_name}_{target}_top{top_k}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved (top {top_k}): {out_png}")

    # Bar plot
    plt.figure(figsize=(8, 6))
    # plot reversed for horizontal bars from largest at top
    top[::-1].plot(x="feature", y="mean_abs_shap", kind="barh", legend=False)
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top {top_k} SHAP Feature Importances\n{model_name} - {target}")
    plt.tight_layout()
    out_png_bar = os.path.join(results_dir, f"shap_bar_{task_name}_{model_name}_{target}_top{top_k}.png")
    plt.savefig(out_png_bar, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP bar plot saved (top {top_k}): {out_png_bar}")
