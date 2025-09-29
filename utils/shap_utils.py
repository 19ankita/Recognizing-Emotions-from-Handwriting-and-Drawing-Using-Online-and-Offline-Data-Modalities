# utils/shap_utils.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import Pipeline

# ---------------------------
# Small helpers
# ---------------------------

def _is_tree_model(est):
    """Detect common sklearn tree/ensemble regressors."""
    return hasattr(est, "feature_importances_")

def _unwrap_pipeline(model):
    """
    Return (transformer, final_estimator, pca_or_None, scaler_or_None).
    transformer is a Pipeline slice (all steps except the final 'model').
    """
    scaler = None
    if isinstance(model, Pipeline):
        transformer = model[:-1] if len(model.steps) > 1 else None
        final_est = model.named_steps["model"]
        pca = model.named_steps.get("pca", None)
        scaler = model.named_steps.get("scaler", None)
        return transformer, final_est, pca, scaler
    else:
        return None, model, None, None

def _ensure_2d(X):
    """SHAP expects 2D data."""
    if hasattr(X, "values"):
        return X.values
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

def _to_numpy(shap_out):
    """
    Normalize SHAP output to a numpy array (n_samples, n_features).
    Works with Explanation objects and raw ndarrays.
    """
    if isinstance(shap_out, np.ndarray):
        return shap_out
    if hasattr(shap_out, "values"):
        return shap_out.values
    return np.asarray(shap_out)

# ---------------------------
# Linear coefficient back-projection (direction)
# ---------------------------

def backproject_linear_coefs(pipeline_or_model, feature_names):
    """
    For a linear model in a pipeline (Scaler -> [PCA] -> Linear),
    back-project coefficients to original feature space.

    Returns a DataFrame with columns:
      - feature, coef_orig, coef_std, intercept_orig
    If scaler/PCA are absent, falls back gracefully.
    """
    transformer, final_est, pca, scaler = _unwrap_pipeline(pipeline_or_model)
    if not hasattr(final_est, "coef_"):
        return None  # not linear or no coef available

    beta = np.ravel(final_est.coef_)  # coefficients in the model's input space

    # If PCA exists, beta is in PCA space -> map back to standardized feature space
    if pca is not None:
        w_std = pca.components_.T @ beta  # (n_features,)
    else:
        w_std = beta

    # Map from standardized space to original space using scaler
    if scaler is not None and hasattr(scaler, "scale_"):
        w_orig = w_std / scaler.scale_
        # Intercept adjustment from standardized to original space:
        # y = w_std^T ((X - mu)/sigma) + b  =>  y = (w_std/sigma)^T X + (b - (w_std/sigma)^T mu)
        mu = getattr(scaler, "mean_", np.zeros_like(w_std))
        intercept_orig = float(final_est.intercept_) - float(np.dot(w_orig, mu))
    else:
        w_orig = w_std
        intercept_orig = float(getattr(final_est, "intercept_", 0.0))

    coef_df = pd.DataFrame(
        {"feature": list(feature_names), "coef_orig": w_orig, "coef_std": w_std}
    ).sort_values("coef_orig", key=np.abs, ascending=False)
    coef_df["intercept_orig"] = intercept_orig
    return coef_df

# ---------------------------
# Main SHAP routine
# ---------------------------

def run_shap_analysis(
    model, X_train, X_test, task_name, model_name, target, top_k=20, results_dir="results"
):
    """
    Run SHAP for a single-output sklearn model or Pipeline.
    If PCA is present, SHAP values are back-projected to the original feature space.

    Saves:
      - shap_importance_...csv  (mean |SHAP| per feature)
      - shap_beeswarm_...png    (signed/directional SHAP)
      - shap_bar_...png         (magnitude-only bar)
      - shap_signed_mean_...csv (mean signed SHAP)
      - shap_bar_signed_...png  (signed bar)
      - coefs_...csv + coefs_bar_...png (linear models only)
    """
    os.makedirs(results_dir, exist_ok=True)

    # 1) Unwrap pipeline and transform data if needed
    transformer, final_model, pca, scaler = _unwrap_pipeline(model)

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
        explainer = shap.Explainer(final_model, X_train_tf)
        shap_raw = explainer(X_test_tf)

    shap_raw = _to_numpy(shap_raw)  # (n_samples, n_features_in_tf)

    # 3) Back-project SHAP to original feature space if PCA was used
    if pca is not None:
        # shap_raw: (n_samples, n_components); components_: (n_components, n_features_orig)
        shap_matrix = shap_raw @ pca.components_
        feature_names = list(X_train.columns) if hasattr(X_train, "columns") else [f"f{i}" for i in range(shap_matrix.shape[1])]
    else:
        shap_matrix = shap_raw
        feature_names = list(X_train.columns) if hasattr(X_train, "columns") else [f"f{i}" for i in range(shap_matrix.shape[1])]

    # 4) Global magnitude (mean |SHAP|)
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap}).sort_values(
        "mean_abs_shap", ascending=False
    )

    out_csv = os.path.join(results_dir, f"shap_importance_{task_name}_{model_name}_{target}.csv")
    importance_df.to_csv(out_csv, index=False)
    print(f"SHAP importances saved: {out_csv}")

    # 5) Select top-k features (by magnitude) for plotting
    top = importance_df.head(top_k).copy()
    top_features = top["feature"].tolist()
    top_idx = [feature_names.index(f) for f in top_features]
    shap_top = shap_matrix[:, top_idx]

    # Prepare X slice for colored beeswarm (optional)
    if hasattr(X_test, "loc"):
        X_test_top = X_test.loc[:, top_features]
    else:
        X_test_top = None

    # ---------------------------
    # (A) Beeswarm: signed direction
    # ---------------------------
    plt.figure()
    shap.summary_plot(shap_top, X_test_top, feature_names=top_features, show=False, plot_type="dot")
    out_bee = os.path.join(results_dir, f"shap_beeswarm_{task_name}_{model_name}_{target}_top{top_k}.png")
    plt.savefig(out_bee, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP beeswarm saved: {out_bee}")

    # ---------------------------
    # (B) Magnitude-only bar (what you already had)
    # ---------------------------
    plt.figure(figsize=(8, 6))
    top[::-1].plot(x="feature", y="mean_abs_shap", kind="barh", legend=False)
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top {top_k} SHAP Feature Importances\n{model_name} - {target}")
    plt.tight_layout()
    out_png_bar = os.path.join(results_dir, f"shap_bar_{task_name}_{model_name}_{target}_top{top_k}.png")
    plt.savefig(out_png_bar, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP bar (magnitude) saved: {out_png_bar}")

    # ---------------------------
    # (C) Signed SHAP (quick up/down view)
    # ---------------------------
    signed_mean = shap_matrix.mean(axis=0)
    signed_df = pd.DataFrame({"feature": feature_names, "signed_mean_shap": signed_mean}).set_index("feature").loc[top_features].reset_index()

    signed_csv = os.path.join(results_dir, f"shap_signed_mean_{task_name}_{model_name}_{target}.csv")
    signed_df.to_csv(signed_csv, index=False)
    print(f"Signed mean SHAP saved: {signed_csv}")

    plt.figure(figsize=(8, 6))
    signed_df[::-1].plot(x="feature", y="signed_mean_shap", kind="barh", legend=False)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Mean SHAP value (signed)")
    plt.title(f"Top {top_k} SHAP (signed)\n{model_name} - {target}")
    plt.tight_layout()
    out_png_signed = os.path.join(results_dir, f"shap_bar_signed_{task_name}_{model_name}_{target}_top{top_k}.png")
    plt.savefig(out_png_signed, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP bar (signed) saved: {out_png_signed}")

    # ---------------------------
    # (D) Linear models: back-projected coefficients (direction)
    # ---------------------------
    coef_df = backproject_linear_coefs(model, feature_names)
    if coef_df is not None:
        out_coef_csv = os.path.join(results_dir, f"coefs_{task_name}_{model_name}_{target}.csv")
        coef_df.to_csv(out_coef_csv, index=False)
        print(f"Back-projected linear coefficients saved: {out_coef_csv}")

        # Bar chart for top-|coef| to mirror SHAP top_k
        coef_top = coef_df.head(top_k).copy()
        plt.figure(figsize=(8, 6))
        coef_top[::-1].plot(x="feature", y="coef_orig", kind="barh", legend=False)
        plt.axvline(0, linewidth=1)
        plt.xlabel("Coefficient (original feature space)")
        plt.title(f"Top {top_k} Linear Coefficients (signed)\n{model_name} - {target}")
        plt.tight_layout()
        out_coef_bar = os.path.join(results_dir, f"coefs_bar_{task_name}_{model_name}_{target}_top{top_k}.png")
        plt.savefig(out_coef_bar, dpi=300, bbox_inches="tight") 
        plt.close()
        print(f"Coefficient bar saved: {out_coef_bar}")
