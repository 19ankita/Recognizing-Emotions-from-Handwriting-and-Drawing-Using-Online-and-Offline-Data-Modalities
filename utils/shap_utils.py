
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import seaborn as sns
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
    os.makedirs(results_dir, exist_ok=True)

    transformer, final_model, pca, scaler = _unwrap_pipeline(model)

    if transformer is not None:
        X_train_tf = transformer.transform(X_train)
        X_test_tf = transformer.transform(X_test)
    else:
        X_train_tf = X_train
        X_test_tf = X_test

    X_train_tf = _ensure_2d(X_train_tf)
    X_test_tf = _ensure_2d(X_test_tf)

    if _is_tree_model(final_model):
        explainer = shap.TreeExplainer(final_model)
        shap_raw = explainer.shap_values(X_test_tf)
    else:
        explainer = shap.Explainer(final_model, X_train_tf)
        shap_raw = explainer(X_test_tf)

    shap_raw = _to_numpy(shap_raw)

    if pca is not None:
        shap_matrix = shap_raw @ pca.components_
        feature_names = list(X_train.columns) if hasattr(X_train, "columns") else [f"f{i}" for i in range(shap_matrix.shape[1])]
    else:
        shap_matrix = shap_raw
        feature_names = list(X_train.columns) if hasattr(X_train, "columns") else [f"f{i}" for i in range(shap_matrix.shape[1])]

    # ---------------------------
    # Global SHAP metrics
    # ---------------------------
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    signed_mean_shap = shap_matrix.mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "signed_mean_shap": signed_mean_shap
    }).sort_values("mean_abs_shap", ascending=False)

    out_csv = os.path.join(results_dir, f"shap_importance_{task_name}_{model_name}_{target}.csv")
    importance_df.to_csv(out_csv, index=False)

    # ---------------------------
    # Per-task signed SHAP output
    # ---------------------------
    top_features = importance_df.head(top_k)["feature"].tolist()
    signed_df = importance_df.set_index("feature").loc[top_features].reset_index()

    signed_csv = os.path.join(results_dir, f"shap_signed_mean_{task_name}_{model_name}_{target}.csv")
    signed_df.to_csv(signed_csv, index=False)
    print(f"Signed mean SHAP saved: {signed_csv}")

    plt.figure(figsize=(8, 6))
    signed_df[::-1].plot(x="feature", y="signed_mean_shap", kind="barh", legend=False)
    plt.axvline(0, linewidth=1, color="black")
    plt.xlabel("Mean SHAP value (signed)")
    plt.title(f"Top {top_k} Signed SHAP\n{model_name} - {target}")
    plt.tight_layout()
    out_png_signed = os.path.join(results_dir, f"shap_bar_signed_{task_name}_{model_name}_{target}_top{top_k}.pdf")
    plt.savefig(out_png_signed, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP bar (signed) saved: {out_png_signed}")

    # ---------------------------
    # Append top-k to global shap_results.csv
    # ---------------------------
    shap_results_path = os.path.join(results_dir, "shap_results.csv")
    top = importance_df.head(top_k).copy()
    top["task"] = task_name
    top["model"] = model_name
    top["target"] = target

    if not os.path.exists(shap_results_path):
        top.to_csv(shap_results_path, index=False)
    else:
        top.to_csv(shap_results_path, mode="a", header=False, index=False)

    print(f"Appended top-{top_k} SHAP features to {shap_results_path}")

    # ---------------------------
    # Aggregated SHAP results → barplots + heatmaps
    # ---------------------------
    try:
        df = pd.read_csv(shap_results_path)

        # (A) Frequency barplot (unsigned)
        K = 5
        top_features_by_target = defaultdict(Counter)
        for (task, model, target_), group in df.groupby(["task", "model", "target"]):
            topk = group.sort_values("mean_abs_shap", ascending=False).head(K)
            top_features_by_target[target_].update(topk["feature"])

        records = []
        for target_, counter in top_features_by_target.items():
            for feature, count in counter.items():
                records.append({"target": target_, "feature": feature, "count": count})

        if records:
            agg_df = pd.DataFrame(records)
            plt.figure(figsize=(12,6))
            sns.barplot(data=agg_df, x="feature", y="count", hue="target")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel(f"Count of Times in Top-{K}")
            plt.title(f"Top-{K} SHAP Features Across Models & Tasks")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "shap_topk_frequency.pdf"), dpi=300)
            plt.close()

        # (B) Heatmap unsigned
        heatmap_df = (
            df.groupby(["target", "feature"])["mean_abs_shap"]
              .mean()
              .reset_index()
              .pivot(index="feature", columns="target", values="mean_abs_shap")
              .fillna(0)
        )
        if not heatmap_df.empty:
            plt.figure(figsize=(10,8))
            sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlGnBu")
            plt.title("Average |SHAP| Importance per Feature and Target")
            plt.ylabel("Feature")
            plt.xlabel("Target")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "shap_importance_heatmap.pdf"), dpi=300)
            plt.close()

        # (C) Heatmap signed
        signed_heatmap_df = (
            df.groupby(["target", "feature"])["signed_mean_shap"]
              .mean()
              .reset_index()
              .pivot(index="feature", columns="target", values="signed_mean_shap")
              .fillna(0)
        )
        if not signed_heatmap_df.empty:
            plt.figure(figsize=(10,8))
            sns.heatmap(signed_heatmap_df, annot=True, fmt=".2f", center=0, cmap="RdBu_r")
            plt.title("Average Signed SHAP Importance per Feature and Target")
            plt.ylabel("Feature")
            plt.xlabel("Target")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "shap_signed_importance_heatmap.pdf"), dpi=300)
            plt.close()

        print("Updated aggregated SHAP plots in results/:")
        print(" - shap_topk_frequency.pdf")
        print(" - shap_importance_heatmap.pdf")
        print(" - shap_signed_importance_heatmap.pdf")

    except Exception as e:
        print(f"Warning: failed to update aggregated SHAP plots → {e}")

