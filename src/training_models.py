import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor


def evaluate_with_cv(model, X, y, model_name, do_cv=False, do_search=False, cv=5):
    """
    Run cross-validation and optional hyperparameter tuning.
    """
    best_model = model
    cv_mse, cv_std = None, None
    
    # Hyperparameter search
    if do_search:
        param_grid = None
        search_type = "grid"
        
        if "Ridge" in model_name:
            param_grid = {"model__alpha": [0.1, 1.0, 10.0, 50.0]}
        elif "Lasso" in model_name:
            param_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0]}
        elif "Elastic Net" in model_name:
            param_grid = {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.2, 0.5, 0.8]}
        elif "Random Forest" in model_name:
            param_grid = {"n_estimators": [100, 200, 500], "max_depth": [None, 10, 20]}
            search_type = "random"
        elif "Gradient Boosting" in model_name:
            param_grid = {
                "n_estimators": [100, 200, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7]
            }
            search_type = "random"
        
        if param_grid:
            if search_type == "grid":
                search = GridSearchCV(model, param_grid, cv=cv, 
                                      scoring="neg_mean_squared_error", n_jobs=-1)
            else:
                search = RandomizedSearchCV(model, param_grid, cv=cv, scoring="neg_mean_squared_error", 
                                            n_jobs=-1, n_iter=20, random_state=42)   
                
            search.fit(X, y)
            best_model = search.best_estimator_
            print(f"Best params for {model_name}: {search.best_params_}")
        
    # Cross-validation evaluation
    if do_cv:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(best_model, X, y, cv=kf, scoring="neg_mean_squared_error")  
        cv_mse = -np.mean(scores)
        cv_std = np.std(scores)
        print(f"Cross-validation MSE for {model_name}: {cv_mse:.3f} (+/- {cv_std:.3f})") 
    
    return best_model, cv_mse, cv_std


def run_model(merged_csv, task_name, model, model_name, target="total", do_cv=False, do_seach=False):
    """
    Train any regression model on TOTAL DASS score and return metrics.

    Args:
        merged_csv (str): Path to merged features + DASS file.
        task_name (str): Task name (e.g., words, cursive).
        model: sklearn regression model instance (e.g., LinearRegression()).
        model_name (str): Name of the model (for logging + results).
    """
    
    # Load merged features + DASS labels for words task
    df = pd.read_csv(merged_csv)
    # Define X (all features except id/user and labels)
    X = df.drop(columns=["id", "user", "depression", "anxiety", "stress", "total"], errors="ignore")
    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])
    
    if target == "total":
        y = df["total"] # single output = total DASS score
    else:
        raise ValueError("Invalid target for run_model. Use run_multioutput_model for multiple targets.")        
    
    # CV + hyperparameter tuning
    best_model, cv_mse, cv_std = evaluate_with_cv(model, X, y, model_name, do_cv, do_seach, cv_folds)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\nResults for task: {task_name} - {model_name} (Target = TOTAL DASS)")
    print(f"  MSE:  {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.3f}")
    
    return {
        "Task": task_name,
        "Model": model_name,
        "Target": "TOTAL DASS",
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "CV_MSE": cv_mse,
        "CV_STD": cv_std
    }


def run_multioutput_model(merged_csv, task_name, model, model_name, do_cv=False, do_search=False):
    """
    Train regression model (multi-output) on Depression, Anxiety, Stress simultaneously.
    """
    df = pd.read_csv(merged_csv)
    # Features
    X = df.drop(columns=["id", "user", "depression", "anxiety", "stress", "total"], errors="ignore")
    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])
    y = df[["depression", "stress", "anxiety"]]
    
    # Wrap model in MultiOutputRegressor
    multi_model = MultiOutputRegressor(model)
    
    # CV + hyperparam tuning
    best_model, cv_mse, cv_std = evaluate_with_cv(multi_model, X, y, model_name, do_cv, do_search, cv_folds)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model.fit(X_train, y_train)
    y_pred = multi_model.predict(X_test)
    
    results = []
    for i, col in enumerate(["depression", "stress", "anxiety"]):
        mse = mean_squared_error(y_test.iloc[:,i], y_pred[:,i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test.iloc[:,i], y_pred[:,i])
        
        print(f"\n Results for {task_name} - {model_name} (Target = {col})")
        print(f" MSE: {mse:.2f}")
        print(f" RMSE: {rmse:.2f}")
        print(f" R2: {r2:.3f}")
        
        
        results.append({
            "Task": task_name,
            "Model": model_name,
            "Target": col.capitalize(),
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "CV_MSE": cv_mse,
            "CV_STD": cv_std
        })
        
    return results    


def run_separate_subscale_models(merged_csv, task_name, model, model_name, do_cv=False, do_search=False):
    """
    Run 3 separate regressions:
        - One for Depression
        - One for Anxiety
        - One for Stress
    Returns metrics for each.
    """
    df = pd.read_csv(merged_csv)
    X = df.drop(["id", "user", "depression", "anxiety", "stress", "total"], errors="ignore")
    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])
    
    results = []
    for target in ["depression", "anxiety", "stress"]:
        if target not in df.columns:
            print(f"Column {target} not found in {merged_csv}, skipping...")
            continue
        y = df[target]
        
    # CV + hyperparam tuning
    best_model, cv_mse, cv_std = evaluate_with_cv(model, X, y, model_name, do_cv, do_search, cv_folds)
        
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )             
    
    # Fit model
    best_model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
       
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nResults for {task_name} - {model_name} (Target = {target})")
    print(f"  MSE:  {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.3f}")

    results.append({
        "Task": task_name,
        "Model": model_name,
        "Target": target.capitalize(),
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "CV_MSE": cv_mse,
        "CV_STD": cv_std
    })

    return results



            
                    