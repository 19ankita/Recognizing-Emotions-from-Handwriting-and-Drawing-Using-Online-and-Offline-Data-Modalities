import pandas as pd
import numpy as np

from utils.shap_utils import run_shap_analysis
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor


def evaluate_with_cv(model, X, y, model_name, do_cv=False, do_search=False, cv=5):

    """
    Evaluate a regression model using optional cross-validation and optional
    hyperparameter tuning (GridSearchCV / RandomizedSearchCV).

    Input
    -----
    model : sklearn estimator
        Base regression model or a wrapped model (e.g., MultiOutputRegressor).
    X : array-like of shape (n_samples, n_features)
        Training feature matrix.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Training target(s).
    model_name : str
        Human-readable model name used to select a predefined search space.
    do_cv : bool, default=False
        If True, run K-Fold CV and report mean/std of MSE.
    do_search : bool, default=False
        If True, perform hyperparameter tuning (grid or random depending on model).
    cv : int, default=5
        Number of folds for CV and/or hyperparameter search.

    Output
    ------
    best_model : sklearn estimator
        The tuned estimator if `do_search=True`, otherwise the original model.
    cv_mse : float or None
        Mean CV MSE (from search best score or from cross_val_score).
        None if neither CV nor search is executed.
    cv_std : float or None
        Standard deviation of CV scores. None if neither CV nor search is executed.

    Notes
    -----
    - For MultiOutputRegressor, parameter names are automatically prefixed with
      'estimator__' during hyperparameter tuning.
    - Hyperparameter search uses negative MSE scoring internally.
    """
    best_model = model
    cv_mse, cv_std = None, None
    
    # Hyperparameter search
    if do_search:
        param_grid = None
        search_type = "grid"
        
        if "Ridge" in model_name:
            param_grid = {"model__alpha": np.logspace(-3, 3, 20)}
            
        elif "Lasso" in model_name:
            param_grid = {"model__alpha": np.logspace(-4, 1, 20)}
            
        elif "Elastic Net" in model_name:
            param_grid = {
                "model__alpha": np.logspace(-2, 2, 15),
                "model__l1_ratio": [0.3, 0.5, 0.7]
            }
            
        elif "Random Forest" in model_name:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_leaf": [1, 3, 5]
            }
            search_type = "random"
            
        elif "Gradient Boosting" in model_name:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.7, 0.9, 1.0],
                "min_samples_leaf": [1, 3, 5]
            }
            search_type = "random"
        
        
        if param_grid and isinstance(model, MultiOutputRegressor):
            param_grid = {f"estimator__{k}": v for k,v in param_grid.items()}
            
        if param_grid:
            if search_type == "grid":
                search = GridSearchCV(model,
                                      param_grid, 
                                      cv=cv, 
                                      scoring="neg_mean_squared_error", 
                                      n_jobs=-1)
            else:
                search = RandomizedSearchCV(model, 
                                            param_grid, 
                                            cv=cv,
                                            scoring="neg_mean_squared_error", 
                                            n_jobs=-1,
                                            n_iter=15, 
                                            random_state=42)   
                
            search.fit(X, y)
            best_model = search.best_estimator_
            
            cv_mse = -search.best_score_
            cv_std = np.std(search.cv_results_["mean_test_score"])
            
            print(f"Best params for {model_name}: {search.best_params_}")
            print(f"CV MSE: {cv_mse:.3f}")

            return best_model, cv_mse, cv_std
        
    # Cross-validation evaluation
    if do_cv:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(best_model, X, y, cv=kf, scoring="neg_mean_squared_error")  
        cv_mse = -np.mean(scores)
        cv_std = np.std(scores)
        print(f"Cross-validation MSE for {model_name}: {cv_mse:.3f} (+/- {cv_std:.3f})") 
    
    return best_model, cv_mse, cv_std


def run_model(merged_csv, task_name, model, mode, model_name, target="total", do_cv=False, do_search=False, cv_folds=5, do_shap=False):

    """
    Train and evaluate a single-output regression model for TOTAL DASS score.

    Input
    -----
    merged_csv : str
        Path to a CSV containing extracted features and DASS labels.
    task_name : str
        Name of the handwriting task (e.g., "house", "words") for logging.
    model : sklearn estimator
        Regression model instance (e.g., LinearRegression(), RandomForestRegressor()).
    mode : str
        Experiment mode label (kept for bookkeeping; not used internally here).
    model_name : str
        Model name used for logging and selecting hyperparameter search space.
    target : str, default="total"
        Only "total" is supported in this function.
    do_cv : bool, default=False
        If True, compute K-Fold CV MSE on the training split.
    do_search : bool, default=False
        If True, run hyperparameter tuning on the training split.
    cv_folds : int, default=5
        Number of folds for CV and/or hyperparameter search.
    do_shap : bool, default=False
        If True, run SHAP explanations after training.

    Output
    ------
    results : dict
        Dictionary containing evaluation metrics on the held-out test split:
        {Task, Model, mode, Target, MSE, RMSE, R2, CV_MSE, CV_STD}.

    Raises
    ------
    ValueError
        If target is not "total".
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

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # CV + hyperparameter tuning
    best_model, cv_mse, cv_std = evaluate_with_cv(model, X_train, y_train, model_name, do_cv, do_search, cv_folds)
    
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
    
    results = {
        "Task": task_name,
        "Model": model_name,
        "mode": "total", 
        "Target": "TOTAL DASS",
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "CV_MSE": float(cv_mse) if cv_mse is not None else None,
        "CV_STD": float(cv_std) if cv_std is not None else None
    }
    
    # Run SHAP after evaluation
    if do_shap:
        run_shap_analysis(best_model, X_train, X_test, task_name, model_name, "TOTAL_DASS")
    
    return results


def run_multioutput_model(merged_csv, task_name, model, mode, model_name, do_cv=False, do_search=False, cv_folds=5, do_shap=False):

    """
    Train and evaluate a multi-output regression model for the three DASS subscales
    (Depression, Stress, Anxiety) simultaneously using MultiOutputRegressor.

    Input
    -----
    merged_csv : str
        Path to a CSV containing extracted features and subscale labels.
    task_name : str
        Task name for logging.
    model : sklearn estimator
        Base regressor that will be wrapped by MultiOutputRegressor.
    mode : str
        Experiment mode label (for bookkeeping).
    model_name : str
        Model name used for logging and selecting hyperparameter search space.
    do_cv : bool, default=False
        If True, run CV on the multi-output model using negative MSE scoring.
    do_search : bool, default=False
        If True, tune hyperparameters on the multi-output model.
    cv_folds : int, default=5
        Number of folds for CV and/or hyperparameter search.
    do_shap : bool, default=False
        If True, compute SHAP for each fitted sub-estimator separately.

    Output
    ------
    results : list[dict]
        One dictionary per target in ["Depression", "Stress", "Anxiety"], each with:
        {Task, Model, mode, Target, MSE, RMSE, R2, CV_MSE, CV_STD}.
    """
    df = pd.read_csv(merged_csv)
    # Features
    X = df.drop(columns=["id", "user", "depression", "anxiety", "stress", "total"], errors="ignore")
    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])
    
    y = df[["depression", "stress", "anxiety"]]
    
    # Wrap model in MultiOutputRegressor
    multi_model = MultiOutputRegressor(model)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # CV + hyperparam tuning
    best_model, cv_mse, cv_std = evaluate_with_cv(multi_model, X_train, y_train, model_name, do_cv, do_search, cv_folds)
    
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    results = []
    for i, col in enumerate(["depression", "stress", "anxiety"]):
        mse = mean_squared_error(y_test.iloc[:,i], y_pred[:,i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test.iloc[:,i], y_pred[:,i])
        
        print(f"\n Results for {task_name} - {model_name} (Target = {col})")
        print(f" MSE: {mse:.2f}")
        print(f" RMSE: {rmse:.2f}")
        print(f" R2: {r2:.3f}")
        
        
        if do_shap:
            # Explain the *i-th* fitted estimator directly (can be Pipeline or a bare model)
            base_estimator_i = best_model.estimators_[i]
            run_shap_analysis(base_estimator_i, X_train, X_test, task_name, model_name, target=col)
        
        results.append({
            "Task": task_name,
            "Model": model_name,
            "mode": "multi-output", 
            "Target": col.capitalize(),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "R2": float(r2),
            "CV_MSE": float(cv_mse) if cv_mse is not None else None,
            "CV_STD": float(cv_std) if cv_std is not None else None
        })
        
        
        
    return results    


def run_separate_subscale_models(merged_csv, task_name, model, mode, model_name, do_cv=False, do_search=False, cv_folds=5, do_shap=False):

    """
    Train and evaluate three separate single-output regression models:
    one each for Depression, Anxiety, and Stress.

    Input
    -----
    merged_csv : str
        Path to a CSV containing extracted features and subscale labels.
    task_name : str
        Task name for logging.
    model : sklearn estimator
        Regression model instance (re-fit independently per target).
    mode : str
        Experiment mode label (for bookkeeping).
    model_name : str
        Model name used for logging and selecting hyperparameter search space.
    do_cv : bool, default=False
        If True, compute K-Fold CV MSE on the training split for each target.
    do_search : bool, default=False
        If True, run hyperparameter tuning separately for each target.
    cv_folds : int, default=5
        Number of folds for CV and/or hyperparameter search.
    do_shap : bool, default=False
        If True, run SHAP explanations for each trained target model.

    Output
    ------
    results : list[dict]
        One dictionary per available target in ["Depression", "Anxiety", "Stress"], each with:
        {Task, Model, mode, Target, MSE, RMSE, R2, CV_MSE, CV_STD}.

    Notes
    -----
    If a target column is missing from the CSV, that target is skipped.
    """
    df = pd.read_csv(merged_csv)
    X = df.drop(columns=["id", "user", "depression", "anxiety", "stress", "total"], errors="ignore")
    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])

    results = []
    for target in ["depression", "anxiety", "stress"]:
        if target not in df.columns:
            print(f"Column {target} not found in {merged_csv}, skipping...")
            continue
        y = df[target]
            
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )             
        
        # CV + hyperparam tuning
        best_model, cv_mse, cv_std = evaluate_with_cv(model, X_train, y_train, model_name, do_cv, do_search, cv_folds)
        
        # Fit model
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nResults for {task_name} - {model_name} (Target = {target})")
        print(f"  MSE:  {mse:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²:   {r2:.3f}")
        
        # Run SHAP after evaluation
        if do_shap:
            run_shap_analysis(best_model, X_train, X_test, task_name, model_name, target=target)

        results.append({
            "Task": task_name,
            "Model": model_name,
            "mode": "subscales", 
            "Target": target.capitalize(),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "R2": float(r2),
            "CV_MSE": float(cv_mse) if cv_mse is not None else None,
            "CV_STD": float(cv_std) if cv_std is not None else None
        })

    return results 
    
     