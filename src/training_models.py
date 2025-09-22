import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor


def run_model(merged_csv, task_name, model, model_name, target="total"):
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
    X = df.drop(columns=["id", "user", "depression", "anxiety", "stress", "total"])
    
    if target == "total":
        y = df["total"] # single output = total DASS score
    else:
        raise ValueError("Invalid target for run_model. Use run_multioutput_model for multiple targets.")        

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

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
        "R2": r2
    }


def run_multioutput_model(merged_csv, task_name, model, model_name):
    """
    Train regression model (multi-output) on Depression, Anxiety, Stress simultaneously.
    """
    df = pd.read_csv(merged_csv)
    
    # Features
    X = df.drop(columns=["id", "user", "depression", "anxiety", "stress", "total"])
    y = df[["depression", "stress", "anxiety"]]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Wrap model in MultiOutputRegressor
    multi_model = MultiOutputRegressor(model)
    
    multi_model.fit(X_train, y_train)
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
            "R2": r2
        })
        
    return results    


def run_separate_subscale_models(merged_csv, task_name, model, model_name):
    """
    Run 3 separate regressions:
        - One for Depression
        - One for Anxiety
        - One for Stress
    Returns metrics for each.
    """
    
    df = pd.read_csv(merged_csv)
    
    X = df.drop(["id", "user", "depression", "anxiety", "stress", "total"])
    
    results = []
    for target in ["depression", "anxiety", "stress"]:
        y = df[target]
        
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )             
    
    # Fit model
    model.fit(X_train, y_train)
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
    })

    return results