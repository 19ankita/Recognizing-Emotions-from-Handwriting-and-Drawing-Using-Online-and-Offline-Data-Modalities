import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def run_single_output_regression(merged_csv):
    """Train single-output linear regression on TOTAL DASS score."""
    
    # Load merged features + DASS labels for words task
    df = pd.read_csv(merged_csv)

    # Define X (all features except id/user and labels)
    X = df.drop(columns=["id", "user", "depression", "anxiety", "stress", "total"])
    y = df["total"] # single output = total DASS score

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Single-output Linear Regression (predicting TOTAL DASS)- MSE: {mse}, RMSE: {rmse}, R2: {r2}")
    
    return model
