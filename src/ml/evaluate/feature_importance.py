import pandas as pd
import joblib
import os
import numpy as np
from scipy import stats

from src.util.read_multiple_csv import import_csv_files
from src.util.feature_names import get_features_name


def feature_importance(model_name):
    """
    Calculate and return feature importance or coefficient values for a given trained machine learning model. 
    This function handles both linear models and tree-based models, producing a dataframe of feature 
    importance or coefficients with corresponding p-values.

    Args:
        model_name (str): Name of the machine learning model (e.g., "Linear Regression", "Random Forest").

    Returns:
        DataFrame: A DataFrame with features and their importance (for tree-based models) 
                  or coefficients and p-values (for linear models).
    """
    
    print('\t-- Feature importance')
    # Retrieve the feature names used in the model
    feature_names = get_features_name()

    # Import the training and test data from CSV files
    data_path = 'artifacts/split_data/'
    data = import_csv_files(data_path)

    # If test data is available, concatenate it with the training data for evaluation
    if 'X_val' in data:
        X = pd.concat([data['X_train'], data['X_val']], ignore_index=True)
        y = pd.concat([data['y_train'], data['y_val']], ignore_index=True)
    else:
        # Otherwise, use only the training data
        X = data['X_train']
        y = data['y_train']

    # Load the pre-trained feature engineering pipeline
    feature_engineering_pipeline = joblib.load('artifacts/feature_engineered_data/feature_engineering_pipeline.joblib')
    
    # Apply the feature engineering pipeline to transform the training data
    X_train_transformed = feature_engineering_pipeline.fit_transform(X)

    # Load the trained model from the specified path
    model_path = 'artifacts/ml_results/{0}/'.format(model_name)
    model = joblib.load(os.path.join(model_path, 'model.pkl'))

    # Handle linear models (e.g., Linear Regression, Ridge, Lasso, ElasticNet)
    if model_name in ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet"]:
        # Extract the linear regression model from the pipeline
        lr_model = model.named_steps['model']

        # Check if the model includes an intercept term
        has_intercept = hasattr(model, 'intercept_') and np.isfinite(model.intercept_)

        # Include intercept in the feature matrix if applicable
        if has_intercept:
            X_new = np.column_stack((np.ones(n), X_train_transformed))
        else:
            X_new = X_train_transformed

        # Calculate coefficients using the pseudoinverse of the feature matrix
        coefficients = np.linalg.pinv(X_new).dot(y)

        # Predict the target variable using the calculated coefficients
        y_pred = X_new.dot(coefficients)
        residuals = y.to_numpy() - y_pred

        # Estimate the variance of residuals (sigma squared)
        residual_sum_of_squares = np.sum(residuals ** 2)
        degrees_of_freedom = len(y) - (X_new.shape[1] + 1)  # Adjust for intercept
        sigma_squared = residual_sum_of_squares / degrees_of_freedom
        
        # Calculate standard errors for the coefficients
        inverse_matrix = np.linalg.pinv(np.dot(X_new.T, X_new))  # Pseudoinverse for matrix inversion
        standard_errors = np.sqrt(sigma_squared * np.diag(inverse_matrix))

        # Calculate t-statistics and p-values for the coefficients
        t_stats = coefficients / standard_errors
        p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=degrees_of_freedom)) for t in t_stats]

        # Include 'Intercept' in the feature list if applicable
        if has_intercept:
            features = ['Intercept'] + feature_names
        else:
            features = feature_names

        # Create a DataFrame with feature names, coefficients, and p-values
        results = pd.DataFrame({
            'feature': features,
            'coefficient': lr_model.coef_[0].tolist(),
            'p_value': p_values[0].tolist()
        }).sort_values('p_value').reset_index(drop=True)

        results.to_excel('artifacts/ml_results/{0}/feature_importance.xlsx'.format(model_name), index=False)

    # Handle tree-based models (e.g., Decision Tree, Random Forest, XGBoost, etc.)
    elif model_name in ["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost"]:
        # Get feature importance scores from the model
        feature_importance = model.named_steps['model'].feature_importances_
        
        # Create a DataFrame with feature names and their importance scores
        results = pd.DataFrame({
            'feature': feature_names,
            'feature_importance': feature_importance,
        }).sort_values('feature_importance', ascending=False).reset_index(drop=True)

        results.to_excel('artifacts/ml_results/{0}/feature_importance.xlsx'.format(model_name), index=False)
    
