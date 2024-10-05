import pandas as pd
import joblib
import os
import sklearn.inspection
from src.util.feature_names import get_features_name
from src.util.read_multiple_csv import import_csv_files

def permutation_importance(model_name):
    """
    Calculate permutation importance for a given trained model and save the results to an Excel file.
    The function loads preprocessed data and a trained model, then applies the permutation importance
    method to assess the impact of each feature.

    Args:
        model_name (str): Name of the machine learning model (e.g., "SVR", "Random Forest").

    Returns:
        None: The permutation importance results are saved in an Excel file.
    """

    # Retrieve the feature names used in the model
    feature_names = get_features_name()

    # Load the training and test data from CSV files
    data_path = 'artifacts/split_data/'
    data = import_csv_files(data_path)

    # If test data exists, concatenate it with the training data for model evaluation
    if 'X_test' in data:
        X = pd.concat([data['X_train'], data['X_test']], ignore_index=True)
        y = pd.concat([data['y_train'], data['y_test']], ignore_index=True)
    else:
        # Otherwise, use only the training data
        X = data['X_train']
        y = data['y_train']

    # Load the feature engineering pipeline and transform the training data
    feature_engineering_pipeline = joblib.load('artifacts/feature_engineered_data/feature_engineering_pipeline.joblib')
    X_train_transformed = feature_engineering_pipeline.fit_transform(X)
    X_val_transformed = feature_engineering_pipeline.fit_transform(data['X_val'])

    # Load the trained model from the specified path
    model_path = 'artifacts/ml_results/{0}/'.format(model_name)
    model = joblib.load(os.path.join(model_path, 'model.pkl'))

    # Fit the model on the transformed training data
    model[-1].fit(X_train_transformed, y.iloc[0:].values.flatten())

    # Perform permutation importance on the validation set
    # This assesses the drop in model performance when each feature is randomly shuffled
    perm_importance = sklearn.inspection.permutation_importance(model[-1], X_val_transformed, data['y_val'])

    # Extract the average importance for each feature
    feature_importances = perm_importance.importances_mean

    # Create a DataFrame to store feature names and their corresponding permutation importance
    results = pd.DataFrame({
        'feature': feature_names,
        'permutation_importance': feature_importances.tolist(),
    }).sort_values('permutation_importance', ascending=False).reset_index(drop=True)

    # Save the permutation importance results to an Excel file
    results.to_excel('artifacts/ml_results/{0}/permutation_importance.xlsx'.format(model_name), index=False)
