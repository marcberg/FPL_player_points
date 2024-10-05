import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.util.read_multiple_csv import import_csv_files

def calculate_regression_metrics(model_name):
    # Load the training and test data from CSV files
    data_path = 'artifacts/split_data/'
    data = import_csv_files(data_path)

    # Load the trained model from the specified path
    model_path = 'artifacts/ml_results/{0}/'.format(model_name)
    model = joblib.load(os.path.join(model_path, 'model.pkl'))

    # Initialize an empty DataFrame to store results
    metrics_df = pd.DataFrame(columns=['dataset', 'R2', 'MAE', 'MSE', 'RMSE'])
    
    # Define a function to calculate and append metrics
    def append_metrics(dataset, X, y):
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        
        # Append the results to the DataFrame
        metrics_df.loc[len(metrics_df)] = [dataset, r2, mae, mse, rmse]
    
    # Calculate metrics for train, validation, and test sets
    append_metrics('train', data['X_train'], data['y_train'])
    try:
        append_metrics('test', data['X_test'], data['y_test'])
    except:
        pass
    append_metrics('validation', data['X_val'], data['y_val'])
    
    metrics_df.to_excel('artifacts/ml_results/{0}/metrics.xlsx'.format(model_name), index=False)

    return metrics_df