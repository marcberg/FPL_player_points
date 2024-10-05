# TODO: create def that does plots
import mlflow
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)

import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from src.ml.evaluate.feature_importance import feature_importance
from src.ml.evaluate.permutation_importance import permutation_importance
from src.ml.evaluate.metrics import calculate_regression_metrics
from src.util.read_multiple_csv import import_csv_files


def plot_feature_importance(fi, algo_name, model_path, plot_number_of_features=20):

    fi_selected = fi.head(plot_number_of_features)
    if algo_name in ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet"]:
        col = 'p_value'
        label = 'P-value'
    elif algo_name in ["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost"]:
        col = 'feature_importance'
        label = 'Feature importance'

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 6})

    plt.barh(fi_selected['feature'], fi_selected[col])
    plt.xlabel(label)
    plt.ylabel('Feature')
    plt.title('{0} (Top {1} features)'.format(label, plot_number_of_features))

    plt.savefig(model_path + 'feature_importance.png')
    mlflow.log_artifact(model_path + 'feature_importance.png')
    plt.close()

def plot_permutation_importance(pi, model_path, plot_number_of_features=20):

    pi_selected = pi.head(plot_number_of_features)

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 6})

    plt.barh(pi_selected['feature'], pi_selected['permutation_importance'])
    plt.xlabel('Permutation importance')
    plt.ylabel('Feature')
    plt.title('{0} (Top {1} features)'.format('Permutation importance', plot_number_of_features))

    plt.savefig(model_path + 'permutation_importance.png')
    mlflow.log_artifact(model_path + 'permutation_importance.png')
    plt.close()

def plot_y_aganst_predicted(y, y_hat):

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_hat, color='blue', alpha=0.5)

    # Add labels and title
    plt.xlabel("Actual Values (y)")
    plt.ylabel("Predicted Values (y_hat)")
    plt.title("Scatter Plot of Actual vs Predicted Values")

def plot_y_against_residuals(y_train, y_train_hat, y_val, y_val_hat, model_path):
    # Calculate residuals
    residuals_train = [y - y_hat for y, y_hat in zip(y_train, y_train_hat)]
    residuals_val = [y - y_hat for y, y_hat in zip(y_val, y_val_hat)]

    # Create a figure with 2 subplots
    fig, axs = plt.subplots(2, 1, figsize=(5, 6))

    # Scatter plot for training data
    axs[0].scatter(y_train, residuals_train, color='blue', alpha=0.5)
    axs[0].axhline(y=0, color='red', linestyle='--')
    axs[0].set_xlabel("Actual Values (y_train)")
    axs[0].set_ylabel("Residuals (y_train - y_hat)")
    axs[0].set_title("Residual Plot for Training Data")
    axs[0].grid(True)

    # Scatter plot for validation data
    axs[1].scatter(y_val, residuals_val, color='green', alpha=0.5)
    axs[1].axhline(y=0, color='red', linestyle='--')
    axs[1].set_xlabel("Actual Values (y_val)")
    axs[1].set_ylabel("Residuals (y_val - y_hat)")
    axs[1].set_title("Residual Plot for Validation Data")
    axs[1].grid(True)

    # Adjust layout
    plt.tight_layout()
    
    plt.savefig(model_path + 'y_against_residuals.png')
    mlflow.log_artifact(model_path + 'y_against_residuals.png')
    plt.close()

def plot_residual_histograms(y_train, y_train_hat, y_val, y_val_hat, model_path):
    # Calculate residuals
    residuals_train = [y - y_hat for y, y_hat in zip(y_train, y_train_hat)]
    residuals_val = [y - y_hat for y, y_hat in zip(y_val, y_val_hat)]

    # Create a figure
    plt.figure(figsize=(10, 6))

    # Plot histogram for training residuals
    plt.hist(residuals_train, bins=100, color='blue', alpha=0.5, label='Training Residuals', density=True)

    # Plot histogram for validation residuals
    plt.hist(residuals_val, bins=100, color='green', alpha=0.5, label='Validation Residuals', density=True)

    # Add labels and title
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residuals: Training vs Validation")
    plt.axvline(0, color='red', linestyle='--', label='Zero Residual Line')  # Line at zero
    plt.legend()
    plt.grid()
    
    plt.savefig(model_path + 'residual_histogram.png')
    mlflow.log_artifact(model_path + 'residual_histogram.png')
    plt.close()

def evaluate_model(algo_name, save_to_mlflow=False):
    
    metrics = calculate_regression_metrics(algo_name)
    feature_importance(algo_name)
    permutation_importance(algo_name)

    if save_to_mlflow:
        mlflow.set_experiment('FPL Player points next game')

        timestamp = " " + str(pd.to_datetime('today'))
        with mlflow.start_run(run_name=algo_name + timestamp):
            mlflow.set_tag('algo', algo_name)
            
            model_path = 'artifacts/ml_results/{0}/'.format(algo_name)

            # Log model parameters
            model = joblib.load(os.path.join(model_path, 'model.pkl'))
            mlflow.log_params(model.named_steps['model'].get_params())

            # Log model parameters
            metrics = pd.read_excel(os.path.join(model_path, 'metrics.xlsx'))

            # log metrics
            def log_metric_mlflow(metric):
                mlflow.log_metric(metric + ' ' + 'train', metrics.loc[metrics.dataset == "train"][metric].iloc[0])
                try:
                    mlflow.log_metric(metric + ' ' + 'test', metrics.loc[metrics.dataset == "test"][metric].iloc[0])
                except:
                    pass
                mlflow.log_metric(metric + ' ' + 'val', metrics.loc[metrics.dataset == "validation"][metric].iloc[0])

            metrics_to_log = ['R2', 'MAE', 'MSE', 'RMSE']

            for m in metrics_to_log:
                log_metric_mlflow(m)

            # feature importance
            fi_path = os.path.join(model_path, 'feature_importance.xlsx')
            if os.path.isfile(fi_path):
                fi = pd.read_excel(fi_path)
                plot_feature_importance(fi, algo_name, model_path)


            # permutation importance
            pi_path = os.path.join(model_path, 'permutation_importance.xlsx')
            pi = pd.read_excel(pi_path)
            plot_permutation_importance(pi, model_path)


            # plot residuals
            ## import the data
            data_path = 'artifacts/split_data/'
            data = import_csv_files(data_path) 

            if 'X_test' in data:
                X_train = pd.concat([data['X_train'], data['X_test']], ignore_index=True)
                y_train = pd.concat([data['y_train'], data['y_test']], ignore_index=True)
            else:
                # Otherwise, use only the training data
                X_train = data['X_train']
                y_train = data['y_train']
                
            ## predict
            y_train_hat = model.predict(X_train)
            y_val_hat = model.predict(data['X_val'])

            ## plot y against residuals
            plot_y_against_residuals(y_train.iloc[:, 0].tolist(), 
                                     y_train_hat.flatten().tolist(), 
                                     data['y_val'].iloc[:, 0].tolist(), 
                                     y_val_hat.flatten().tolist(), 
                                     model_path)
            
            ## plot residuals histogram
            plot_residual_histograms(y_train.iloc[:, 0].tolist(), 
                                     y_train_hat.flatten().tolist(), 
                                     data['y_val'].iloc[:, 0].tolist(), 
                                     y_val_hat.flatten().tolist(), 
                                     model_path)

            # Save model
            mlflow.sklearn.log_model(model, 'model')

        mlflow.end_run()
    
    return metrics