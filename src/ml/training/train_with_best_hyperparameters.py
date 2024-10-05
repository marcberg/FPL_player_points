import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline

from src.util.read_multiple_csv import import_csv_files

def train_with_best_hyperparameters(algo_name, algorithm):

    # load feature engineering pipeline
    feature_engineering_pipeline = joblib.load('artifacts/feature_engineered_data/feature_engineering_pipeline.joblib')

    # setup pipeline
    model_path = 'artifacts/ml_results/{0}/'.format(algo_name)
    best_params = joblib.load(os.path.join(model_path, 'best_params.joblib'))

    model_pipeline = Pipeline([
        ('preprocessing', feature_engineering_pipeline),
        ('model', algorithm.set_params(**best_params))
    ])

    # import the data
    data_path = 'artifacts/split_data/'
    data = import_csv_files(data_path) 
    
    if 'X_test' in data:
        X = pd.concat([data['X_train'], data['X_test']], ignore_index=True)
        y = pd.concat([data['y_train'], data['y_test']], ignore_index=True)
    else:
        X = data['X_train']
        y = data['y_train']

    # training
    model_pipeline.fit(X, y.iloc[0:].values.flatten())

    # Save the pipeline to a file
    joblib.dump(model_pipeline, os.path.join(model_path, 'model.pkl'))