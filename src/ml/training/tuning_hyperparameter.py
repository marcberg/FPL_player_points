import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline

from src.util.read_multiple_csv import import_csv_files

def sample_data(X, y, indices_train=None, indices_test=None, fraction=1.0, random_state=42):

    if indices_train is not None and indices_test is not None:

        test_fold = np.append(indices_train, indices_test)

        # Combine X, y, indices_train, indices_test for consistent sampling
        combined_data = pd.concat([X, y, pd.Series(test_fold, name="test_fold")], axis=1)
        
        # Sample based on the fraction
        sampled_data = combined_data.sample(frac=fraction, random_state=random_state).reset_index(drop=True)
        
        # Split the data back into X, y, indices_train, indices_test
        X_sampled = sampled_data.iloc[:, :-2]  # All columns except last two are features (X)
        y_sampled = sampled_data.iloc[:, -2].values   # Third-to-last column is target (y)
        test_fold_sampled = sampled_data.iloc[:, -1].values # Last column is indices_test
        
        return X_sampled, y_sampled, test_fold_sampled
    
    else:
        # Only sample X and y
        combined_data = pd.concat([X, y], axis=1)
        
        # Sample based on the fraction
        sampled_data = combined_data.sample(frac=fraction, random_state=random_state).reset_index(drop=True)
        
        # Split the data back into X and y
        X_sampled = sampled_data.iloc[:, :-1]  # All columns except last are features (X)
        y_sampled = sampled_data.iloc[:, -1].values   # Last column is target (y)
        
        return X_sampled, y_sampled

def select_param_from_grid(grid, algo_name):
    """
    Selects the best hyperparameters from the grid search results based on validation criteria and saves the grid results to an Excel file.

    Args:
        grid (GridSearchCV or RandomizedSearchCV): The grid search object after fitting.
        algo_name (str): Name of the algorithm to use for saving the grid search results.

    Returns:
        dict: A dictionary containing the selected best hyperparameters for the algorithm.
    """

    # Convert grid search results to a DataFrame, sort by rank and reset index
    cv_results = pd.DataFrame(grid.cv_results_).sort_values(by=["rank_test_score"], ascending=True).reset_index(drop=True)
    
    # Calculate validation criteria for selection
    # TODO: Can this be better? of course, but what?
    cv_results['div'] = cv_results.mean_train_score / cv_results.mean_test_score
    cv_results['ok'] = np.where((cv_results['div'] >= 0.8) & 
                                (cv_results['div'] <= 1.3), 1, 0)
    
    # Create a directory for saving results
    path = 'artifacts/ml_results/{0}/'.format(algo_name)
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Save the results to an Excel file
    cv_results.to_excel(os.path.join(path, 'grid_result.xlsx'), index=False)

    # Select the best parameters based on 'ok' criteria or use grid.best_params_
    if np.sum(cv_results['ok']) > 0:
        bp = cv_results.loc[cv_results['ok'] == 1]['params'].iloc[0]
    else:
        bp = grid.best_params_

    # Clean the selected parameters to remove prefix before '__'
    nbp = {}
    for k, v in bp.items():
        nbp[k[k.index('__')+2:]] = v

    joblib.dump(nbp, os.path.join(path, 'best_params.joblib'))

def grid_search(algorithms, 
                params,
                perform_crossvalidation=False,
                random_grid=True, 
                n_random_hyperparameters=10,
                fraction=1.0):
    """
    Performs a grid search or randomized grid search on multiple algorithms with predefined feature engineering pipelines.

    Args:
        feature_engineering_pipeline (Pipeline): A pre-processing pipeline to be applied before modeling.
        perform_crossvalidation (bool, optional): Whether to perform cross-validation or not. Defaults to False.
        random_grid (bool, optional): Whether to perform randomized search or full grid search. Defaults to True.
        n_random_hyperparameters (int, optional): Number of hyperparameters for randomized search. Defaults to 10.

    Returns:
        dict: A dictionary where the keys are algorithms and the values are the best parameters found for each model.
    """
    # load feature engineering pipeline
    feature_engineering_pipeline = joblib.load('artifacts/feature_engineered_data/feature_engineering_pipeline.joblib')

    # Load split data (train/test) from CSV files
    path = 'artifacts/split_data/'
    data = import_csv_files(path) 
    
    # Iterate through each model and perform grid search
    for i in range(len(list(algorithms))):
        print('\n- ' + list(algorithms.keys())[i] + '\n')

        algorithm = list(algorithms.values())[i]
        param = list(params.values())[i]

        # Create a pipeline with feature engineering and model
        pipeline = Pipeline([
            ('preprocessing', feature_engineering_pipeline),
            ('model', algorithm)
        ])

        # Perform cross-validation or use predefined train/test split
        num_cores = os.cpu_count()
        use_num_cores = (num_cores if num_cores <= 4 
                 else num_cores - 1 if num_cores <= 8 
                 else num_cores - 2)
        os.environ['LOKY_MAX_CPU_COUNT'] = str(use_num_cores)
        if perform_crossvalidation:
            X = data['X_train']
            y = data['y_train']

            if fraction < 1.0:
                X, y = sample_data(X, y, fraction=fraction)

            if random_grid:
                grid = RandomizedSearchCV(estimator=pipeline, 
                                          param_distributions=param, 
                                          n_iter=n_random_hyperparameters,
                                          cv=5, 
                                          n_jobs=use_num_cores, 
                                          scoring='neg_mean_squared_error', 
                                          error_score=np.nan, 
                                          return_train_score=True, 
                                          verbose=3)
            else:
                grid = GridSearchCV(estimator=pipeline, 
                                    param_grid=param, 
                                    cv=5, 
                                    n_jobs=use_num_cores, 
                                    scoring='neg_mean_squared_error', 
                                    error_score=np.nan, 
                                    return_train_score=True, 
                                    verbose=3)
                
        else:
            # Concatenate train and test data for predefined split
            X = pd.concat([data['X_train'], data['X_val']], ignore_index=True)
            y = pd.concat([data['y_train'], data['y_val']], ignore_index=True)

            # Define train/test indices for PredefinedSplit
            indices_train = np.full((data['X_train'].shape[0],), -1, dtype=int)
            indices_test = np.full((data['X_val'].shape[0],), 0, dtype=int)
            
            test_fold = np.append(indices_train, indices_test)
            if fraction < 1.0:
                X, y, test_fold = sample_data(X, y, indices_train, indices_test, fraction=fraction)

            ps = PredefinedSplit(test_fold)

            if random_grid:
                grid = RandomizedSearchCV(estimator=pipeline, 
                                          param_distributions=param, 
                                          n_iter=n_random_hyperparameters,
                                          cv=ps, 
                                          n_jobs=use_num_cores, 
                                          scoring='neg_mean_squared_error', 
                                          error_score=np.nan, 
                                          return_train_score=True, 
                                          verbose=3)
            else:
                grid = GridSearchCV(estimator=pipeline, 
                                    param_grid=param, 
                                    n_iter=n_random_hyperparameters, 
                                    cv=ps, 
                                    n_jobs=use_num_cores, 
                                    scoring='neg_mean_squared_error', 
                                    error_score=np.nan, 
                                    return_train_score=True, 
                                    verbose=3)

        # Fit the grid search
        grid.fit(X, y)

        # Select the best parameters and store in results
        select_param_from_grid(grid=grid, algo_name=list(algorithms.keys())[i])
