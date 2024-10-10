import pandas as pd

from src.ml.training.tuning_hyperparameter import grid_search
from src.ml.training.algo_and_hyperparameters import algorithms_params
from src.ml.training.train_with_best_hyperparameters import train_with_best_hyperparameters
from src.ml.evaluate.evaluate import evaluate_model


def train_models(run_grids=True,
                 fraction=1.0,
                 perform_crossvalidation=True,
                 random_grid=True, 
                 n_random_hyperparameters=10,

                 run_final_models=True, 
                 
                 algo_linear_regression=None,
                 algo_ridge=None,
                 algo_lasso=None,
                 algo_elasticnet=None,
                 algo_decision_tree=None,
                 algo_random_forest=None,
                 algo_gradient_boosting=None,
                 algo_xgboost=None,
                 algo_lightgbm=None,
                 algo_catboost=None,
                 algo_svr=None,
                 algo_keras=None,

                 save_to_mlflow=False
                 ):
    

    algorithms, params = algorithms_params()

    # Mapping the booleans to corresponding dictionary keys
    condition_map = {
        'Linear Regression': algo_linear_regression,
        "Ridge Regression": algo_ridge,
        "Lasso Regression": algo_lasso,
        "ElasticNet": algo_elasticnet,
        'Decision Tree': algo_decision_tree,
        "Random Forest": algo_random_forest,
        "Gradient Boosting": algo_gradient_boosting,
        "XGBoost": algo_xgboost,
        "LightGBM": algo_lightgbm,
        "CatBoost": algo_catboost,
        "SVR": algo_svr,
        "KerasRegressor": algo_keras
    }

    selected_algorithms = {name: algo for name, algo in algorithms.items() if condition_map.get(name, False)}
    selected_params = {name: param for name, param in params.items() if condition_map.get(name, False)}

    if run_grids:
        print('Performing hyperparameter tuning')
        grid_search(algorithms=selected_algorithms, 
                    params=selected_params,
                    perform_crossvalidation=perform_crossvalidation,
                    random_grid=random_grid, 
                    n_random_hyperparameters=n_random_hyperparameters,
                    fraction=fraction
                    )


    model_performance = {}
    timestamp = " " + str(pd.to_datetime('today'))
    if run_final_models:
        print('Train each algorithm with their best hyperparameters')
        for algo in range(len(list(selected_algorithms))):
            algo_name = list(selected_algorithms.keys())[algo]
            algorithm = list(selected_algorithms.values())[algo]
            print('- ' + algo_name)
                
            train_with_best_hyperparameters(algo_name, algorithm)

            print('-- Evaluating the model')
            metrics = evaluate_model(algo_name, timestamp, save_to_mlflow=save_to_mlflow)

            # save the models metric to decide which performs the best that we can score with
            metric = metrics.loc[metrics.dataset == "validation"]['MSE'].iloc[0]
            model_performance[algo_name] = metric
        
        print(model_performance)
        # Select the name with the lowest value
        lowest_model = min(model_performance, key=model_performance.get)
        lowest_value = model_performance[lowest_model]

        # Save to a CSV file
        df = pd.DataFrame({'best_algo': [lowest_model]})
        df.to_csv('artifacts/ml_results/best_performing_algorithm.csv', index=False)

        # Print the result
        print("\n")
        print(f"The model with the best performance is '{lowest_model}' with a MSE of {lowest_value}.")

