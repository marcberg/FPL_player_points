import argparse

from src.util.spark_session import get_spark_session
from src.data.preprocessing import preprocess_data
from src.ml.preprocess.train_test_validation_split import split_data
from src.ml.preprocess.feature_selection import feature_selection_by_test
from src.ml.preprocess.feature_engineering_pipeline import fe_pipeline
from src.ml.train import train_models
from src.ml.score import score_data

import warnings
warnings.filterwarnings("ignore")

import logging
# Set up basic logging
logging.basicConfig(level=logging.ERROR)

def main(args):
    # Check if any algo-specific parameters are specified
    args.run_all_algos = True  # Default value

    # List of all algorithm-specific flags
    algo_flags = [
        args.algo_linear_regression,
        args.algo_ridge,
        args.algo_lasso,
        args.algo_elasticnet,
        args.algo_decision_tree,
        args.algo_random_forest,
        args.algo_gradient_boosting,
        args.algo_xgboost,
        args.algo_lightgbm,
        args.algo_catboost,
        args.algo_svr,
        args.algo_keras
    ]

    # If any of the algorithm-specific flags are True, set run_all_algos to False
    if any(algo_flags):
        args.run_all_algos = False
    
    if args.part == 'preprocess':
        spark = get_spark_session()
        sc = spark.sparkContext
        sc.setLogLevel("ERROR")
        try:
            preprocess_data(spark, fraction=args.sample_fraction_train_data)
            spark.stop()
        except Exception as error:
            spark.stop()
            print("An error occurred:", error)

    elif args.part == 'train' or args.part == 'train_parts':

        if args.split_data or args.part == 'train':
            split_data(train_size=args.train_size, perform_crossvalidation=False)

        if args.feature_selection or args.part == 'train':
            selected_features = feature_selection_by_test(select_p_value=0.05)
            fe_pipeline(selected_features)

        # Determine the flags for running grids and final models based on conditions
        run_grids = args.run_grids if args.run_grids else not args.run_final_models
        run_final_models = args.run_final_models if args.run_final_models else not args.run_grids


        if args.train_models or args.part == 'train':
            # Set crossvalidation based on user input
            perform_crossvalidation = False
            if args.crossvalidation:
                perform_crossvalidation = True
            elif args.grid_search:
                perform_crossvalidation = False

            train_models(run_grids=run_grids,
                        fraction=args.sample_fraction_grid_data,
                        perform_crossvalidation=perform_crossvalidation,
                        random_grid=args.search_all_grid,
                        n_random_hyperparameters=args.n_random_hyperparameters,

                        run_final_models=run_final_models,

                        algo_linear_regression=args.algo_linear_regression or args.run_all_algos,
                        algo_ridge=args.algo_ridge or args.run_all_algos,
                        algo_lasso=args.algo_lasso or args.run_all_algos,
                        algo_elasticnet=args.algo_elasticnet or args.run_all_algos,
                        algo_decision_tree=args.algo_decision_tree or args.run_all_algos,
                        algo_random_forest=args.algo_random_forest or args.run_all_algos,
                        algo_gradient_boosting=args.algo_gradient_boosting or args.run_all_algos,
                        algo_xgboost=args.algo_xgboost or args.run_all_algos,
                        algo_lightgbm=args.algo_lightgbm or args.run_all_algos,
                        algo_catboost=args.algo_catboost or args.run_all_algos,
                        algo_svr=False, #args.algo_svr or args.run_all_algos,
                        algo_keras=args.algo_keras or args.run_all_algos,
                        
                        save_to_mlflow=args.save_to_mlflow)
            
    elif args.part == 'score':
        score_data()

    elif args.part == 'all':
        spark = get_spark_session()
        sc = spark.sparkContext
        sc.setLogLevel("ERROR")
        preprocess_data(spark, fraction=args.sample_fraction_train_data)
        spark.stop()

        split_data(train_size=args.train_size, perform_crossvalidation=False)

        selected_features = feature_selection_by_test(select_p_value=0.05)
        fe_pipeline(selected_features)

        # Same logic applied for running grids and final models
        run_grids = args.run_grids if args.run_grids else not args.run_final_models
        run_final_models = args.run_final_models if args.run_final_models else not args.run_grids

        # Set crossvalidation based on user input
        perform_crossvalidation = False
        if args.crossvalidation:
            perform_crossvalidation = True
        elif args.grid_search:
            perform_crossvalidation = False

        train_models(run_grids=run_grids,
                     fraction=args.sample_fraction_grid_data,
                     perform_crossvalidation=perform_crossvalidation,
                     random_grid=args.search_all_grid,
                     n_random_hyperparameters=args.n_random_hyperparameters,

                    run_final_models=run_final_models,

                     algo_linear_regression=args.algo_linear_regression or args.run_all_algos,
                     algo_ridge=args.algo_ridge or args.run_all_algos,
                     algo_lasso=args.algo_lasso or args.run_all_algos,
                     algo_elasticnet=args.algo_elasticnet or args.run_all_algos,
                     algo_decision_tree=args.algo_decision_tree or args.run_all_algos,
                     algo_random_forest=args.algo_random_forest or args.run_all_algos,
                     algo_gradient_boosting=args.algo_gradient_boosting or args.run_all_algos,
                     algo_xgboost=args.algo_xgboost or args.run_all_algos,
                     algo_lightgbm=args.algo_lightgbm or args.run_all_algos,
                     algo_catboost=args.algo_catboost or args.run_all_algos,
                     algo_svr=False, #args.algo_svr or args.run_all_algos,
                     algo_keras=args.algo_keras or args.run_all_algos,
                     
                     save_to_mlflow=args.save_to_mlflow)
        
        score_data()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parts or the entire ML pipeline")
    parser.add_argument('--part', type=str, choices=['preprocess', 'train', 'train_parts', 'score', 'all'], default='all',
                        help="Choose which part of the project to run (default: run all parts)")
    parser.add_argument('--split_data', action='store_true', help="Split data into train, test and val")
    parser.add_argument('--feature_selection', action='store_true', help="Select significant features to the model")
    parser.add_argument('--train_models', action='store_true', help="Train models")

    # Sample data
    parser.add_argument('--sample_fraction_train_data', type=float, default=1.0, help="Sample fraction saved train data (default: 1.0, must be between 0.01 and 1.0)")
    parser.add_argument('--sample_fraction_grid_data', type=float, default=1.0, help="Sample fraction data used in grid (default: 1.0, must be between 0.01 and 1.0)")

    # Algorithm-specific flags
    parser.add_argument('--algo_linear_regression', action='store_true', help="Use Linear Regression algorithm")
    parser.add_argument('--algo_ridge', action='store_true', help="Use Ridge Regression algorithm")
    parser.add_argument('--algo_lasso', action='store_true', help="Use Lasso Regression algorithm")
    parser.add_argument('--algo_elasticnet', action='store_true', help="Use ElasticNet algorithm")
    parser.add_argument('--algo_decision_tree', action='store_true', help="Use Decision Tree algorithm")
    parser.add_argument('--algo_random_forest', action='store_true', help="Use Random Forest algorithm")
    parser.add_argument('--algo_gradient_boosting', action='store_true', help="Use Gradient Boosting algorithm")
    parser.add_argument('--algo_xgboost', action='store_true', help="Use XGBoost algorithm")
    parser.add_argument('--algo_lightgbm', action='store_true', help="Use LightGBM algorithm")
    parser.add_argument('--algo_catboost', action='store_true', help="Use CatBoost algorithm")
    parser.add_argument('--algo_svr', action='store_true', help="Use Support Vector Regression algorithm")
    parser.add_argument('--algo_keras', action='store_true', help="Use Keras Regressor")

    # New flags for mlflow and grid search
    parser.add_argument('--search_all_grid', action='store_false', help="Use random grid search for hyperparameters (default: False)")
    parser.add_argument('--run_grids', action='store_true', help="Run grid search (default: True unless run_final_models specified)")
    parser.add_argument('--run_final_models', action='store_true', help="Run final models (default: True unless run_grids specified)")
    parser.add_argument('--save_to_mlflow', action='store_true', help="Save models to MLFlow (default: False)")
    
    # Cross-validation and grid search flags
    parser.add_argument('--crossvalidation', action='store_true', help="Perform cross-validation (default: False)")
    parser.add_argument('--grid_search', action='store_true', help="Perform grid search (default: False)")

    # Custom values with defaults for n_random_hyperparameters and train_size
    parser.add_argument('--n_random_hyperparameters', type=int, default=10, help="Number of random hyperparameters to sample (default: 10, minimum: 1)")
    parser.add_argument('--train_size', type=float, default=0.6, help="Train set size (default: 0.6, must be between 0.01 and 0.99)")

    args = parser.parse_args()

    # Validating arguments
    if not (0.01 <= args.train_size <= 0.99):
        raise ValueError("train_size must be between 0.01 and 0.99")
    
    if args.n_random_hyperparameters < 1:
        raise ValueError("n_random_hyperparameters must be at least 1")
    
    if not (0.1 <= args.sample_fraction_train_data <= 1.0):
        raise ValueError("sample_fraction_train_data must be between 0.1 and 1.0")
    
    if not (0.1 <= args.sample_fraction_grid_data <= 1.0):
        raise ValueError("sample_fraction_grid_data must be between 0.1 and 1.0")


    main(args)
