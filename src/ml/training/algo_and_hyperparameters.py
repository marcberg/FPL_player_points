from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from src.ml.training.deep_learning_model import keras_reg

def algorithms_params():

    algorithms = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "ElasticNet": ElasticNet(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "XGBoost": xgb.XGBRegressor(),
        "LightGBM": lgb.LGBMRegressor(),
        "CatBoost": CatBoostRegressor(),
        "SVR": SVR(),
        "KerasRegressor": keras_reg
    }

    params = {
        "Linear Regression": {
            'model__fit_intercept': [True, False],  
        },
        "Ridge Regression": {
            'model__alpha': [0.001, 0.01, 0.1, 1, 10],  # Regularization strength
            'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga'],
            'model__max_iter': [1000, 5000, 10000]
        },
        "Lasso Regression": {
            'model__alpha': [0.001, 0.01, 0.1, 1, 10],
            'model__max_iter': [1000, 5000, 10000]
        },
        "ElasticNet": {
            'model__alpha': [0.001, 0.01, 0.1, 1],
            'model__l1_ratio': [0.1, 0.5, 0.7, 1],
            'model__max_iter': [1000, 5000, 10000]
        },
        "Decision Tree": {
            'model__criterion': ['squared_error', 'friedman_mse'],  # Regressors use MSE
            'model__max_depth': [None, 2, 3, 4, 5, 7, 9, 12, 15],
            'model__min_samples_leaf': [1, 2, 5, 10, 20],
            'model__min_samples_split': [2, 5, 10]
        },
        "Random Forest": {
            'model__bootstrap': [True, False],
            'model__max_features': ['sqrt', 'log2', None],
            'model__max_depth': [None, 2, 3, 5, 7, 9, 12, 15],
            'model__min_samples_leaf': [1, 2, 5, 10, 20],
            'model__min_samples_split': [2, 5, 10],
            'model__n_estimators': [100, 200, 500, 1000]
        },
        "Gradient Boosting": {
            'model__loss': ['squared_error', 'huber'],
            'model__learning_rate': [0.001, 0.01, 0.1, 0.2],
            'model__min_samples_leaf': [1, 2, 5, 10],
            'model__max_depth': [3, 4, 5, 7, 9],
            'model__n_estimators': [100, 200, 500]
        },
        "XGBoost": {
            'model__max_depth': [3, 5, 7, 10],
            'model__learning_rate': [0.001, 0.01, 0.05, 0.1],
            'model__n_estimators': [100, 200, 500],
            'model__min_child_weight': [1, 3, 5],
            'model__gamma': [0, 0.1, 0.5],
            'model__reg_lambda': [0, 0.1, 1, 10]
        },
        "LightGBM": {
            'model__max_depth': [-1, 3, 5, 7],
            'model__learning_rate': [0.001, 0.01, 0.05, 0.1],
            'model__num_leaves': [31, 63, 127],
            'model__min_child_samples': [10, 20, 30],
            'model__subsample': [0.8, 1.0],
            'model__colsample_bytree': [0.8, 1.0],
            'model__reg_alpha': [0.0, 0.1],
            'model__reg_lambda': [0.0, 0.1, 1],
            'model__n_estimators': [100, 200, 500],
            'model__verbosity': [-1]
        },
        "CatBoost": {
            'model__depth': [4, 6, 10],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__iterations': [100, 200, 500],
            'model__l2_leaf_reg': [1, 3, 5, 7, 9],  # L2 regularization
            'model__silent': [True]
        },
        "SVR": {
            'model__kernel': ['linear', 'poly', 'rbf'],
            'model__C': [0.1, 1, 10, 100],
            'model__epsilon': [0.01, 0.1, 0.2],
            'model__degree': [2, 3, 4]  # Only relevant for 'poly' kernel
        },
        "KerasRegressor": {
            'model__model__units_list': [[16], [32], [64], [128], [16, 32], [32, 64], [64, 128], [16, 32, 64], [32, 64, 128]], 
            'model__model__activation': ['relu', 'tanh'],
            'model__model__optimizer': ['adam', 'sgd'],
            'model__model__learning_rate': [0.0001, 0.001, 0.005],
            'model__epochs': [50, 100, 200],
            'model__batch_size': [10, 20, 40]
        }
    }

    return algorithms, params
