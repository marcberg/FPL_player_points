# FPL_player_points

This ML project predicts the number of points each player will get the next round.

In this project:
- We get raw data from an API (I saved historical data (missing some) so I add new data to the historical).
- Preprocess it to create features and target using PySpark.
- Split the data in to train, test and validation. Test is not created if you perform crossvalidation.
- Perform significance test on features against target. Significant features are then put in my feature engineering pipeline. 
- Hyperparameter tuning is performed on selected algorithms where you can choose random grid search or search whole grid. You can also select if you want to perform crossvalidation or not. 
- Train each selected algorithms with their best performing hyperparameters and evaluate which performs the best. You choose to save result to mlflow. 
- Score data with the best performing model. 

You can choose which parts to run or run all, and there are other options you can choose like for example:
- Sample data when creating the training, test and validation.
- Sample data for the grid search.
- Decide fraction on training-data (the rest goes in to test and validation).
- Number of folds when performing crossvalidation.
- Number of iterations when doing random grid search.
- If you want to save model performance to mlflow or not.
- You can choose if you want to run all algorithms that are listed in the project or select which ones you want to use.


## Dependencies 

- Anaconda 
- PySpark

## Enviroment & libraries

Setup enviroment and install packages

```
conda create --name venv_fpl_player_points python=3.11.5 -y

conda activate venv_fpl_player_points

conda install ipykernel -y

pip install -r requirements.txt

```


## Folder structure

```

FPL_player_points/
│
├─ analysis/
│
├─ artifacts/
│    ├─ data/
│    ├─ feature_engineered_data/
│    ├─ ml_results/
│    └─ split_data/
│
├─ src/
│    ├─ data/
│    ├─ ml/
│    └─ util/
│
└─ main.py

```

### analysis
TODO!


### artifacts

data:
- csv files with preprocessed training-data (X and y), score-data and column-type.

feature_engineered_data:
- result from significance test and the created feature engineering pipeline.

ml_results:
- each algorithm have their own folder with latest result from grid and evaluation from their best performing model.
- also stores the name of the best performing algorithm and the final predictions.

split_data:
- train, test and val. 


### src

data:
- functions used to read and preprocess the data into training and score. 

ml:
- preprocess/: significance test and feature engineering pipeline.
- training/: all the functions used for grid search and training the final model.
- evaluate/: functions used to evaluate the final models. 
- train.py: a training pipeline used in main.py.
- score.py: function used for scoring pipeline used in main.py.

util:
- common functions used all over the project.


## Recommendations

- you must perform grid search for an algorithm before training the final model. 
- if you change anything to the data (features, logic, new data) you should perform a new grid search before training the final model. Else the previous grid search in no longer up to date.
- some algorithms takes a long time to run even if sample the data, so start by doing the easier like linear regression and decision tree. 


## Examples on how to run main.py

run the whole project with default values
```
python main.py
```

run the just the preprocess-, train- or score-parts
```
python main.py --part preprocess
python main.py --part train
python main.py --part score
```

run specific parts of the training
```
python main.py --part train_parts --split_data
python main.py --part train_parts --feature_selection
python main.py --part train_parts --train_models
```

run specific algorithms while performing only grid search on 10% of the training-data. Using random grid search with 10 random hyperparameters.
```
python main.py --part train_parts --train_models --run_grids --random_grid --sample_fraction_grid_data 0.1 --n_random_hyperparameters 10 --algo_linear_regression --algo_decision_tree
```

run the final models with specified algorithms and saving the result to mlflow
```
python main.py --part train_parts --train_models --algo_linear_regression --algo_decision_tree --save_to_mlflow
```


## mlflow

open a new terminal and write
```

conda activate venv_fpl_player_points

mlflow server

```

click the link that appears and it should open in your webbrowser. Don't close the terminal while using mlflow.


## Todo-list

- Add API and raw-data in to this project (currently in another project which I read from). Then update code and folder-structure.
- Add deep learning-model
- Add/update docstrings
- Add/update comments in code
- Remove warnings (either by turn them off or by fixing the code)
- Add more print() so you know whats running. And make the existing better.
- fixes with training-data after analysis:
    - leave some features with NaN so that the feature engineering pipeline can handle them (for example goals_conceeded = 0, really well if you play, misleading if you don't play.)
