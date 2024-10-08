import pyspark.sql.functions as f
import numpy as np
import pandas as pd

from src.data.player_data import player_data
from src.data.team_data import team_data
from src.util.save_csv import save_multiple_csv


def preprocess_data(spark, fraction=1.0):
    """
    Preprocesses the data by joining player and team data, filling NaN values in specified columns, 
    and preparing data for machine learning tasks such as training and scoring. The processed data is saved as CSV files.

    Args:
        spark (SparkSession): The Spark session to use for executing SQL queries.
        fraction (float, optional): Fraction of the data to sample. Defaults to 1.0.

    Returns:
        None
    """
    print('Preprocessing data')
    player_calculations = player_data(spark)
    team_calculations = team_data(spark)

    data = player_calculations \
        .join(team_calculations, on=["team", "opponent", "kickoff_time", "season_start_year"], how='inner')

    # Replace NaN with 0 in specified columns
    data = data.distinct().sample(fraction=fraction) # .fillna(0, subset=columns_to_fill_with_zero)

    # train
    df_train = data.filter((f.col("data") == "train") & f.col("next_game_home").isNotNull()).toPandas()

    target_column = 'target'  # Replace with your target column
    X = df_train.drop(columns=[target_column, "kickoff_time", "player_id", "playername", "next_kickoff_time", "data"])
    y = df_train[target_column]

    # score
    score = data.filter((f.col("data") == "score")).toPandas()

    # cols
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    cols_data = {
        'target': [target_column], 
        'numeric': numeric_cols.to_list(),
        'categorical': categorical_cols.to_list()
    }

    # Convert the dictionary to a DataFrame
    cols_df = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in cols_data.items()]))

    # Melt the DataFrame to get two columns: "col" and "type"
    cols_df_melted = cols_df.melt(var_name='type', value_name='col').dropna()

    save_multiple_csv(obj_dict={'X': X, 
                                'y': y,
                                'score': score,
                                'cols_df': cols_df_melted
                                },
                      path="artifacts/data/"
                    )

    spark.stop()
