
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

def only_include_features(feature_list, included_features):
    """
    Filters the input feature list to only include features that are present in the included_features list.

    Args:
        feature_list (list): A list of features to be filtered.
        included_features (list): A list of features to be retained in the output.

    Returns:
        list: A list of features that are both in the feature_list and included_features.
    """
    cols = [feature for feature in feature_list if feature in included_features]
    return cols

def fe_pipeline(included_features):
    """
    Creates and returns a preprocessing pipeline for feature engineering based on the input included features.
    This pipeline handles categorical and numerical features separately, applies PCA, and scales data.

    Args:
        included_features (list): A list of features that should be included in the pipeline processing.

    Returns:
        ColumnTransformer: A ColumnTransformer object that defines a pipeline for preprocessing the data.
    """
    # Define categorical preprocessing pipeline
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing_data", add_indicator=True)),
            ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    # Define numerical preprocessing pipeline
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler())
        ]
    )

    # Define PCA pipelines
    pca_pipeline_3 = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=3))
        ]
    )

    pca_pipeline_5 = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=5))
        ]
    )

    # Read columns from a CSV file (assuming this CSV contains column names and their types)
    cols = pd.read_csv('artifacts/data/cols_df.csv')
    numeric_cols = cols.loc[cols.type == 'numeric']['col'].to_list()
    categorical_cols = cols.loc[cols.type == 'categorical']['col'].to_list()

    # Define specific column categories for team, expected, rolling averages, and others
    team_cols = ['team_scored', 'team_conceded', 'home', 'win', 'draw', 'loss', 'next_game_home', 'next_kickoff_time',
                 'team_scored_rolling_avg', 'team_conceded_rolling_avg', 'home_rolling_avg', 'win_rolling_avg',
                 'draw_rolling_avg', 'loss_rolling_avg']

    expected_cols = [feature for feature in numeric_cols if ("expected_" in feature and feature not in team_cols)]
    rolling_avg_cols = [feature for feature in numeric_cols if ("_rolling_avg" in feature and feature not in team_cols + expected_cols)]
    other_cols = [feature for feature in numeric_cols if (feature not in [team_cols, rolling_avg_cols, expected_cols])]

    # Filter the columns using the only_include_features function based on included features
    numeric_cols = only_include_features(numeric_cols, included_features=included_features)
    categorical_cols = only_include_features(categorical_cols, included_features=included_features)
    team_cols = only_include_features(team_cols, included_features=included_features)
    expected_cols = only_include_features(expected_cols, included_features=included_features)
    rolling_avg_cols = only_include_features(rolling_avg_cols, included_features=included_features)
    other_cols = only_include_features(other_cols, included_features=included_features)

    # Create a preprocessor that combines all pipelines using a ColumnTransformer
    preprocessor = ColumnTransformer(
        [
            ("cat_pipelines", cat_pipeline, categorical_cols),
            ("num_pipeline", num_pipeline, numeric_cols),
            ("pca_all", pca_pipeline_5, numeric_cols),
            ("pca_team", pca_pipeline_3, team_cols),
            ("pca_expected", pca_pipeline_3, expected_cols),
            ("pca_rolling_avg", pca_pipeline_5, rolling_avg_cols),
            ("other_pca", pca_pipeline_3, other_cols),
        ]
    )

    joblib.dump(preprocessor, 'artifacts/feature_engineered_data/feature_engineering_pipeline.joblib')