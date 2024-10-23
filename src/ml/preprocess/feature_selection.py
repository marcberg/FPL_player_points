import pandas as pd
import numpy as np
from datetime import datetime
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import statsmodels.api as sm
import statsmodels.formula.api as smf

def test_significant(df, feature, target):
    """
    Test the significance of a feature's relationship with the target using one-way ANOVA.
    
    Inputs:
    - df: pandas DataFrame containing the dataset.
    - feature: string representing the feature column to be tested.
    - target: string representing the target column in the dataset.

    Output:
    - result: pandas DataFrame with two columns:
        - 'feature': the name of the feature being tested.
        - 'PR(>F)': the p-value from ANOVA, indicating significance.
    """
    
    # Check if the feature is categorical (string) or a numeric feature with few unique values (<=5)
    if is_string_dtype(df[feature]) or (is_numeric_dtype(df[feature]) and len(df[feature].unique()) <= 5):

        # Create a temporary DataFrame with the feature and target
        df_temp = pd.DataFrame({"feature": df[feature], "target": df[target]})

        # Perform one-way ANOVA
        model = smf.ols('target ~ feature', data=df_temp).fit()
        anova_table = sm.stats.anova_lm(model)

        # Store the feature name and p-value in the result DataFrame
        result = pd.DataFrame({"feature": [feature], "PR(>F)": [anova_table['PR(>F)'].iloc[0]]})
        
    # If the feature is numeric, bin the values to make it categorical-like for ANOVA
    elif is_numeric_dtype(df[feature]):

        # Create bins based on percentiles to discretize the numeric feature
        bins = np.nanpercentile(df[feature], [0, 20, 40, 60, 80, 100])
        middle = bins[3]
        bins = [i for n, i in enumerate(bins) if i not in bins[:n]]  # Remove duplicates
        if len(bins) == 2 and bins[0] == middle:
            bins = [bins[0], bins[0]+0.1, bins[1]]
        elif len(bins) == 2 and bins[-1] == middle:
            bins = [bins[0], bins[1]-0.1, bins[1]]

        bins[0] = bins[0]-0.1
        bins[-1] = bins[-1]+0.1
                
        # Create a temporary DataFrame with the feature, binned feature, and target
        df_temp = pd.DataFrame({"feature": df[feature], "target": df[target]})
        df_temp[feature+'_bins'] = pd.cut(pd.to_numeric(df[feature]), bins, include_lowest=True)

        # Perform one-way ANOVA on the binned feature
        model = smf.ols(f"target ~ {feature+'_bins'}", data=df_temp).fit()
        anova_table = sm.stats.anova_lm(model)

        # Store the feature name and p-value in the result DataFrame
        result = pd.DataFrame({"feature": [feature], "PR(>F)": [anova_table['PR(>F)'].iloc[0]]})
    
    return result


def feature_selection_by_test(select_p_value=0.05):
    """
    Select features based on their significance to the target variable using ANOVA test.
    
    Inputs:
    - select_p_value: float, threshold for p-value below which a feature is considered significant.

    Output:
    - selected_features: list of feature names that are statistically significant.
    """
    
    print('\nSelect significant features and build feature engineering pipeline...')
    # get the training-set
    X_train = pd.read_csv('artifacts/split_data/X_train.csv')
    y_train = pd.read_csv('artifacts/split_data/y_train.csv')

    # Separate features into numerical and categorical columns
    cols = pd.read_csv('artifacts/data/cols_df.csv')
    target = cols.loc[cols.type == 'target']['col'].to_list()
    numeric_cols = cols.loc[cols.type == 'numeric']['col'].to_list()
    categorical_cols = cols.loc[cols.type == 'categorical']['col'].to_list()

    # Combine both numerical and categorical features
    features = numeric_cols + categorical_cols
    
    p_values = pd.DataFrame()  # To store p-values for all features
    df = pd.concat([X_train, y_train], axis=1)
    for p in features:
        # Test significance of each feature and store the result
        p_value = test_significant(df, p, target[0])
        p_values = pd.concat([p_values, p_value]).reset_index(drop=True)

    # Save the p-values to an Excel file with a timestamped filename
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    p_values.to_excel('artifacts/feature_engineered_data/feature_significance.xlsx'.format(current_time), index=False, header=True)
    
    # Select features where the p-value is below the specified threshold
    selected_features = p_values.loc[p_values["PR(>F)"] < select_p_value]["feature"].to_list()

    return selected_features
