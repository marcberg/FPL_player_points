import joblib
import pandas as pd

from src.util.read_multiple_csv import import_csv_files

def get_features_name():
    # import the data
    data_path = 'artifacts/split_data/'
    data = import_csv_files(data_path) 
        
    if 'X_val' in data:
        X = pd.concat([data['X_train'], data['X_val']], ignore_index=True)
    else:
        X = data['X_train']

    # import the pipeline and fit it so all one hot encoding names exists
    feature_engineering_pipeline = joblib.load('artifacts/feature_engineered_data/feature_engineering_pipeline.joblib')
    feature_engineering_pipeline.fit(X)


    # Loop through each transformer in the ColumnTransformer to get categorical and numerical features used in the model
    # some features are filtered out after significance-test.
    feature_names = []
    for name, transformer, columns in feature_engineering_pipeline.transformers:
        if name == "cat_pipelines":  
            cat_feature_names = feature_engineering_pipeline.named_transformers_['cat_pipelines']['one_hot_encoder'].get_feature_names_out(columns)
            feature_names.extend(cat_feature_names.tolist())
        elif name == "num_pipeline":  
            feature_names.extend(columns)
            missing_value_columns = [col for col in columns if X[col].isna().any()]
            feature_names.extend(['missing_indicator_' + s for s in missing_value_columns])
        elif 'pca' in name:
            pca = transformer.named_steps.get('pca', None)
            n_components = pca.n_components
            feature_names.extend([f"{name}_pca_{i+1}" for i in range(n_components)])  

    return feature_names