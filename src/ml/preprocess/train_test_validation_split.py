from sklearn.model_selection import train_test_split
import pandas as pd

from src.util.delete_files_in_folder import delete_files_in_folder
from src.util.save_csv import save_multiple_csv

def split_data(train_size=0.6, perform_crossvalidation=True, random_state=None):
    """
    Splits the data into train, test, and validation sets.
    If cv is True, no validation set is created for cross-validation.

    Returns a dictionary with keys: 'train', 'val', and 'test' (if cv=False).

    Parameters:
    X: Features (Pandas DataFrame or NumPy array)
    y: Target (Pandas Series or NumPy array)
    test_size: Proportion of the data to use as the test set (ignored if cv is True)
    val_size: Proportion of the training data to use as the validation set
    cv: Boolean, if True, splits only into train and validation
    random_state: Seed for reproducibility

    Returns:
    Dictionary with 'train', 'val', and 'test' keys containing the respective splits.
    """
    
    print('Split up data')
    path = 'artifacts/split_data/'
    delete_files_in_folder(path)
    
    X = pd.read_csv('artifacts/data/X.csv')
    y = pd.read_csv('artifacts/data/y.csv')

    if perform_crossvalidation:

        # Only split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-train_size), random_state=random_state)

        save_multiple_csv(obj_dict={'X_train': X_train, 
                                    'y_train': y_train,
                                    'X_test': X_test,
                                    'y_test': y_test
                                },
                          path=path)

        return {
            'train': (X_train, y_train),
            'test': (X_test, y_test),
            'val': None  # No val-set when performing cross-validation
        }
    
    else:
        # Split into train, test, and validation sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1-train_size), random_state=random_state)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

        save_multiple_csv(obj_dict={'X_train': X_train, 
                                    'y_train': y_train,
                                    'X_test': X_test,
                                    'y_test': y_test,
                                    'X_val': X_val,
                                    'y_val': y_val
                                },
                          path=path)
        
        return {
            'train': (X_train, y_train),
            'test': (X_test, y_test),
            'val': (X_val, y_val)
        }
