import os
import pandas as pd

def import_csv_files(folder_path):
    """
    Imports all CSV files in the specified folder and loads them into a dictionary of Pandas DataFrames.

    Args:
        folder_path (str): The path to the folder containing CSV files.
    
    Returns:
        dict: A dictionary where the keys are the CSV file names (without extension) and the values are the corresponding Pandas DataFrames.
    """
    # Find all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = {}
    
    # Load each CSV file into a Pandas DataFrame
    for file in csv_files:
        file_name = os.path.splitext(file)[0]  # Get file name without extension
        file_path = os.path.join(folder_path, file)
        dataframes[file_name] = pd.read_csv(file_path)
    
    return dataframes
