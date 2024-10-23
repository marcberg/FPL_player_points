import os

def save_multiple_csv(obj_dict, path="."):
    """
    Save multiple pandas DataFrames as separate CSV files.

    Parameters:
    obj_dict (dict): A dictionary where the keys are the variable names (strings) and the values are the DataFrames to be saved.
    path (str): The directory path where the CSV files should be saved. Defaults to the current directory.
    
    Returns:
    None: Saves the files to the specified path with the file names based on the variable names.
    """
    # Ensure the directory exists
    if not os.path.exists(path):
        os.makedirs(path)
    
    for obj_name, obj in obj_dict.items():
        file_path = os.path.join(path, f"{obj_name}.csv")
        obj.to_csv(file_path, index=False, header=True)
        print(f"- {file_path} is created")