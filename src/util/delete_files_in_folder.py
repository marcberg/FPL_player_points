import os
import glob

def delete_files_in_folder(folder_path):
    """
    Deletes all files in the specified folder.

    Args:
        folder_path (str): The path to the folder where files should be deleted.
    
    Returns:
        None: This function does not return anything but prints the names of deleted files.
    """
    # Get all files in the folder
    files = glob.glob(os.path.join(folder_path, "*"))
    
    for file in files:
        if os.path.isfile(file):
            os.remove(file)