
# TODO: fix file-path
def create_temp_view(name, spark):
    """
    Loads a CSV file into a Spark DataFrame and creates a temporary SQL view for querying.

    Inputs:
    - name (str): The name of the CSV file (without the extension) to load from the 'fetched_data' directory. 
      This name will also be used to create the temporary SQL view.
      
    Output:
    - Returns the Spark DataFrame that was created from the CSV file. Additionally, this DataFrame is registered 
      as a temporary SQL view accessible by the provided name, allowing SQL queries to be run on it via Spark.

    Process:
    - The function builds the file path for the CSV file based on the provided name.
    - Reads the CSV file using Spark, inferring the schema and using the first row as headers.
    - Registers the resulting DataFrame as a temporary view (accessible in Spark SQL) with the same name as the input.
    """
    
    # Construct the file path for the CSV file using the given name
    file_path = f"../FPL_predictions/artifacts/fetched_data/{name}.csv"
    
    # Read the CSV file into a Spark DataFrame, with headers and schema inference enabled
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    # Create or replace a temporary SQL view using the DataFrame, accessible by the name parameter
    df.createOrReplaceTempView(name)

    # Return the Spark DataFrame
    return df
