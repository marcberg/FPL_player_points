from pyspark.sql import Window
import pyspark.sql.functions as f

def calculate_rolling_avg(df, partition_by_cols, order_by_cols, window_size, cols):
    """
    Calculate rolling average for specified columns in a PySpark DataFrame, partitioned by team and season.
    
    Parameters:
    df (pyspark.sql.DataFrame): The input PySpark DataFrame.
    cols (list): List of column names to calculate the rolling average on.
    window_size (int): The size of the rolling window.
    partition_by_cols (list): List of columns to partition by (e.g., team and season).
    
    Returns:
    pyspark.sql.DataFrame: DataFrame with rolling average columns added.
    """

    window_spec = Window.partitionBy(partition_by_cols).orderBy(order_by_cols).rowsBetween(-window_size + 1, 0)
    
    for col in cols:
        df = df.withColumn(f"{col}_rolling_avg", f.avg(f.col(col)).over(window_spec))
    
    return df
