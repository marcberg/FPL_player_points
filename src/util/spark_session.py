from pyspark.sql import SparkSession

def get_spark_session():
    """
    Initializes and returns a SparkSession.

    Args:
        None: This function does not require any input.
    
    Returns:
        SparkSession: A SparkSession object that can be used to run Spark operations.
    """
    print('Setup spark session')
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName('local') \
        .getOrCreate()
    
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    
    return spark