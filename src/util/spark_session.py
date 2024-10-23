import os
from pyspark import SparkConf
from pyspark.sql import SparkSession

def get_spark_session():
    """
    Initializes and returns a SparkSession with appropriate logging configuration.

    Args:
        None: This function does not require any input.
    
    Returns:
        SparkSession: A SparkSession object that can be used to run Spark operations.
    """
    
    print('\nSetup spark session...')

    # Set environment variable to point to the directory containing log4j.properties
    os.environ["SPARK_CONF_DIR"] = "."

    # Setup Spark configuration
    conf = SparkConf()
    conf.set("spark.executor.extraJavaOptions", "-Dlog4j.configuration=file:./log4j.properties")
    conf.set("spark.driver.extraJavaOptions", "-Dlog4j.configuration=file:./log4j.properties")

    # Initialize Spark session with the custom configuration
    spark = SparkSession.builder \
        .appName('local') \
        .config(conf=conf) \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    
    return spark
