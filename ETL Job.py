import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, when, year
from pyspark.ml.feature import MinMaxScaler, VectorAssembler

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Load data
df = spark.read.csv("s3://your-bucket/electric_vehicle_population_data.csv", header=True, inferSchema=True)

# Filter data: Keep only records from 2020 onwards
df_filtered = df.filter(year(col("ModelYear")) >= 2020)

# Data Cleaning: Fill missing 'VehicleLocation' with 'Unknown'
df_cleaned = df_filtered.withColumn("VehicleLocation", when(col("VehicleLocation").isNull(), "Unknown").otherwise(col("VehicleLocation")))

# Remove duplicates based on VIN
df_cleaned = df_cleaned.dropDuplicates(["VIN"])

# Column Operations: Rename column, drop column, and add new column
df_renamed = df_cleaned.withColumnRenamed("EVModel", "Model")
df_dropped = df_renamed.drop("VehicleID")
df_with_age = df_dropped.withColumn("Age", 2024 - col("ModelYear"))

# Data Type Conversion: Convert 'BatteryCapacity' to float
df_converted = df_with_age.withColumn("BatteryCapacity", col("BatteryCapacity").cast("float"))

# Normalization: Normalize 'BatteryCapacity' to range [0, 1]
assembler = VectorAssembler(inputCols=["BatteryCapacity"], outputCol="BatteryCapacityVec")
df_vector = assembler.transform(df_converted)

scaler = MinMaxScaler(inputCol="BatteryCapacityVec", outputCol="ScaledBatteryCapacity")
scaler_model = scaler.fit(df_vector)
df_scaled = scaler_model.transform(df_vector).drop("BatteryCapacityVec")

# Aggregation: Count number of vehicles per make and model
df_aggregated = df_scaled.groupBy("Make", "Model").count()

# Merge with additional dataset (if available)
# Assuming we have another dataset with charging station information
# charging_stations_df = spark.read.csv("s3://your-bucket/charging_stations.csv", header=True, inferSchema=True)
# merged_df = df_converted.join(charging_stations_df, df_converted["Region"] == charging_stations_df["Region"], "inner")

# Convert back to DynamicFrame
transformed_data = DynamicFrame.fromDF(df_aggregated, glueContext, "transformed_data")

# Write transformed data to Snowflake
sink1 = glueContext.write_dynamic_frame.from_options(frame=transformed_data, connection_type="snowflake", connection_options={
    "dbtable": "electric_vehicle_population_transformed",
    "database": "my_database",
    "schema": "public",
    "sfURL": "my_snowflake_url",
    "user": "my_user",
    "password": "my_password"
})

job.commit()
