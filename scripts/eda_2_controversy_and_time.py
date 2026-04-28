from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import os

MASTER_URL = "spark://172.31.21.166:7077" #MASTER_PRIVATE_IP
INPUT_PATH = "s3a://chris-joe-datsbd-s2026-v2/project/feature_base_v1/"
OUTPUT_DIR = "outputs/eda_section2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("EDA_Section2_Controversy_And_Time")
    .master(MASTER_URL)
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider"
    )
    .getOrCreate()
)

df = spark.read.parquet(INPUT_PATH)

# Use full cleaned feature base for controversy/time analysis
all_df = df

# --------------------------------------------------
# 1. Controversy by subreddit
# --------------------------------------------------
controversy_by_subreddit = (
    all_df.groupBy("subreddit")
    .agg(
        F.count("*").alias("total_comments"),
        F.sum("controversiality").alias("controversial_comments")
    )
    .withColumn(
        "controversy_rate",
        F.col("controversial_comments") / F.col("total_comments")
    )
    .orderBy(F.desc("controversy_rate"))
)

print("\nCONTROVERSY BY SUBREDDIT")
controversy_by_subreddit.show(20, truncate=False)

# --------------------------------------------------
# 2. Controversy by month
# --------------------------------------------------
controversy_by_month = (
    all_df.groupBy("year", "month")
    .agg(
        F.count("*").alias("total_comments"),
        F.sum("controversiality").alias("controversial_comments")
    )
    .withColumn(
        "controversy_rate",
        F.col("controversial_comments") / F.col("total_comments")
    )
    .orderBy("year", "month")
)

print("\nCONTROVERSY BY MONTH")
controversy_by_month.show(50, truncate=False)

# --------------------------------------------------
# 3. Controversy by hour
# --------------------------------------------------
controversy_by_hour = (
    all_df.groupBy("hour")
    .agg(
        F.count("*").alias("total_comments"),
        F.sum("controversiality").alias("controversial_comments")
    )
    .withColumn(
        "controversy_rate",
        F.col("controversial_comments") / F.col("total_comments")
    )
    .orderBy("hour")
)

print("\nCONTROVERSY BY HOUR")
controversy_by_hour.show(24, truncate=False)

# --------------------------------------------------
# 4. Controversy by weekday
# Spark dayofweek: 1=Sunday, 2=Monday, ..., 7=Saturday
# --------------------------------------------------
weekday_labels = (
    all_df.withColumn(
        "weekday_name",
        F.when(F.col("weekday") == 1, "Sunday")
         .when(F.col("weekday") == 2, "Monday")
         .when(F.col("weekday") == 3, "Tuesday")
         .when(F.col("weekday") == 4, "Wednesday")
         .when(F.col("weekday") == 5, "Thursday")
         .when(F.col("weekday") == 6, "Friday")
         .when(F.col("weekday") == 7, "Saturday")
    )
)

controversy_by_weekday = (
    weekday_labels.groupBy("weekday", "weekday_name")
    .agg(
        F.count("*").alias("total_comments"),
        F.sum("controversiality").alias("controversial_comments")
    )
    .withColumn(
        "controversy_rate",
        F.col("controversial_comments") / F.col("total_comments")
    )
    .orderBy("weekday")
)

print("\nCONTROVERSY BY WEEKDAY")
controversy_by_weekday.show(10, truncate=False)

# --------------------------------------------------
# 5. Controversy over time by subreddit
# --------------------------------------------------
controversy_over_time_by_subreddit = (
    all_df.groupBy("subreddit", "year", "month")
    .agg(
        F.count("*").alias("total_comments"),
        F.sum("controversiality").alias("controversial_comments")
    )
    .withColumn(
        "controversy_rate",
        F.col("controversial_comments") / F.col("total_comments")
    )
    .orderBy("subreddit", "year", "month")
)

print("\nCONTROVERSY OVER TIME BY SUBREDDIT")
controversy_over_time_by_subreddit.show(100, truncate=False)

# --------------------------------------------------
# 6. Save outputs
# --------------------------------------------------
controversy_by_subreddit.toPandas().to_csv(
    f"{OUTPUT_DIR}/controversy_by_subreddit.csv", index=False
)

controversy_by_month.toPandas().to_csv(
    f"{OUTPUT_DIR}/controversy_by_month.csv", index=False
)

controversy_by_hour.toPandas().to_csv(
    f"{OUTPUT_DIR}/controversy_by_hour.csv", index=False
)

controversy_by_weekday.toPandas().to_csv(
    f"{OUTPUT_DIR}/controversy_by_weekday.csv", index=False
)

controversy_over_time_by_subreddit.toPandas().to_csv(
    f"{OUTPUT_DIR}/controversy_over_time_by_subreddit.csv", index=False
)

print(f"\nSaved summary outputs to: {OUTPUT_DIR}")

spark.stop()