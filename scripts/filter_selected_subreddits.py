from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import os

MASTER_URL = "spark://172.31.21.33:7077"
INPUT_PATH = "s3a://eleni-zournatzi-datsbd-s2026/reddit-project/parquet/comments/"
OUTPUT_PATH = "s3a://eleni-zournatzi-datsbd-s2026/project/filtered_comments_v1/"
#MASTER_URL = "spark://172.31.21.206:7077"
#INPUT_PATH = "s3a://chris-joe-datsbd-s2026-v2/reddit-data/parquet/comments/"
#OUTPUT_PATH = "s3a://chris-joe-datsbd-s2026-v2/project/filtered_comments_v1/"

SELECTED_SUBREDDITS = [
    "politics",
    "worldnews",
    "news",
    "WhitePeopleTwitter",
    "conspiracy",
    "changemyview",
]

spark = (
    SparkSession.builder
    .appName("FilterSelectedSubreddits")
    .master(MASTER_URL)
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider"
    )
    .getOrCreate()
)

df = spark.read.parquet(INPUT_PATH)

filtered = df.filter(F.col("subreddit").isin(SELECTED_SUBREDDITS))

(
    filtered.write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)

summary = (
    filtered.groupBy("subreddit")
    .count()
    .orderBy(F.desc("count"))
)

summary.show(truncate=False)

spark.stop()