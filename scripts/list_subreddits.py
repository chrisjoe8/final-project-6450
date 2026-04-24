from pyspark.sql import SparkSession
import pyspark.sql.functions as F

MASTER_URL = "spark://172.31.21.206:7077"
S3_PATH = "s3a://chris-joe-datsbd-s2026-v2/reddit-data/parquet/comments/yyyy=2024/mm=01/"

spark = (
    SparkSession.builder
    .appName("ListSubreddits")
    .master(MASTER_URL)
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider"
    )
    .getOrCreate()
)

top_subreddits = (
    spark.read.parquet(S3_PATH)
    .groupBy("subreddit")
    .count()
    .orderBy(F.desc("count"))
)

top_subreddits.show(100, truncate=False)
top_subreddits.toPandas().to_csv("outputs/top_subreddits_jan2024.csv", index=False)

spark.stop()