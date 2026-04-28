from pyspark.sql import SparkSession
import pyspark.sql.functions as F

MASTER_URL = "spark://172.31.27.59:7077" # MASTER_PRIVATE_IP
INPUT_PATH = "s3a://eleni-zournatzi-datsbd-s2026/reddit-project/parquet/comments/"
OUTPUT_PATH = "s3a://eleni-zournatzi-datsbd-s2026/project/filtered_comments_v1/"
#MASTER_URL = "spark://172.31.21.206:7077"
#INPUT_PATH = "s3a://chris-joe-datsbd-s2026-v2/project/filtered_comments_v1/"
#OUTPUT_PATH = "s3a://chris-joe-datsbd-s2026-v2/project/feature_base_v1/"

spark = (
    SparkSession.builder
    .appName("BuildFeatureBaseV1")
    .master(MASTER_URL)
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider"
    )
    .getOrCreate()
)

df = spark.read.parquet(INPUT_PATH)

df2 = (
    df.select(
        "id",
        "author",
        "subreddit",
        "body",
        "controversiality",
        "created_utc",
        "score",
        "link_id",
        "parent_id"
    )
    .withColumn("created_ts", F.from_unixtime("created_utc").cast("timestamp"))
    .withColumn("year", F.year("created_ts"))
    .withColumn("month", F.month("created_ts"))
    .withColumn("day", F.dayofmonth("created_ts"))
    .withColumn("hour", F.hour("created_ts"))
    .withColumn("weekday", F.dayofweek("created_ts"))
    .withColumn("body_length_chars", F.length("body"))
    .withColumn("body_length_words", F.size(F.split(F.col("body"), r"\s+")))
    .withColumn(
        "is_removed_or_deleted",
        F.when(
            (F.col("body") == "[removed]") | (F.col("body") == "[deleted]"),
            1
        ).otherwise(0)
    )
    .withColumn(
        "is_deleted_author",
        F.when(F.col("author") == "[deleted]", 1).otherwise(0)
    )
)

(
    df2.write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)

print("\nROW COUNT")
print(df2.count())

print("\nREMOVED/DELETED BODY DISTRIBUTION")
df2.groupBy("is_removed_or_deleted").count().orderBy("is_removed_or_deleted").show()

print("\nDELETED AUTHOR DISTRIBUTION")
df2.groupBy("is_deleted_author").count().orderBy("is_deleted_author").show()

print("\nCONTROVERSIALITY DISTRIBUTION")
df2.groupBy("controversiality").count().orderBy("controversiality").show()

print("\nSAMPLE")
df2.select(
    "author", "subreddit", "controversiality", "score",
    "year", "month", "hour",
    "body_length_chars", "body_length_words",
    "is_removed_or_deleted", "is_deleted_author", "body"
).show(10, truncate=80)

spark.stop()