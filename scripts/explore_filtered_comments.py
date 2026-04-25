from pyspark.sql import SparkSession
import pyspark.sql.functions as F

MASTER_URL = "spark://172.31.21.206:7077"
INPUT_PATH = "s3a://chris-joe-datsbd-s2026-v2/project/filtered_comments_v1/"

spark = (
    SparkSession.builder
    .appName("ExploreFilteredComments")
    .master(MASTER_URL)
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider"
    )
    .getOrCreate()
)

df = spark.read.parquet(INPUT_PATH)

print("\nROW COUNT")
print(df.count())

print("\nSCHEMA")
df.printSchema()

print("\nSUBREDDIT COUNTS")
df.groupBy("subreddit").count().orderBy(F.desc("count")).show(20, truncate=False)

print("\nCONTROVERSIALITY DISTRIBUTION")
if "controversiality" in df.columns:
    df.groupBy("controversiality").count().orderBy("controversiality").show()
else:
    print("controversiality column not found")

print("\nNULL CHECKS")
cols_to_check = [c for c in ["author", "subreddit", "body", "controversiality", "created_utc"] if c in df.columns]
null_exprs = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in cols_to_check]
df.select(null_exprs).show(truncate=False)

print("\nSAMPLE ROWS")
sample_cols = [c for c in ["author", "subreddit", "controversiality", "score", "created_utc", "body"] if c in df.columns]
df.select(sample_cols).show(10, truncate=80)

spark.stop()