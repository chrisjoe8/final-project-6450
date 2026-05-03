from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import os

MASTER_URL = "spark://172.31.95.81:7077" # MASTER_PRIVATE_IP
INPUT_PATH = "s3a://chris-joe-datsbd-s2026-v2/project/feature_base_v1/"
OUTPUT_PATH = "s3a://chris-joe-datsbd-s2026-v2/project/ml_model_base_v1/"
OUTPUT_DIR = "outputs/ml_prep_model_base"

os.makedirs(OUTPUT_DIR, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("ML_Prep_Model_Base")
    .master(MASTER_URL)
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider"
    )
    .getOrCreate()
)

# --------------------------------------------------
# 1. Load feature base
# --------------------------------------------------
df = spark.read.parquet(INPUT_PATH)

print("\nFULL FEATURE BASE ROW COUNT")
print(df.count())

# --------------------------------------------------
# 2. Keep rows valid for modeling
#    - remove deleted/removed text
#    - remove deleted authors
#    - keep non-empty body
# --------------------------------------------------
model_df = (
    df.filter(F.col("is_removed_or_deleted") == 0)
      .filter(F.col("is_deleted_author") == 0)
      .filter(F.col("body").isNotNull())
      .filter(F.length(F.trim(F.col("body"))) > 0)
)

print("\nMODEL-VALID ROW COUNT")
print(model_df.count())

# --------------------------------------------------
# 3. Create user-level cross-community features
# --------------------------------------------------
user_features = (
    model_df.groupBy("author")
    .agg(
        F.countDistinct("subreddit").alias("distinct_subreddits_per_user"),
        F.count("*").alias("total_comments_per_user"),
        F.avg("score").alias("avg_score_per_user"),
        F.sum("controversiality").alias("total_controversial_comments_per_user")
    )
    .withColumn(
        "is_multi_subreddit_user",
        F.when(F.col("distinct_subreddits_per_user") > 1, 1).otherwise(0)
    )
)

# --------------------------------------------------
# 4. Create subreddit-level features
# --------------------------------------------------
subreddit_features = (
    model_df.groupBy("subreddit")
    .agg(
        F.count("*").alias("subreddit_total_comments"),
        F.avg("score").alias("subreddit_avg_score"),
        F.avg("body_length_chars").alias("subreddit_avg_body_length_chars"),
        F.avg("body_length_words").alias("subreddit_avg_body_length_words"),
        F.avg("controversiality").alias("subreddit_controversy_rate")
    )
)

# --------------------------------------------------
# 5. Join features back to comment-level data
# --------------------------------------------------
model_df = (
    model_df.join(user_features, on="author", how="left")
            .join(subreddit_features, on="subreddit", how="left")
)

# --------------------------------------------------
# 6. Create a few simple interpretable ML features
# --------------------------------------------------
model_df = (
    model_df
    .withColumn("label", F.col("controversiality").cast("int"))
    .withColumn("score_rounded", F.round(F.col("score"), 2))
    .withColumn("body_length_chars_rounded", F.round(F.col("body_length_chars"), 2))
    .withColumn("body_length_words_rounded", F.round(F.col("body_length_words"), 2))
    .withColumn("avg_score_per_user_rounded", F.round(F.col("avg_score_per_user"), 2))
    .withColumn("subreddit_avg_score_rounded", F.round(F.col("subreddit_avg_score"), 2))
    .withColumn(
        "subreddit_controversy_rate_rounded",
        F.round(F.col("subreddit_controversy_rate"), 4)
    )
    .withColumn(
        "log_body_length_words",
        F.round(F.log1p(F.col("body_length_words")), 4)
    )
    .withColumn(
        "log_total_comments_per_user",
        F.round(F.log1p(F.col("total_comments_per_user")), 4)
    )
)

# --------------------------------------------------
# 7. Keep only columns needed for ML
# --------------------------------------------------
ml_base = model_df.select(
    "id",
    "author",
    "subreddit",
    "body",
    "label",
    "controversiality",
    "score_rounded",
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "body_length_chars_rounded",
    "body_length_words_rounded",
    "log_body_length_words",
    "distinct_subreddits_per_user",
    "total_comments_per_user",
    "log_total_comments_per_user",
    "avg_score_per_user_rounded",
    "total_controversial_comments_per_user",
    "is_multi_subreddit_user",
    "subreddit_total_comments",
    "subreddit_avg_score_rounded",
    "subreddit_avg_body_length_chars",
    "subreddit_avg_body_length_words",
    "subreddit_controversy_rate_rounded"
)

# --------------------------------------------------
# 8. Basic checks
# --------------------------------------------------
print("\nML BASE ROW COUNT")
print(ml_base.count())

print("\nLABEL DISTRIBUTION")
(
    ml_base.groupBy("label")
    .count()
    .orderBy("label")
    .show()
)

print("\nSUBREDDIT DISTRIBUTION")
(
    ml_base.groupBy("subreddit")
    .count()
    .orderBy(F.desc("count"))
    .show(20, truncate=False)
)

print("\nMULTI-SUBREDDIT USER DISTRIBUTION")
(
    ml_base.groupBy("is_multi_subreddit_user")
    .count()
    .orderBy("is_multi_subreddit_user")
    .show()
)

print("\nSAMPLE ROWS")
(
    ml_base.select(
        "subreddit",
        "label",
        "score_rounded",
        "hour",
        "weekday",
        "body_length_words_rounded",
        "distinct_subreddits_per_user",
        "is_multi_subreddit_user",
        "subreddit_controversy_rate_rounded"
    )
    .show(10, truncate=False)
)

# --------------------------------------------------
# 9. Save ML-ready base to S3
# --------------------------------------------------
(
    ml_base.write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)

# --------------------------------------------------
# 10. Save small summary outputs locally on master
# --------------------------------------------------
label_dist = (
    ml_base.groupBy("label")
    .count()
    .orderBy("label")
)

subreddit_dist = (
    ml_base.groupBy("subreddit")
    .count()
    .orderBy(F.desc("count"))
)

multi_user_dist = (
    ml_base.groupBy("is_multi_subreddit_user")
    .count()
    .orderBy("is_multi_subreddit_user")
)

label_dist.toPandas().to_csv(
    f"{OUTPUT_DIR}/label_distribution.csv", index=False
)

subreddit_dist.toPandas().to_csv(
    f"{OUTPUT_DIR}/subreddit_distribution.csv", index=False
)

multi_user_dist.toPandas().to_csv(
    f"{OUTPUT_DIR}/multi_subreddit_user_distribution.csv", index=False
)

with open(f"{OUTPUT_DIR}/ml_prep_notes.txt", "w") as f:
    f.write("ML PREP MODEL BASE SUMMARY\n")
    f.write(f"INPUT_PATH: {INPUT_PATH}\n")
    f.write(f"OUTPUT_PATH: {OUTPUT_PATH}\n")
    f.write(f"FULL_FEATURE_BASE_ROWS: {df.count()}\n")
    f.write(f"MODEL_VALID_ROWS: {model_df.count()}\n")
    f.write(f"ML_BASE_ROWS: {ml_base.count()}\n")

print(f"\nSaved ML-ready base to: {OUTPUT_PATH}")
print(f"Saved summary outputs to: {OUTPUT_DIR}")

spark.stop()