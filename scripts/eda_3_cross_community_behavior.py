from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import os

MASTER_URL = "spark://172.31.86.90:7077" #MASTER_PRIVATE_IP
INPUT_PATH = "s3a://chris-joe-datsbd-s2026-v2/project/feature_base_v1/"
OUTPUT_DIR = "outputs/eda_section3"

os.makedirs(OUTPUT_DIR, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("EDA_Section3_CrossCommunityBehavior")
    .master(MASTER_URL)
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider"
    )
    .getOrCreate()
)

df = spark.read.parquet(INPUT_PATH)

# --------------------------------------------------
# 1. User-valid subset for cross-community analysis
# --------------------------------------------------
user_df = df.filter(F.col("is_deleted_author") == 0)

print("\nUSER-VALID ROW COUNT")
print(user_df.count())

print("\nDISTINCT USER COUNT")
print(user_df.select("author").distinct().count())

# --------------------------------------------------
# 2. Count distinct subreddits per user
# --------------------------------------------------
user_subreddit_counts = (
    user_df.groupBy("author")
    .agg(
        F.countDistinct("subreddit").alias("distinct_subreddits"),
        F.count("*").alias("total_comments"),
        F.sum("controversiality").alias("controversial_comments"),
        F.avg("score").alias("avg_score")
    )
    .withColumn(
        "user_type",
        F.when(F.col("distinct_subreddits") == 1, "single_subreddit")
         .otherwise("multi_subreddit")
    )
)

print("\nUSER TYPE COUNTS")
user_type_counts = (
    user_subreddit_counts.groupBy("user_type")
    .count()
    .orderBy("user_type")
)
user_type_counts.show(truncate=False)

# --------------------------------------------------
# 3. Attach user_type back to comment-level data
# --------------------------------------------------
user_labeled_df = user_df.join(
    user_subreddit_counts.select("author", "distinct_subreddits", "user_type"),
    on="author",
    how="inner"
)

# --------------------------------------------------
# 4. Compare single vs multi-community users
# --------------------------------------------------
user_type_comment_summary = (
    user_labeled_df.groupBy("user_type")
    .agg(
        F.count("*").alias("total_comments"),
        F.sum("controversiality").alias("controversial_comments"),
        F.round(F.avg("score"), 2).alias("avg_score"),
        F.round(F.avg("body_length_chars"), 2).alias("avg_body_length_chars"),
        F.round(F.avg("body_length_words"), 2).alias("avg_body_length_words")
    )
    .withColumn(
        "controversy_rate",
        F.round(F.col("controversial_comments") / F.col("total_comments"), 4)
    )
    .orderBy("user_type")
)

print("\nSINGLE VS MULTI-COMMUNITY COMMENT SUMMARY")
user_type_comment_summary.show(truncate=False)

# --------------------------------------------------
# 5. Compare users themselves (user-level summary)
# --------------------------------------------------
user_type_user_summary = (
    user_subreddit_counts.groupBy("user_type")
    .agg(
        F.count("*").alias("num_users"),
        F.round(F.avg("distinct_subreddits"), 2).alias("avg_distinct_subreddits"),
        F.round(F.avg("total_comments"), 2).alias("avg_comments_per_user"),
        F.round(F.avg("controversial_comments"), 2).alias("avg_controversial_comments_per_user"),
        F.round(F.avg("avg_score"), 2).alias("avg_user_mean_score")
    )
    .orderBy("user_type")
)

print("\nSINGLE VS MULTI-COMMUNITY USER SUMMARY")
user_type_user_summary.show(truncate=False)

# --------------------------------------------------
# 6. Controversy rate by subreddit AND user_type
# --------------------------------------------------
subreddit_user_type_summary = (
    user_labeled_df.groupBy("subreddit", "user_type")
    .agg(
        F.count("*").alias("total_comments"),
        F.sum("controversiality").alias("controversial_comments"),
        F.round(F.avg("score"), 2).alias("avg_score")
    )
    .withColumn(
        "controversy_rate",
        F.round(F.col("controversial_comments") / F.col("total_comments"), 4)
    )
    .orderBy("subreddit", "user_type")
)

print("\nSUBREDDIT x USER TYPE SUMMARY")
subreddit_user_type_summary.show(50, truncate=False)

# --------------------------------------------------
# 7. Cross-community participation over time
#    (share of comments from multi-community users by month)
# --------------------------------------------------
cross_community_over_time = (
    user_labeled_df.groupBy("year", "month", "user_type")
    .agg(F.count("*").alias("total_comments"))
    .orderBy("year", "month", "user_type")
)

print("\nCROSS-COMMUNITY PARTICIPATION OVER TIME")
cross_community_over_time.show(100, truncate=False)

# --------------------------------------------------
# 8. Subreddit pair co-participation
# --------------------------------------------------
user_subreddit_pairs_base = (
    user_labeled_df.select("author", "subreddit").distinct()
)

a = user_subreddit_pairs_base.alias("a")
b = user_subreddit_pairs_base.alias("b")

subreddit_pair_counts = (
    a.join(
        b,
        on=(
            (F.col("a.author") == F.col("b.author")) &
            (F.col("a.subreddit") < F.col("b.subreddit"))
        ),
        how="inner"
    )
    .groupBy(
        F.col("a.subreddit").alias("subreddit_a"),
        F.col("b.subreddit").alias("subreddit_b")
    )
    .agg(F.countDistinct(F.col("a.author")).alias("shared_users"))
    .orderBy(F.desc("shared_users"))
)

print("\nTOP SUBREDDIT PAIRS BY SHARED USERS")
subreddit_pair_counts.show(50, truncate=False)

# --------------------------------------------------
# 9. Save outputs
# --------------------------------------------------
user_type_counts.toPandas().to_csv(
    f"{OUTPUT_DIR}/user_type_counts.csv", index=False
)

user_type_comment_summary.toPandas().to_csv(
    f"{OUTPUT_DIR}/user_type_comment_summary.csv", index=False
)

user_type_user_summary.toPandas().to_csv(
    f"{OUTPUT_DIR}/user_type_user_summary.csv", index=False
)

subreddit_user_type_summary.toPandas().to_csv(
    f"{OUTPUT_DIR}/subreddit_user_type_summary.csv", index=False
)

cross_community_over_time.toPandas().to_csv(
    f"{OUTPUT_DIR}/cross_community_over_time.csv", index=False
)

subreddit_pair_counts.toPandas().to_csv(
    f"{OUTPUT_DIR}/subreddit_pair_counts.csv", index=False
)

print(f"\nSaved summary outputs to: {OUTPUT_DIR}")

spark.stop()