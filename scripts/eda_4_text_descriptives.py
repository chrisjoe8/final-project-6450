from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StopWordsRemover
import os

MASTER_URL = "spark://172.31.21.166:7077" #MASTER_PRIVATE_IP
INPUT_PATH = "s3a://chris-joe-datsbd-s2026-v2/project/feature_base_v1/"
OUTPUT_DIR = "outputs/eda_section4"

os.makedirs(OUTPUT_DIR, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("EDA_Section4_TextDescriptives")
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
# 1. Build text-valid + user-valid subset
# --------------------------------------------------
text_user_df = df.filter(
    (F.col("is_removed_or_deleted") == 0) &
    (F.col("is_deleted_author") == 0)
)

print("\nTEXT + USER VALID ROW COUNT")
print(text_user_df.count())

# --------------------------------------------------
# 2. Recreate user_type (single vs multi-subreddit)
# --------------------------------------------------
user_subreddit_counts = (
    text_user_df.groupBy("author")
    .agg(F.countDistinct("subreddit").alias("distinct_subreddits"))
    .withColumn(
        "user_type",
        F.when(F.col("distinct_subreddits") == 1, "single_subreddit")
         .otherwise("multi_subreddit")
    )
)

text_user_labeled_df = text_user_df.join(
    user_subreddit_counts.select("author", "user_type"),
    on="author",
    how="inner"
)

# --------------------------------------------------
# 3. Text length by controversy status
# --------------------------------------------------
text_length_by_controversy = (
    text_user_labeled_df.groupBy("controversiality")
    .agg(
        F.count("*").alias("total_comments"),
        F.avg("body_length_chars").alias("avg_body_length_chars"),
        F.expr("percentile_approx(body_length_chars, 0.5)").alias("median_body_length_chars"),
        F.avg("body_length_words").alias("avg_body_length_words"),
        F.expr("percentile_approx(body_length_words, 0.5)").alias("median_body_length_words")
    )
    .orderBy("controversiality")
)

print("\nTEXT LENGTH BY CONTROVERSY STATUS")
text_length_by_controversy.show(truncate=False)

# --------------------------------------------------
# 4. Text length by subreddit
# --------------------------------------------------
text_length_by_subreddit = (
    text_user_labeled_df.groupBy("subreddit")
    .agg(
        F.count("*").alias("total_comments"),
        F.avg("body_length_chars").alias("avg_body_length_chars"),
        F.expr("percentile_approx(body_length_chars, 0.5)").alias("median_body_length_chars"),
        F.avg("body_length_words").alias("avg_body_length_words"),
        F.expr("percentile_approx(body_length_words, 0.5)").alias("median_body_length_words")
    )
    .orderBy(F.desc("total_comments"))
)

print("\nTEXT LENGTH BY SUBREDDIT")
text_length_by_subreddit.show(20, truncate=False)

# --------------------------------------------------
# 5. Tokenize text using Spark built-ins
# --------------------------------------------------
token_base = (
    text_user_labeled_df
    .select("author", "subreddit", "controversiality", "user_type", "body")
    .withColumn("body_clean", F.lower(F.col("body")))
    .withColumn("body_clean", F.regexp_replace("body_clean", r"http\S+", " "))
    .withColumn("body_clean", F.regexp_replace("body_clean", r"www\.\S+", " "))
    .withColumn("body_clean", F.regexp_replace("body_clean", r"[^a-z0-9\s]", " "))
    .withColumn("body_clean", F.regexp_replace("body_clean", r"\s+", " "))
    .withColumn("tokens", F.split(F.trim(F.col("body_clean")), r"\s+"))
)

remover = StopWordsRemover(inputCol="tokens", outputCol="tokens_nostop")
token_base = remover.transform(token_base)

tokens_df = (
    token_base
    .withColumn("token", F.explode("tokens_nostop"))
    .filter(F.length("token") >= 3)
    .filter(~F.col("token").rlike(r"^\d+$"))
)

# --------------------------------------------------
# 6. Most common terms by controversial vs non-controversial
# --------------------------------------------------
term_counts_by_controversy = (
    tokens_df.groupBy("controversiality", "token")
    .count()
)

w1 = Window.partitionBy("controversiality").orderBy(F.desc("count"), F.asc("token"))

top_terms_by_controversy = (
    term_counts_by_controversy
    .withColumn("rank", F.row_number().over(w1))
    .filter(F.col("rank") <= 20)
    .orderBy("controversiality", "rank")
)

print("\nTOP TERMS BY CONTROVERSY STATUS")
top_terms_by_controversy.show(50, truncate=False)

# --------------------------------------------------
# 7. Most common terms by subreddit
# --------------------------------------------------
term_counts_by_subreddit = (
    tokens_df.groupBy("subreddit", "token")
    .count()
)

w2 = Window.partitionBy("subreddit").orderBy(F.desc("count"), F.asc("token"))

top_terms_by_subreddit = (
    term_counts_by_subreddit
    .withColumn("rank", F.row_number().over(w2))
    .filter(F.col("rank") <= 20)
    .orderBy("subreddit", "rank")
)

print("\nTOP TERMS BY SUBREDDIT")
top_terms_by_subreddit.show(150, truncate=False)

# --------------------------------------------------
# 8. Most common terms by user_type
# --------------------------------------------------
term_counts_by_user_type = (
    tokens_df.groupBy("user_type", "token")
    .count()
)

w3 = Window.partitionBy("user_type").orderBy(F.desc("count"), F.asc("token"))

top_terms_by_user_type = (
    term_counts_by_user_type
    .withColumn("rank", F.row_number().over(w3))
    .filter(F.col("rank") <= 20)
    .orderBy("user_type", "rank")
)

print("\nTOP TERMS BY USER TYPE")
top_terms_by_user_type.show(50, truncate=False)

# --------------------------------------------------
# 9. Save outputs
# --------------------------------------------------
text_length_by_controversy.toPandas().to_csv(
    f"{OUTPUT_DIR}/text_length_by_controversy.csv", index=False
)

text_length_by_subreddit.toPandas().to_csv(
    f"{OUTPUT_DIR}/text_length_by_subreddit.csv", index=False
)

top_terms_by_controversy.toPandas().to_csv(
    f"{OUTPUT_DIR}/top_terms_by_controversy.csv", index=False
)

top_terms_by_subreddit.toPandas().to_csv(
    f"{OUTPUT_DIR}/top_terms_by_subreddit.csv", index=False
)

top_terms_by_user_type.toPandas().to_csv(
    f"{OUTPUT_DIR}/top_terms_by_user_type.csv", index=False
)

print(f"\nSaved summary outputs to: {OUTPUT_DIR}")

spark.stop()