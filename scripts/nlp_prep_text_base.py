from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import os

MASTER_URL = "spark://172.31.86.233:7077" # MASTER_PRIVATE_IP
INPUT_PATH = "s3a://chris-joe-datsbd-s2026-v2/project/feature_base_v1/"
OUTPUT_PATH = "s3a://chris-joe-datsbd-s2026-v2/project/nlp_text_base_v1/"
OUTPUT_DIR = "outputs/nlp_prep_text_base"

os.makedirs(OUTPUT_DIR, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("NLP_Prep_Text_Base")
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
# 2. Keep rows valid for text analysis
#    - remove deleted/removed text
#    - remove deleted authors
#    - keep non-null/non-empty body
# --------------------------------------------------
text_df = (
    df.filter(F.col("is_removed_or_deleted") == 0)
      .filter(F.col("is_deleted_author") == 0)
      .filter(F.col("body").isNotNull())
      .filter(F.length(F.trim(F.col("body"))) > 0)
)

print("\nTEXT-VALID ROW COUNT")
print(text_df.count())

# --------------------------------------------------
# 3. Standardize time to a readable timestamp
#    Keep original UTC field too
# --------------------------------------------------
text_df = text_df.withColumn(
    "created_ts_utc",
    F.from_unixtime("created_utc").cast("timestamp")
)

# --------------------------------------------------
# 4. Create cleaned text fields
# --------------------------------------------------
text_df = (
    text_df
    .withColumn("body_lower", F.lower(F.col("body")))
    .withColumn("body_no_urls", F.regexp_replace("body_lower", r"http\S+", " "))
    .withColumn("body_no_urls", F.regexp_replace("body_no_urls", r"www\.\S+", " "))
    .withColumn("body_alpha_num", F.regexp_replace("body_no_urls", r"[^a-z0-9\s]", " "))
    .withColumn("body_clean", F.regexp_replace("body_alpha_num", r"\s+", " "))
    .withColumn("body_clean", F.trim(F.col("body_clean")))
)

# --------------------------------------------------
# 5. Recompute token/length features on cleaned text
# --------------------------------------------------
text_df = (
    text_df
    .withColumn(
        "clean_word_count",
        F.when(
            F.length(F.col("body_clean")) > 0,
            F.size(F.split(F.col("body_clean"), r"\s+"))
        ).otherwise(0)
    )
    .withColumn("clean_char_count", F.length(F.col("body_clean")))
)

# --------------------------------------------------
# 6. Keep only columns needed for NLP + later reference
# --------------------------------------------------
nlp_base = text_df.select(
    "id",
    "author",
    "subreddit",
    "controversiality",
    "score",
    "created_utc",
    "created_ts_utc",
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "body",
    "body_clean",
    "body_length_chars",
    "body_length_words",
    "clean_char_count",
    "clean_word_count"
)

# --------------------------------------------------
# 7. Basic checks
# --------------------------------------------------
print("\nNLP BASE ROW COUNT")
print(nlp_base.count())

print("\nSUBREDDIT COUNTS")
(
    nlp_base.groupBy("subreddit")
    .count()
    .orderBy(F.desc("count"))
    .show(20, truncate=False)
)

print("\nCONTROVERSIALITY DISTRIBUTION")
(
    nlp_base.groupBy("controversiality")
    .count()
    .orderBy("controversiality")
    .show()
)

print("\nAVERAGE CLEAN TEXT LENGTH BY SUBREDDIT")
(
    nlp_base.groupBy("subreddit")
    .agg(
        F.round(F.avg("clean_char_count"), 2).alias("avg_clean_char_count"),
        F.round(F.avg("clean_word_count"), 2).alias("avg_clean_word_count")
    )
    .orderBy(F.desc("avg_clean_word_count"))
    .show(20, truncate=False)
)

print("\nSAMPLE ROWS")
(
    nlp_base.select(
        "author",
        "subreddit",
        "controversiality",
        "score",
        "created_ts_utc",
        "body",
        "body_clean",
        "clean_word_count"
    )
    .show(10, truncate=100)
)

# --------------------------------------------------
# 8. Save NLP-ready base to S3
# --------------------------------------------------
(
    nlp_base.write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)

# --------------------------------------------------
# 9. Save small summary outputs locally on master
# --------------------------------------------------
subreddit_counts = (
    nlp_base.groupBy("subreddit")
    .count()
    .orderBy(F.desc("count"))
)

controversy_dist = (
    nlp_base.groupBy("controversiality")
    .count()
    .orderBy("controversiality")
)

avg_clean_length = (
    nlp_base.groupBy("subreddit")
    .agg(
        F.round(F.avg("clean_char_count"), 2).alias("avg_clean_char_count"),
        F.round(F.avg("clean_word_count"), 2).alias("avg_clean_word_count")
    )
    .orderBy(F.desc("avg_clean_word_count"))
)

subreddit_counts.toPandas().to_csv(
    f"{OUTPUT_DIR}/subreddit_counts.csv", index=False
)

controversy_dist.toPandas().to_csv(
    f"{OUTPUT_DIR}/controversiality_distribution.csv", index=False
)

avg_clean_length.toPandas().to_csv(
    f"{OUTPUT_DIR}/avg_clean_text_length_by_subreddit.csv", index=False
)

with open(f"{OUTPUT_DIR}/nlp_prep_notes.txt", "w") as f:
    f.write("NLP PREP TEXT BASE SUMMARY\n")
    f.write(f"INPUT_PATH: {INPUT_PATH}\n")
    f.write(f"OUTPUT_PATH: {OUTPUT_PATH}\n")
    f.write(f"FULL_FEATURE_BASE_ROWS: {df.count()}\n")
    f.write(f"TEXT_VALID_ROWS: {text_df.count()}\n")
    f.write(f"NLP_BASE_ROWS: {nlp_base.count()}\n")

print(f"\nSaved NLP-ready base to: {OUTPUT_PATH}")
print(f"Saved summary outputs to: {OUTPUT_DIR}")

spark.stop()