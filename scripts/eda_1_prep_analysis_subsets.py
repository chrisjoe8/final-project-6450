from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import os


MASTER_URL = "spark://172.31.21.166:7077" #MASTER_PRIVATE_IP
INPUT_PATH = "s3a://chris-joe-datsbd-s2026-v2/project/feature_base_v1/"
OUTPUT_DIR = "outputs/eda_section1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("EDA_Section1_PrepareAnalysisSubsets")
    .master(MASTER_URL)
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider"
    )
    .getOrCreate()
)

# -----------------------------
# 1. Load feature base
# -----------------------------
df = spark.read.parquet(INPUT_PATH)

# -----------------------------
# 2. Define reusable analysis subsets
# -----------------------------
# Full cleaned feature base
all_df = df

# Text-valid subset for text descriptives / later NLP
text_df = df.filter(F.col("is_removed_or_deleted") == 0)

# User-valid subset for cross-community user analysis
user_df = df.filter(F.col("is_deleted_author") == 0)

# Strict subset valid for both text and user-based analysis
text_user_df = df.filter(
    (F.col("is_removed_or_deleted") == 0) &
    (F.col("is_deleted_author") == 0)
)

# -----------------------------
# 3. Print core row counts
# -----------------------------
print("\nROW COUNTS BY ANALYSIS SUBSET")
print(f"all_df: {all_df.count()}")
print(f"text_df: {text_df.count()}")
print(f"user_df: {user_df.count()}")
print(f"text_user_df: {text_user_df.count()}")

# -----------------------------
# 4. Subreddit counts for each subset
# -----------------------------
print("\nSUBREDDIT COUNTS - ALL DATA")
all_subreddit_counts = (
    all_df.groupBy("subreddit")
    .count()
    .orderBy(F.desc("count"))
)
all_subreddit_counts.show(20, truncate=False)

print("\nSUBREDDIT COUNTS - TEXT-VALID DATA")
text_subreddit_counts = (
    text_df.groupBy("subreddit")
    .count()
    .orderBy(F.desc("count"))
)
text_subreddit_counts.show(20, truncate=False)

print("\nSUBREDDIT COUNTS - USER-VALID DATA")
user_subreddit_counts = (
    user_df.groupBy("subreddit")
    .count()
    .orderBy(F.desc("count"))
)
user_subreddit_counts.show(20, truncate=False)

print("\nSUBREDDIT COUNTS - TEXT + USER VALID DATA")
text_user_subreddit_counts = (
    text_user_df.groupBy("subreddit")
    .count()
    .orderBy(F.desc("count"))
)
text_user_subreddit_counts.show(20, truncate=False)

# -----------------------------
# 5. Controversiality distributions
# -----------------------------
def show_controversy_distribution(name, data):
    print(f"\nCONTROVERSIALITY DISTRIBUTION - {name}")
    (
        data.groupBy("controversiality")
        .count()
        .orderBy("controversiality")
        .show()
    )

show_controversy_distribution("ALL DATA", all_df)
show_controversy_distribution("TEXT-VALID DATA", text_df)
show_controversy_distribution("USER-VALID DATA", user_df)
show_controversy_distribution("TEXT + USER VALID DATA", text_user_df)

# -----------------------------
# 6. Controversy rate by subreddit in all_df
# -----------------------------
print("\nCONTROVERSY RATE BY SUBREDDIT - ALL DATA")
controversy_rate_by_subreddit = (
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

controversy_rate_by_subreddit.show(20, truncate=False)

# -----------------------------
# 7. Save small summary outputs
# -----------------------------
all_subreddit_counts.toPandas().to_csv(
    f"{OUTPUT_DIR}/all_subreddit_counts.csv", index=False
)

text_subreddit_counts.toPandas().to_csv(
    f"{OUTPUT_DIR}/text_subreddit_counts.csv", index=False
)

user_subreddit_counts.toPandas().to_csv(
    f"{OUTPUT_DIR}/user_subreddit_counts.csv", index=False
)

text_user_subreddit_counts.toPandas().to_csv(
    f"{OUTPUT_DIR}/text_user_subreddit_counts.csv", index=False
)

controversy_rate_by_subreddit.toPandas().to_csv(
    f"{OUTPUT_DIR}/controversy_rate_by_subreddit.csv", index=False
)

# Save subset sizes as a small text file
with open(f"{OUTPUT_DIR}/subset_row_counts.txt", "w") as f:
    f.write("ROW COUNTS BY ANALYSIS SUBSET\n")
    f.write(f"all_df: {all_df.count()}\n")
    f.write(f"text_df: {text_df.count()}\n")
    f.write(f"user_df: {user_df.count()}\n")
    f.write(f"text_user_df: {text_user_df.count()}\n")

print(f"\nSaved summary outputs to: {OUTPUT_DIR}")

spark.stop()