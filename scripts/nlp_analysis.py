from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StopWordsRemover
import os

MASTER_URL = "spark://172.31.86.233:7077"
INPUT_PATH = "s3a://chris-joe-datsbd-s2026-v2/project/nlp_text_base_v1/"
OUTPUT_DIR = "outputs/nlp_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("NLP_Analysis")
    .master(MASTER_URL)
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider"
    )
    .getOrCreate()
)

# --------------------------------------------------
# 1. Load NLP-ready base
# --------------------------------------------------
df = spark.read.parquet(INPUT_PATH)

print("\nNLP BASE ROW COUNT")
print(df.count())

# --------------------------------------------------
# 2. Tokenize cleaned text
# --------------------------------------------------
token_df = (
    df.withColumn("tokens", F.split(F.col("body_clean"), r"\s+"))
)

remover = StopWordsRemover(inputCol="tokens", outputCol="tokens_nostop")
token_df = remover.transform(token_df)

tokens_exploded = (
    token_df
    .withColumn("token", F.explode("tokens_nostop"))
    .filter(F.length("token") >= 3)
    .filter(~F.col("token").rlike(r"^\d+$"))
)

print("\nTOKENIZED ROW COUNT")
print(tokens_exploded.count())

# --------------------------------------------------
# 3. Very simple lexicon-based sentiment
#    (kept intentionally straightforward/interpretable)
# --------------------------------------------------
positive_words = [
    "good", "great", "better", "best", "right", "love", "support", "peace",
    "safe", "success", "fair", "hope", "help", "agree", "freedom", "positive"
]

negative_words = [
    "bad", "worse", "worst", "wrong", "hate", "war", "dead", "death",
    "kill", "racist", "corrupt", "stupid", "evil", "lie", "lies", "violent",
    "violence", "terror", "terrorist", "crime", "illegal", "attack", "attacks"
]

positive_set = set(positive_words)
negative_set = set(negative_words)

sentiment_token_counts = (
    tokens_exploded
    .withColumn(
        "is_positive_token",
        F.when(F.col("token").isin(list(positive_set)), 1).otherwise(0)
    )
    .withColumn(
        "is_negative_token",
        F.when(F.col("token").isin(list(negative_set)), 1).otherwise(0)
    )
)

comment_sentiment = (
    sentiment_token_counts.groupBy(
        "id", "author", "subreddit", "controversiality", "score",
        "year", "month", "hour", "weekday"
    )
    .agg(
        F.sum("is_positive_token").alias("positive_token_count"),
        F.sum("is_negative_token").alias("negative_token_count"),
        F.count("*").alias("token_count")
    )
    .withColumn(
        "sentiment_score",
        F.col("positive_token_count") - F.col("negative_token_count")
    )
    .withColumn(
        "sentiment_score_per_token",
        F.round(
            F.when(F.col("token_count") > 0,
                   F.col("sentiment_score") / F.col("token_count"))
             .otherwise(0.0),
            4
        )
    )
)

# --------------------------------------------------
# 4. Sentiment by controversiality
# --------------------------------------------------
sentiment_by_controversy = (
    comment_sentiment.groupBy("controversiality")
    .agg(
        F.count("*").alias("total_comments"),
        F.round(F.avg("positive_token_count"), 2).alias("avg_positive_tokens"),
        F.round(F.avg("negative_token_count"), 2).alias("avg_negative_tokens"),
        F.round(F.avg("sentiment_score"), 2).alias("avg_sentiment_score"),
        F.round(F.avg("sentiment_score_per_token"), 4).alias("avg_sentiment_score_per_token")
    )
    .orderBy("controversiality")
)

print("\nSENTIMENT BY CONTROVERSIALITY")
sentiment_by_controversy.show(truncate=False)

# --------------------------------------------------
# 5. Sentiment by subreddit
# --------------------------------------------------
sentiment_by_subreddit = (
    comment_sentiment.groupBy("subreddit")
    .agg(
        F.count("*").alias("total_comments"),
        F.round(F.avg("positive_token_count"), 2).alias("avg_positive_tokens"),
        F.round(F.avg("negative_token_count"), 2).alias("avg_negative_tokens"),
        F.round(F.avg("sentiment_score"), 2).alias("avg_sentiment_score"),
        F.round(F.avg("sentiment_score_per_token"), 4).alias("avg_sentiment_score_per_token")
    )
    .orderBy(F.desc("total_comments"))
)

print("\nSENTIMENT BY SUBREDDIT")
sentiment_by_subreddit.show(20, truncate=False)

# --------------------------------------------------
# 6. Distinctive terms by controversiality
#    Use relative frequency + lift vs overall frequency
# --------------------------------------------------
overall_term_counts = (
    tokens_exploded.groupBy("token")
    .agg(F.count("*").alias("overall_count"))
)

overall_total_tokens = tokens_exploded.count()

overall_term_counts = overall_term_counts.withColumn(
    "overall_share",
    F.col("overall_count") / F.lit(overall_total_tokens)
)

controversy_term_counts = (
    tokens_exploded.groupBy("controversiality", "token")
    .agg(F.count("*").alias("group_count"))
)

controversy_totals = (
    tokens_exploded.groupBy("controversiality")
    .agg(F.count("*").alias("group_total_tokens"))
)

distinctive_terms_by_controversy = (
    controversy_term_counts
    .join(controversy_totals, on="controversiality", how="left")
    .join(overall_term_counts.select("token", "overall_share"), on="token", how="left")
    .withColumn("group_share", F.col("group_count") / F.col("group_total_tokens"))
    .withColumn("lift", F.round(F.col("group_share") / F.col("overall_share"), 4))
    .filter(F.col("group_count") >= 500)
)

w1 = Window.partitionBy("controversiality").orderBy(F.desc("lift"), F.desc("group_count"), F.asc("token"))

top_distinctive_terms_by_controversy = (
    distinctive_terms_by_controversy
    .withColumn("rank", F.row_number().over(w1))
    .filter(F.col("rank") <= 20)
    .select(
        "controversiality", "rank", "token",
        F.round("group_share", 6).alias("group_share"),
        "group_count", "lift"
    )
    .orderBy("controversiality", "rank")
)

print("\nTOP DISTINCTIVE TERMS BY CONTROVERSIALITY")
top_distinctive_terms_by_controversy.show(50, truncate=False)

# --------------------------------------------------
# 7. Distinctive terms by subreddit
# --------------------------------------------------
subreddit_term_counts = (
    tokens_exploded.groupBy("subreddit", "token")
    .agg(F.count("*").alias("group_count"))
)

subreddit_totals = (
    tokens_exploded.groupBy("subreddit")
    .agg(F.count("*").alias("group_total_tokens"))
)

distinctive_terms_by_subreddit = (
    subreddit_term_counts
    .join(subreddit_totals, on="subreddit", how="left")
    .join(overall_term_counts.select("token", "overall_share"), on="token", how="left")
    .withColumn("group_share", F.col("group_count") / F.col("group_total_tokens"))
    .withColumn("lift", F.round(F.col("group_share") / F.col("overall_share"), 4))
    .filter(F.col("group_count") >= 500)
)

w2 = Window.partitionBy("subreddit").orderBy(F.desc("lift"), F.desc("group_count"), F.asc("token"))

top_distinctive_terms_by_subreddit = (
    distinctive_terms_by_subreddit
    .withColumn("rank", F.row_number().over(w2))
    .filter(F.col("rank") <= 20)
    .select(
        "subreddit", "rank", "token",
        F.round("group_share", 6).alias("group_share"),
        "group_count", "lift"
    )
    .orderBy("subreddit", "rank")
)

print("\nTOP DISTINCTIVE TERMS BY SUBREDDIT")
top_distinctive_terms_by_subreddit.show(150, truncate=False)

# --------------------------------------------------
# 8. Save outputs
# --------------------------------------------------
sentiment_by_controversy.toPandas().to_csv(
    f"{OUTPUT_DIR}/sentiment_by_controversy.csv", index=False
)

sentiment_by_subreddit.toPandas().to_csv(
    f"{OUTPUT_DIR}/sentiment_by_subreddit.csv", index=False
)

top_distinctive_terms_by_controversy.toPandas().to_csv(
    f"{OUTPUT_DIR}/top_distinctive_terms_by_controversy.csv", index=False
)

top_distinctive_terms_by_subreddit.toPandas().to_csv(
    f"{OUTPUT_DIR}/top_distinctive_terms_by_subreddit.csv", index=False
)

with open(f"{OUTPUT_DIR}/nlp_analysis_notes.txt", "w") as f:
    f.write("NLP ANALYSIS SUMMARY\n")
    f.write("Outputs saved:\n")
    f.write("- sentiment_by_controversy.csv\n")
    f.write("- sentiment_by_subreddit.csv\n")
    f.write("- top_distinctive_terms_by_controversy.csv\n")
    f.write("- top_distinctive_terms_by_subreddit.csv\n")

print(f"\nSaved summary outputs to: {OUTPUT_DIR}")

spark.stop()