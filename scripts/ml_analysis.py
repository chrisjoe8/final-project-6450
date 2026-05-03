from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import os
import pandas as pd

MASTER_URL = "spark://172.31.95.81:7077" # MASTER_PRIVATE_IP
INPUT_PATH = "s3a://chris-joe-datsbd-s2026-v2/project/ml_model_base_v1/"
OUTPUT_DIR = "outputs/ml_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("ML_Analysis_LogReg_RF")
    .master(MASTER_URL)
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider"
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# --------------------------------------------------
# 1. Load ML-ready base
# --------------------------------------------------
df = spark.read.parquet(INPUT_PATH)

print("\nML BASE ROW COUNT")
print(df.count())

print("\nLABEL DISTRIBUTION")
df.groupBy("label").count().orderBy("label").show()

# --------------------------------------------------
# 2. Keep modeling columns only
# --------------------------------------------------
model_df = df.select(
    "label",
    "subreddit",
    "score_rounded",
    "month",
    "hour",
    "weekday",
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
# 3. Train / test split
# --------------------------------------------------
train_df, test_df = model_df.randomSplit([0.8, 0.2], seed=42)

print("\nTRAIN / TEST ROW COUNTS")
print(f"train_df: {train_df.count()}")
print(f"test_df:  {test_df.count()}")

# --------------------------------------------------
# 4. Class weighting for imbalanced logistic regression
# --------------------------------------------------
label_counts = train_df.groupBy("label").count().collect()
label_count_dict = {row["label"]: row["count"] for row in label_counts}

neg_count = label_count_dict.get(0, 1)
pos_count = label_count_dict.get(1, 1)
total_count = neg_count + pos_count

neg_weight = total_count / (2 * neg_count)
pos_weight = total_count / (2 * pos_count)

train_df = train_df.withColumn(
    "classWeightCol",
    F.when(F.col("label") == 1, F.lit(pos_weight)).otherwise(F.lit(neg_weight))
)

test_df = test_df.withColumn("classWeightCol", F.lit(1.0))

print("\nCLASS WEIGHTS")
print(f"negative_class_weight: {round(float(neg_weight), 4)}")
print(f"positive_class_weight: {round(float(pos_weight), 4)}")

# --------------------------------------------------
# 5. Shared preprocessing
# --------------------------------------------------
categorical_cols = ["subreddit"]
numeric_cols = [
    "score_rounded",
    "month",
    "hour",
    "weekday",
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
]

indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in categorical_cols
]

encoders = [
    OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
    for c in categorical_cols
]

assembler = VectorAssembler(
    inputCols=[f"{c}_ohe" for c in categorical_cols] + numeric_cols,
    outputCol="features",
    handleInvalid="keep"
)

# --------------------------------------------------
# 6. Models
# --------------------------------------------------
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeightCol",
    maxIter=30,
    regParam=0.01,
    elasticNetParam=0.0
)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=50,
    maxDepth=8,
    seed=42
)

lr_pipeline = Pipeline(stages=indexers + encoders + [assembler, lr])
rf_pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

# --------------------------------------------------
# 7. Fit models
# --------------------------------------------------
print("\nFITTING LOGISTIC REGRESSION...")
lr_model = lr_pipeline.fit(train_df)

print("\nFITTING RANDOM FOREST...")
rf_model = rf_pipeline.fit(train_df)

# --------------------------------------------------
# 8. Predictions
# --------------------------------------------------
lr_preds = lr_model.transform(test_df)
rf_preds = rf_model.transform(test_df)

# --------------------------------------------------
# 9. Evaluation helper
# --------------------------------------------------
binary_eval = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

accuracy_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

f1_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

weighted_precision_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

weighted_recall_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)

def get_metrics(pred_df, model_name):
    auc = round(float(binary_eval.evaluate(pred_df)), 4)
    accuracy = round(float(accuracy_eval.evaluate(pred_df)), 4)
    f1 = round(float(f1_eval.evaluate(pred_df)), 4)
    weighted_precision = round(float(weighted_precision_eval.evaluate(pred_df)), 4)
    weighted_recall = round(float(weighted_recall_eval.evaluate(pred_df)), 4)

    confusion = (
        pred_df.groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
        .toPandas()
    )

    metrics = {
        "model": model_name,
        "auc_roc": auc,
        "accuracy": accuracy,
        "f1": f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall
    }
    return metrics, confusion

lr_metrics, lr_confusion = get_metrics(lr_preds, "Logistic Regression")
rf_metrics, rf_confusion = get_metrics(rf_preds, "Random Forest")

metrics_df = pd.DataFrame([lr_metrics, rf_metrics])

print("\nMODEL METRICS")
print(metrics_df.to_string(index=False))

print("\nLOGISTIC REGRESSION CONFUSION COUNTS")
print(lr_confusion.to_string(index=False))

print("\nRANDOM FOREST CONFUSION COUNTS")
print(rf_confusion.to_string(index=False))

# --------------------------------------------------
# 10. Feature importance / interpretability
# --------------------------------------------------
# Because one-hot encoding expands subreddit into multiple columns,
# the VectorAssembler input names do not align 1-to-1 with the final
# model coefficient/importances vector length. To keep this robust,
# we create generic feature indices for export.

lr_stage = lr_model.stages[-1]
lr_coeffs = lr_stage.coefficients.toArray().tolist()

lr_feature_importance = pd.DataFrame({
    "feature_index": list(range(len(lr_coeffs))),
    "coefficient": lr_coeffs
})
lr_feature_importance["abs_coefficient"] = lr_feature_importance["coefficient"].abs()
lr_feature_importance = lr_feature_importance.sort_values(
    by="abs_coefficient",
    ascending=False
).reset_index(drop=True)

print("\nTOP 20 LOGISTIC REGRESSION COEFFICIENTS")
print(
    lr_feature_importance[["feature_index", "coefficient"]]
    .head(20)
    .round(4)
    .to_string(index=False)
)

rf_stage = rf_model.stages[-1]
rf_importances = rf_stage.featureImportances.toArray().tolist()

rf_feature_importance = pd.DataFrame({
    "feature_index": list(range(len(rf_importances))),
    "importance": rf_importances
})
rf_feature_importance = rf_feature_importance.sort_values(
    by="importance",
    ascending=False
).reset_index(drop=True)

print("\nTOP 20 RANDOM FOREST FEATURE IMPORTANCES")
print(
    rf_feature_importance.head(20)
    .round(4)
    .to_string(index=False)
)

# --------------------------------------------------
# 11. Save outputs
# --------------------------------------------------
metrics_df.to_csv(f"{OUTPUT_DIR}/model_metrics.csv", index=False)
lr_confusion.to_csv(f"{OUTPUT_DIR}/logreg_confusion_counts.csv", index=False)
rf_confusion.to_csv(f"{OUTPUT_DIR}/rf_confusion_counts.csv", index=False)
lr_feature_importance.to_csv(f"{OUTPUT_DIR}/logreg_feature_coefficients.csv", index=False)
rf_feature_importance.to_csv(f"{OUTPUT_DIR}/rf_feature_importances.csv", index=False)

with open(f"{OUTPUT_DIR}/ml_analysis_notes.txt", "w") as f:
    f.write("ML ANALYSIS SUMMARY\n")
    f.write("Models run:\n")
    f.write("- Logistic Regression\n")
    f.write("- Random Forest\n\n")
    f.write("Metrics:\n")
    for _, row in metrics_df.iterrows():
        f.write(
            f"{row['model']}: "
            f"AUC={row['auc_roc']}, "
            f"Accuracy={row['accuracy']}, "
            f"F1={row['f1']}, "
            f"Weighted Precision={row['weighted_precision']}, "
            f"Weighted Recall={row['weighted_recall']}\n"
        )

print(f"\nSaved summary outputs to: {OUTPUT_DIR}")

spark.stop()