# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Bronze: raw data with Auto Loader
# MAGIC Loads both Lending Club files into Delta tables using Auto Loader. Auto Loader treats each
# MAGIC landing folder as a stream: it processes only files it hasn't seen before and remembers its
# MAGIC progress in a checkpoint, so new files added later are picked up without reloading everything.
# MAGIC
# MAGIC Bronze keeps every value as text, exactly as it arrived. Types are fixed in silver.

# COMMAND ----------

import re
from pyspark.sql import functions as F

CATALOG = "workspace"
SCHEMA = "lending_club"
RAW = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
CHECKPOINTS = f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints"


def clean_name(name):
    """Delta column names can't contain spaces or symbols, e.g. 'Debt-To-Income Ratio' -> 'debt_to_income_ratio'."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def load_bronze(source):
    landing = f"{RAW}/landing/{source}"
    target = f"{CATALOG}.{SCHEMA}.bronze_{source}"
    stream = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.schemaLocation", f"{CHECKPOINTS}/bronze_{source}/schema")
        .option("cloudFiles.inferColumnTypes", "false")   # keep everything as text in bronze
        .option("header", "true")
        .option("multiLine", "true")                        # some text fields contain line breaks
        .option("escape", '"')
        .load(landing)
    )
    stream = stream.select([F.col(f"`{c}`").alias(clean_name(c)) for c in stream.columns])
    stream = (
        stream
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_ingested_at", F.current_timestamp())
    )
    query = (
        stream.writeStream
        .option("checkpointLocation", f"{CHECKPOINTS}/bronze_{source}/checkpoint")
        .trigger(availableNow=True)                         # process everything available, then stop
        .toTable(target)
    )
    query.awaitTermination()
    print(f"{target}: {spark.table(target).count():,} rows")

# COMMAND ----------

load_bronze("accepted")

# COMMAND ----------

load_bronze("rejected")
